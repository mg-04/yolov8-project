#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Exhaustive single-flip DETECTION census, stratified by magnitude.

sentry_sweep.py samples ~48 sites per bit and measures SDC, which is the number
that matters but carries sampling error and conflates detection with the
network's own tolerance. This script instead enumerates EVERY parameter and asks
only "would the check flag this flip?" -- no inference, no sampling, exact.

The stratification is the point. A bit-30 flip does opposite things to the two
magnitude populations:

    |w| < 2   exponent MSB is 0 -> flip SETS it   -> value explodes (~1e19)
    |w| >= 2  exponent MSB is 1 -> flip CLEARS it -> value collapses (~1e-40)

A signed [min,max] bound brackets zero, so it catches the explosion and is
structurally BLIND to the collapse. Since biases hold 69.4% of the high-magnitude
population, that blindness is concentrated exactly where the damage is.

Usage:  ./coverage_census.py
"""
import logging, os, sys
import torch

logging.getLogger("ultralytics").setLevel(logging.ERROR)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src")
from ultralytics import YOLO
import weightSentry

BITS = list(range(31, 22, -1))
model = YOLO(f"{ROOT}/yolov8n.pt")
model.model.fuse()
core = model.model.cuda().eval()

CONFIGS = [
    ("WS signed per-chan",   dict(mode="chan", signed=True)),
    ("WS signed K=128",      dict(mode="k", K=128, signed=True)),
    ("exp 1-sided per-chan", dict(mode="chan", signed=False, two_sided=False)),
    ("exp 2-sided per-chan", dict(mode="chan", signed=False, two_sided=True)),
]

convs = [(n, m) for n, m in core.named_modules() if isinstance(m, torch.nn.Conv2d)]
# (bounds key, tensor, kind) for every protected tensor
tensors = []
for n, m in convs:
    tensors.append((n, m.weight.data, "weight"))
    if m.bias is not None:
        tensors.append((n + "#bias", m.bias.data, "bias"))

tot = {k: 0 for k in ("weight", "bias")}
hi_tot = {k: 0 for k in ("weight", "bias")}
for _, t, kind in tensors:
    tot[kind] += t.numel()
    hi_tot[kind] += int((t.abs() >= 2).sum())
print(f"population: weights {tot['weight']:,} ({hi_tot['weight']:,} with |x|>=2), "
      f"biases {tot['bias']:,} ({hi_tot['bias']:,} with |x|>=2)")
print("every cell below is an EXHAUSTIVE count over that population, not a sample\n")

rows = []
for lbl, kw in CONFIGS:
    b = weightSentry.Bounds(core, **kw)
    print(f"{lbl}   ({b.bytes/1024:.2f} KB metadata, {b.n_units:,} units)")
    print(f"{'bit':>4} | {'weight all':>11} {'w |x|<2':>9} {'w |x|>=2':>9} "
          f"| {'bias all':>9} {'b |x|<2':>9} {'b |x|>=2':>9}")
    for bit in BITS:
        # det[kind][stratum] = (detected, total)
        acc = {("weight", "lo"): [0, 0], ("weight", "hi"): [0, 0],
               ("bias", "lo"): [0, 0], ("bias", "hi"): [0, 0]}
        for key, t, kind in tensors:
            flipped = (t.view(torch.int32) ^ (1 << bit)).view(torch.float32)
            # elementwise: each element judged against its own unit bound, so
            # this is exactly "flip this one element" evaluated in parallel
            det = ~b.in_range(key, flipped)
            high = t.abs() >= 2
            for strat, mask in (("hi", high), ("lo", ~high)):
                a = acc[(kind, strat)]
                a[0] += int((det & mask).sum())
                a[1] += int(mask.sum())
        def pct(kind, strat):
            d, n = acc[(kind, strat)]
            return f"{100.0*d/n:8.2f}%" if n else "       -"
        def pctall(kind):
            d = acc[(kind, "lo")][0] + acc[(kind, "hi")][0]
            n = acc[(kind, "lo")][1] + acc[(kind, "hi")][1]
            return f"{100.0*d/n:10.2f}%" if n else "         -"
        print(f"{bit:>4} | {pctall('weight')} {pct('weight','lo')} {pct('weight','hi')} "
              f"| {pctall('bias')[2:]} {pct('bias','lo')} {pct('bias','hi')}")
        for kind in ("weight", "bias"):
            for strat in ("lo", "hi"):
                d, n = acc[(kind, strat)]
                rows.append(dict(config=lbl, meta_kb=round(b.bytes/1024, 2), bit=bit,
                                 kind=kind, stratum={"lo": "|x|<2", "hi": "|x|>=2"}[strat],
                                 detected=d, total=n,
                                 det_pct=round(100.0*d/n, 4) if n else 0.0))
    print(flush=True)
    del b
    torch.cuda.empty_cache()

import csv
out = f"{ROOT}/results/sentry_coverage_census.csv"
os.makedirs(f"{ROOT}/results", exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {out}")
