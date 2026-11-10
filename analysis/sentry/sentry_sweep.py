#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Defended vs undefended injection campaign: does the range check actually
retire the risk the census says it should?

Every configuration sees the IDENTICAL injection sites, drawn once per bit from
a fixed seed, so a difference between two rows is attributable to the defence
and nothing else -- the same discipline campaign.py uses across bit positions.

`kind` selects the fault site population, and the choice is STRATIFIED on
purpose. Biases are 5,728 of 3,151,904 parameters (0.18%), so sampling "both"
uniformly draws a bias roughly once in 550 injections and would tell you
nothing about them. Sweeping `weight` and `bias` separately measures each
population's per-fault behaviour; weight them by 0.9982/0.0018 afterwards if a
combined random-fault number is wanted.

Verdicts come from fi_lib.outcome, so SDC/DUE mean exactly what they mean in
every other campaign here. mAP is skipped on purpose: it is slow, and a coarse
threshold on it misses detection-level corruption, which is the whole point.

Usage:  ./sentry_sweep.py [samples_per_bit] [n_images] [weight|bias|both] [bits...]
"""
import csv, logging, os, random, sys, time
import torch

logging.getLogger("ultralytics").setLevel(logging.ERROR)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src")
from ultralytics import YOLO
import fi_lib, weightSentry

SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 48
NIMG = int(sys.argv[2]) if len(sys.argv) > 2 else 24
KIND = sys.argv[3] if len(sys.argv) > 3 else "bias"
BITS = [int(b) for b in sys.argv[4:]] or [31, 30, 29, 27, 25, 23]
DATA = "coco-val2017-sub127.yaml"
SEED = 0

model = YOLO(f"{ROOT}/yolov8n.pt")
model.model.fuse()
core = model.model.cuda().eval()
detect = fi_lib.make_detector(core)
golden_w = weightSentry.snapshot(core)
convs = [(n, m) for n, m in core.named_modules() if isinstance(m, torch.nn.Conv2d)]

# site pool: (layer index, is_bias, tensor numel), sampled proportional to size
pool = []
for li, (n, m) in enumerate(convs):
    if KIND in ("weight", "both"):
        pool.append((li, False, m.weight.numel()))
    if KIND in ("bias", "both") and m.bias is not None:
        pool.append((li, True, m.bias.numel()))
pool_total = sum(p[2] for p in pool)

CONFIGS = [
    ("undefended", None),
    ("WS signed per-chan", dict(mode="chan", signed=True)),
    ("WS signed K=128", dict(mode="k", K=128, signed=True)),
    ("exp 2-sided per-chan", dict(mode="chan", signed=False, two_sided=True)),
]
BOUNDS = [(lbl, weightSentry.Bounds(core, **kw) if kw else None) for lbl, kw in CONFIGS]

imgs = fi_lib.dataset_images(DATA, NIMG)
gold = {p: detect(p)[0] for p in imgs}
print(f"{len(imgs)} images, {sum(len(v) for v in gold.values())} golden detections")
print(f"kind={KIND}  site pool {pool_total:,} values across {len(pool)} tensors")
for lbl, b in BOUNDS:
    cost = "-" if b is None else (f"{b.bytes/1024:8.2f} KB total, "
                                  f"{b.bias_bytes/1024:6.2f} KB of it bias")
    print(f"  {lbl:22s} {cost}")
print(f"bits={BITS} samples/bit={SAMPLES}\n", flush=True)


def run_once():
    agg = {}
    for p in imgs:
        dets, nn, ni = detect(p)
        v = fi_lib.outcome(gold[p], dets, nn, ni)[0]
        agg[v] = agg.get(v, 0) + 1
    return agg


rows, t0 = [], time.time()
print(f"{'bit':>4} {'config':<22} {'masked':>7}{'benign':>7}{'SDC':>6}{'DUE':>6}"
      f"{'SDC%':>7}  {'flagged':>8}{'exact':>7}{'false':>7}")
for bit in BITS:
    for lbl, b in BOUNDS:
        rng = random.Random(SEED + bit)     # identical sites for every config
        agg, tot_st = {}, [0, 0, 0]
        for _ in range(SAMPLES):
            r = rng.randrange(pool_total)
            pi = 0
            while r >= pool[pi][2]:
                r -= pool[pi][2]; pi += 1
            li, is_bias, _ = pool[pi]
            name, layer = convs[li]
            key = name + "#bias" if is_bias else name
            t = (layer.bias if is_bias else layer.weight).data
            t.view(-1)[r] = fi_lib.flip(t.view(-1)[r], bit)
            h = st = None
            if b is not None:
                h, st = weightSentry.attach(core, b, commit=True, golden=golden_w)
            a = run_once()
            if h is not None:
                weightSentry.detach(h)
                tot_st[0] += st.flagged + st.b_flagged
                tot_st[1] += st.exact + st.b_exact
                tot_st[2] += st.false + st.b_false
            t.copy_(golden_w[key])          # full restore
            for k, v in a.items():
                agg[k] = agg.get(k, 0) + v
        g = lambda k: agg.get(k, 0)
        n = sum(agg.values())
        sdcp = 100.0 * g("SDC") / n if n else 0.0
        print(f"{bit:>4} {lbl:<22} {g('masked'):>7}{g('benign'):>7}{g('SDC'):>6}"
              f"{g('DUE'):>6}{sdcp:>6.2f}%  {tot_st[0]:>8}{tot_st[1]:>7}{tot_st[2]:>7}",
              flush=True)
        rows.append(dict(bit=bit, kind=KIND, config=lbl, samples=SAMPLES,
                         images=len(imgs), masked=g("masked"), benign=g("benign"),
                         sdc=g("SDC"), due=g("DUE"), sdc_pct=round(sdcp, 4),
                         flagged=tot_st[0], exact=tot_st[1], false=tot_st[2],
                         meta_kb=round(b.bytes / 1024, 2) if b else 0))
    print(flush=True)

out = f"{ROOT}/results/sentry_{KIND}_n{SAMPLES}_img{NIMG}.csv"
os.makedirs(f"{ROOT}/results", exist_ok=True)
with open(out, "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wtr.writeheader(); wtr.writerows(rows)
print(f"{len(rows)} rows -> {out}   ({time.time()-t0:.0f}s)")
