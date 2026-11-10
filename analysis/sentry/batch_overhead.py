#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Does the range check's cost amortize over the batch?

WeightSentry reports 1.1% GPU overhead at batch 64 and 2.8% at batch 16. That
scaling is not a coincidence: the check runs ONCE PER CONV PER FORWARD CALL,
over the weight tensor, whose size does not depend on how many images are in
flight. So the check's ABSOLUTE cost should be flat in batch while the baseline
forward grows linearly -- overhead% ~ 1/batch. Batch 1, the latency-critical
single-camera case, pays the whole bill. This script measures that curve.

Two caveats on what the numbers mean, both of which make these an UPPER BOUND:

  * Bounds stores lo/hi/emax EXPANDED to full tensor shape so the check
    vectorises, so a signed check streams 3x12 MB (w, lo, hi) where a deployed
    one streams 12 MB of weights plus a 44 KB bound table. The memory-traffic
    component is inflated roughly 3x.

  * in_range() is several eager elementwise ops, each its own CUDA kernel, so
    the check adds ~5 launches x 64 convs per forward. Launch overhead is
    batch-independent, exactly like the weight read, so it does not distort the
    SHAPE of the curve -- but it inflates the magnitude versus a fused kernel.

Comparing `exp 1-sided` against `WS signed` discriminates the two mechanisms.
Exponent-only reads LESS (w + emax = 24 MB vs w + lo + hi = 36 MB) but executes
MORE (shift, mask, compare, isfinite vs two float compares). If overhead tracks
bytes, exp-only wins; if it tracks instructions or launches, it loses.

Usage:  ./batch_overhead.py [batches] [iters]
        ./batch_overhead.py 1,2,4,8,16,32,64 30
"""
import logging, os, sys, time
import cv2
import torch

logging.getLogger("ultralytics").setLevel(logging.ERROR)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src")
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import fi_lib, weightSentry

BATCHES = [int(b) for b in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 4, 8, 16, 32, 64]
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
WARMUP = 10
DATA = "coco-val2017-sub127.yaml"

model = YOLO(f"{ROOT}/yolov8n.pt")
model.model.fuse()
core = model.model.cuda().eval()

CONFIGS = [
    ("undefended",           None),
    ("WS signed per-chan",   dict(mode="chan", signed=True)),
    ("WS signed K=128",      dict(mode="k", K=128, signed=True)),
    ("exp 1-sided per-chan", dict(mode="chan", signed=False, two_sided=False)),
    ("exp 2-sided per-chan", dict(mode="chan", signed=False, two_sided=True)),
]
BOUNDS = [(l, weightSentry.Bounds(core, **kw) if kw else None) for l, kw in CONFIGS]

nconv = sum(1 for _, m in core.named_modules() if isinstance(m, torch.nn.Conv2d))
nw = sum(m.weight.numel() for _, m in core.named_modules() if isinstance(m, torch.nn.Conv2d))
nb = sum(m.bias.numel() for _, m in core.named_modules()
         if isinstance(m, torch.nn.Conv2d) and m.bias is not None)

print(f"RTX 4070 / yolov8n fused: {nconv} convs, {nw:,} weights ({nw*4/2**20:.2f} MB), "
      f"{nb:,} biases ({nb*4/1024:.2f} KB)")
print(f"{ITERS} timed iterations after {WARMUP} warmup, forward pass only (no NMS)\n")

# ---- deployed metadata cost, recomputed from current code ----------------
print(f"{'config':<22}{'units':>9}{'total KB':>10}{'bias KB':>9}{'sim bytes read/fwd':>21}")
for lbl, b in BOUNDS:
    if b is None:
        print(f"{lbl:<22}{'-':>9}{0.0:>10.2f}{'-':>9}{nw*4/2**20:>18.1f} MB")
        continue
    # what the vectorised check actually streams per forward, upper bound
    tabs = 2 if b.signed else (2 if b.two_sided else 1)
    sim = (nw + nb) * 4 * (1 + tabs)
    print(f"{lbl:<22}{b.n_units:>9,}{b.bytes/1024:>10.2f}{b.bias_bytes/1024:>9.2f}"
          f"{sim/2**20:>18.1f} MB")
print()

# ---- build real-image batches -------------------------------------------
lb = LetterBox((640, 640))
paths = fi_lib.dataset_images(DATA, 8)
ims = []
for p in paths:
    im = lb(image=cv2.imread(p))
    ims.append(torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255))
base = torch.stack(ims)


def bench(x, iters):
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            core(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


print(f"{'batch':>5} {'config':<22}{'ms/fwd':>9}{'ms/img':>9}{'+ms':>8}{'overhead':>10}")
rows = []
for B in BATCHES:
    reps = (B + base.shape[0] - 1) // base.shape[0]
    try:
        x = base.repeat(reps, 1, 1, 1)[:B].cuda().contiguous()
    except RuntimeError as e:
        print(f"{B:>5} skipped: {str(e).splitlines()[0][:60]}")
        continue
    ref = None
    for lbl, b in BOUNDS:
        h = None
        if b is not None:
            h, _ = weightSentry.attach(core, b, commit=True, golden=None)
        try:
            bench(x, WARMUP)
            ms = bench(x, ITERS)
        except RuntimeError as e:
            if h: weightSentry.detach(h)
            print(f"{B:>5} {lbl:<22} OOM/{str(e).splitlines()[0][:40]}")
            break
        if h:
            weightSentry.detach(h)
        if ref is None:
            ref = ms
            print(f"{B:>5} {lbl:<22}{ms:>9.3f}{ms/B:>9.3f}{'-':>8}{'-':>10}")
        else:
            print(f"{B:>5} {lbl:<22}{ms:>9.3f}{ms/B:>9.3f}{ms-ref:>8.3f}"
                  f"{100*(ms-ref)/ref:>9.2f}%")
        rows.append(dict(batch=B, config=lbl, ms_fwd=round(ms, 4),
                         ms_img=round(ms / B, 4),
                         delta_ms=round(ms - ref, 4) if ref else 0.0,
                         overhead_pct=round(100 * (ms - ref) / ref, 3) if ref else 0.0,
                         meta_kb=round(b.bytes / 1024, 2) if b else 0.0))
    del x
    torch.cuda.empty_cache()
    print(flush=True)

# ---- cost of a repair actually firing -----------------------------------
# The site must be LOW magnitude. A bit-30 flip on |w| >= 2 CLEARS the already-set
# exponent MSB, collapsing the weight to ~1e-40, which a signed [min,max] bound
# brackets and therefore never flags -- see coverage_census.py. Picking such a
# site would time the no-fault path and report it as the repair path.
print("repair path: one detectable bit-30 flip, batch 1, commit=False")
big = max(((n, m) for n, m in core.named_modules() if isinstance(m, torch.nn.Conv2d)),
          key=lambda nm: nm[1].weight.numel())
x1 = base[:1].cuda().contiguous()
for lbl, b in BOUNDS[1:]:
    name, conv = big
    gold = conv.weight.data.clone()
    flat = conv.weight.data.view(-1)
    idx = int((flat.abs() < 2).nonzero()[0])          # low magnitude: flip explodes
    h, _ = weightSentry.attach(core, b, commit=False, golden=None)
    bench(x1, WARMUP); clean = bench(x1, ITERS)
    weightSentry.detach(h)
    flat[idx] = fi_lib.flip(flat[idx], 30)
    h, st = weightSentry.attach(core, b, commit=False, golden=None)
    bench(x1, WARMUP); dirty = bench(x1, ITERS)
    weightSentry.detach(h)
    conv.weight.data.copy_(gold)
    tag = "" if st.flagged else "  <-- NOT DETECTED, timing is the no-fault path"
    print(f"  {lbl:<22} clean {clean:7.3f} ms  1 fault {dirty:7.3f} ms  "
          f"delta {dirty-clean:+6.3f} ms ({100*(dirty-clean)/clean:+5.2f}%)  "
          f"flagged/fwd={st.flagged/(WARMUP+ITERS):.2f}{tag}")
print(f"  (site: {big[0]}, {big[1].weight.numel():,} weights, "
      f"{100*big[1].weight.numel()/nw:.1f}% of the model)")

if rows:
    import csv as _csv
    out = f"{ROOT}/results/sentry_batch_overhead.csv"
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out}")
