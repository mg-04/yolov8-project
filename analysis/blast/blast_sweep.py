#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Sweep A: blast radius.

How much of the Detect output does one fault change? Purely structural: no NMS,
no golden detections, no mAP. One extra forward pass per injection, so ~20 ms
against ~1.6 s for the full detection pipeline.

The output is 84 x 8400, where the 8,400 are candidate positions across three
scales (P3 80x80 = 6,400, P4 40x40 = 1,600, P5 20x20 = 400). An anchor counts as
changed if any of its 84 values differs from golden by more than THRESH, or
became NaN/Inf.

Because every anchor has a known grid position, changed anchors have known pixel
locations. For activation faults, which are injected at a known (fy, fx), that
gives the spatial spread of the fault directly.

Usage:
  ./analysis/blast/blast_sweep.py weight [samples] [bits...]
  ./analysis/blast/blast_sweep.py act    [samples] [bits...]
"""
import csv, logging, os, random, statistics, sys, time

import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "src"))
import fi_lib                                                    # noqa: E402
from ultralytics import YOLO                                     # noqa: E402
from ultralytics.data.augment import LetterBox                   # noqa: E402

logging.getLogger("ultralytics").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = "/home/mgong2/datasets/coco128/images/train2017"
OUTDIR = os.path.join(ROOT, "results")
IMAGES = 4          # blast radius is image-dependent for activation faults
THRESH = 1e-3       # per-value difference that counts as "changed"
SEED = 0

MODE = sys.argv[1] if len(sys.argv) > 1 else "weight"
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 16
BITS = [int(b) for b in sys.argv[3:]] or [31, 30, 29, 24, 23, 22]
if MODE not in ("weight", "act"):
    sys.exit("mode must be 'weight' or 'act'")

os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, f"blast_{MODE}_n{SAMPLES}.csv")

model = YOLO(os.path.join(ROOT, "yolov8n.pt"))
model.model.fuse()
core = model.model.cuda().eval()
convs = [(n, m) for n, m in core.named_modules() if isinstance(m, torch.nn.Conv2d)]

# anchor index -> (x, y) centre in 640-space, matching the P3/P4/P5 concat order
GRID = [((xx + 0.5) * s, (yy + 0.5) * s)
        for s, n in ((8, 80), (16, 40), (32, 20))
        for yy in range(n) for xx in range(n)]
SCALE_END = (6400, 8000, 8400)        # cumulative P3, P4, P5 boundaries
assert len(GRID) == 8400

_lb = LetterBox((640, 640))
imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))[:IMAGES]
X = []
for f in imgs:
    im = _lb(image=cv2.imread(os.path.join(IMG_DIR, f)))
    X.append(torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy())
             .float().div(255).unsqueeze(0).cuda())


def forward(x):
    with torch.no_grad():
        out = core(x)
    t = out[0] if isinstance(out, (list, tuple)) else out
    return t.float()


GOLD = [forward(x) for x in X]


def spread(gold, corr, inj_xy):
    """-> (n_changed, frac_P3, frac_P4, frac_P5, median_dist or None)"""
    d = (corr - gold).abs().amax(1)[0]          # worst change per anchor
    changed = (torch.isnan(d) | torch.isinf(d) | (d > THRESH))
    idx = changed.nonzero().flatten().tolist()
    if not idx:
        return 0, 0.0, 0.0, 0.0, None
    p3 = sum(1 for i in idx if i < SCALE_END[0])
    p4 = sum(1 for i in idx if SCALE_END[0] <= i < SCALE_END[1])
    p5 = len(idx) - p3 - p4
    dist = None
    if inj_xy is not None:
        ds = [((GRID[i][0] - inj_xy[0]) ** 2 + (GRID[i][1] - inj_xy[1]) ** 2) ** 0.5
              for i in idx]
        dist = statistics.median(ds)
    n = len(idx)
    return n, p3 / n, p4 / n, p5 / n, dist


rng = random.Random(SEED)
rows = []
t0 = time.time()
total = len(convs) * SAMPLES * len(BITS)
bar = tqdm(total=total, desc=f"blast/{MODE}", unit="inj", ncols=90)

for bit in BITS:
    rng = random.Random(SEED)          # same targets at every bit
    for li, (name, layer) in enumerate(convs):
        for _ in range(SAMPLES):
            if MODE == "weight":
                idx = rng.randrange(layer.weight.numel())
                flat = layer.weight.data.view(-1)
                orig = flat[idx].clone()
                flat[idx] = fi_lib.flip(flat[idx], bit)
                per_img = [spread(g, forward(x), None) for g, x in zip(GOLD, X)]
                flat[idx] = orig
                tgt, inj = idx, dict(chan="", fy="", fx="")
            else:
                chan, fy, fx = rng.randrange(1024), rng.random(), rng.random()

                def hook(mod, i_, o, chan=chan, fy=fy, fx=fx, bit=bit):
                    if o.dim() != 4:
                        return o
                    _n, c, h, w = o.shape
                    y, x_ = int(fy * h), int(fx * w)
                    o[:, chan % c, y, x_] = fi_lib.flip(o[:, chan % c, y, x_], bit)
                    return o

                h = layer.register_forward_hook(hook)
                per_img = [spread(g, forward(x), (fx * 640, fy * 640))
                           for g, x in zip(GOLD, X)]
                h.remove()
                tgt, inj = "", dict(chan=chan, fy=round(fy, 4), fx=round(fx, 4))

            n = statistics.mean(p[0] for p in per_img)
            dists = [p[4] for p in per_img if p[4] is not None]
            rows.append(dict(
                layer=name, layer_idx=li, bit=bit, weight_idx=tgt, **inj,
                n_changed=round(n, 1), frac_changed=round(n / 8400, 5),
                frac_p3=round(statistics.mean(p[1] for p in per_img), 4),
                frac_p4=round(statistics.mean(p[2] for p in per_img), 4),
                frac_p5=round(statistics.mean(p[3] for p in per_img), 4),
                median_dist=round(statistics.mean(dists), 1) if dists else "",
                images=len(per_img)))
            bar.update(1)
bar.close()

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

nz = [r["n_changed"] for r in rows]
print(f"\nelapsed {time.time()-t0:.0f}s | {len(rows):,} injections | {len(X)} images each")
print(f"anchors changed: median {statistics.median(nz):,.0f} / 8400 "
      f"({100*statistics.median(nz)/8400:.1f}%)   max {max(nz):,.0f}")
print(f"wrote {OUT}")
