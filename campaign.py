#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Full-network weight fault-injection campaign.

Sweeps every conv layer in YOLOv8n, injecting single-bit weight flips and
measuring mAP on coco128. Model is loaded once; each injection is applied
in place and restored, so cost per injection is one val() pass.

Usage:  ./campaign.py [samples_per_layer] [bit]
"""
import contextlib, csv, io, logging, random, sys, time
import torch
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

WEIGHTS = "/home/mgong2/tools/yolov8-project/yolov8n.pt"
DATA = "coco128.yaml"
OUT = "/home/mgong2/tools/yolov8-project/campaign_results.csv"
SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 16
BIT = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SEED = 0

model = YOLO(WEIGHTS)
model.model.fuse()
convs = [(n, m) for n, m in model.model.named_modules() if isinstance(m, torch.nn.Conv2d)]


def evaluate():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = model.val(data=DATA, device=0, verbose=False, plots=False, batch=32)
    return r.box.map, r.box.map50


def inject(layer, idx, bit):
    """Flip one bit in place; return the original value for restoration."""
    flat = layer.weight.data.view(-1)
    orig = flat[idx].clone()
    flat[idx] = (flat[idx].view(torch.int32) ^ (1 << bit)).view(torch.float32)
    return orig


golden_map, golden_map50 = evaluate()
print(f"golden: mAP50-95={golden_map:.4f}  mAP50={golden_map50:.4f}")
print(f"layers={len(convs)}  samples/layer={SAMPLES}  bit={BIT}")
print(f"total injections={len(convs)*SAMPLES}\n", flush=True)

rng = random.Random(SEED)
rows = []
t0 = time.time()

print(f"{'#':>3} {'layer':<28}{'params':>9}{'mean mAP':>10}{'worst':>8}{'crit%':>7}", flush=True)
for li, (name, layer) in enumerate(convs):
    n = layer.weight.numel()
    idxs = [rng.randrange(n) for _ in range(SAMPLES)]
    maps = []
    for idx in idxs:
        orig = inject(layer, idx, BIT)
        mp, mp50 = evaluate()
        layer.weight.data.view(-1)[idx] = orig  # restore
        maps.append(mp)
        rows.append(
            dict(layer=name, layer_idx=li, params=n, weight_idx=idx, bit=BIT,
                 map=round(mp, 5), map50=round(mp50, 5),
                 delta=round(mp - golden_map, 5),
                 critical=int(mp < 0.5 * golden_map))
        )
    crit = 100.0 * sum(m < 0.5 * golden_map for m in maps) / len(maps)
    print(f"{li:>3} {name:<28}{n:>9,}{sum(maps)/len(maps):>10.4f}"
          f"{min(maps):>8.4f}{crit:>6.0f}%", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

final_map, _ = evaluate()
drift = abs(final_map - golden_map)
print(f"\ngolden re-check: {final_map:.4f} (drift {drift:.2e}) "
      f"{'OK' if drift < 1e-9 else '*** RESTORE FAILED ***'}")

crit_total = sum(r["critical"] for r in rows)
print(f"elapsed {time.time()-t0:.0f}s | {len(rows)} injections")
print(f"critical (mAP < 50% of golden): {crit_total}/{len(rows)} = {100*crit_total/len(rows):.1f}%")
print(f"wrote {OUT}")
