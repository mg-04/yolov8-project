#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Weight fault-injection campaign over every conv layer in YOLOv8n.

Reports BOTH metrics, because they disagree in ways that matter:

  mAP      dataset-aggregate accuracy. Comparable across campaigns, but a
           coarse threshold on it misses detection-level corruption entirely.
  verdict  per-image three-way taxonomy (masked / benign / SDC / DUE) against
           golden detections. DUE = output carries NaN/Inf, so it is DETECTABLE;
           SDC = wrong but structurally clean, therefore silent.

Usage:  ./campaign.py [samples_per_layer] [bit]
"""
import contextlib, csv, io, logging, os, random, sys, time
import torch
from ultralytics import YOLO

import fi_lib

logging.getLogger("ultralytics").setLevel(logging.ERROR)

ROOT = "/home/mgong2/tools/yolov8-project"
WEIGHTS = f"{ROOT}/yolov8n.pt"
DATA = "coco128.yaml"
IMG_DIR = "/home/mgong2/datasets/coco128/images/train2017"
RESULTS_DIR = f"{ROOT}/results"
SDC_IMAGES = 16  # fixed subset used for per-image verdict classification
SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 16
BIT = int(sys.argv[2]) if len(sys.argv) > 2 else 30
SEED = 0

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT = f"{RESULTS_DIR}/campaign_bit{BIT:02d}_n{SAMPLES}.csv"

model = YOLO(WEIGHTS)
model.model.fuse()
core = model.model.cuda().eval()
detect = fi_lib.make_detector(core)
convs = [(n, m) for n, m in model.model.named_modules() if isinstance(m, torch.nn.Conv2d)]


def eval_map():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = model.val(data=DATA, device=0, verbose=False, plots=False, batch=32)
    return r.box.map, r.box.map50


def inject(layer, idx, bit):
    """Flip one bit in place; return the original value for restoration."""
    flat = layer.weight.data.view(-1)
    orig = flat[idx].clone()  # .clone() is essential: flat[idx] is a VIEW
    flat[idx] = fi_lib.flip(flat[idx], bit)
    return orig


golden_map, golden_map50 = eval_map()
sdc_imgs = [os.path.join(IMG_DIR, f)
            for f in sorted(x for x in os.listdir(IMG_DIR) if x.endswith(".jpg"))[:SDC_IMAGES]]
sdc_gold = {p: detect(p)[0] for p in sdc_imgs}

print(f"golden: mAP50-95={golden_map:.4f}  mAP50={golden_map50:.4f}")
print(f"verdict subset: {len(sdc_imgs)} images, "
      f"{sum(len(v) for v in sdc_gold.values())} golden detections")
print(f"layers={len(convs)}  samples/layer={SAMPLES}  bit={BIT}")
print(f"total injections={len(convs)*SAMPLES}\n", flush=True)

rng = random.Random(SEED)
rows = []
t0 = time.time()

print(f"{'#':>3} {'layer':<28}{'params':>9}{'meanmAP':>9}{'crit%':>6}"
      f"{'mask':>6}{'benign':>7}{'SDC':>6}{'DUE':>6}", flush=True)
for li, (name, layer) in enumerate(convs):
    n = layer.weight.numel()
    maps, agg = [], {}
    for _ in range(SAMPLES):
        idx = rng.randrange(n)
        orig = inject(layer, idx, BIT)
        mp, mp50 = eval_map()
        verdicts, tl, tp, tc, nan_tot = {}, 0, 0, 0, 0
        for pth in sdc_imgs:
            dets, nn, ni = detect(pth)
            v, cat, lost, ph, cf = fi_lib.outcome(sdc_gold[pth], dets, nn, ni)[:5]
            verdicts[v] = verdicts.get(v, 0) + 1
            tl += lost; tp += ph; tc += cf; nan_tot += nn
        layer.weight.data.view(-1)[idx] = orig  # restore
        maps.append(mp)
        for k, v in verdicts.items():
            agg[k] = agg.get(k, 0) + v
        rows.append(dict(
            layer=name, layer_idx=li, params=n, weight_idx=idx, bit=BIT,
            map=round(mp, 5), map50=round(mp50, 5), delta=round(mp - golden_map, 5),
            sdc_images=len(sdc_imgs),
            v_masked=verdicts.get("masked", 0), v_benign=verdicts.get("benign", 0),
            v_sdc=verdicts.get("SDC", 0), v_due=verdicts.get("DUE", 0),
            lost=tl, phantom=tp, class_flip=tc, out_nan=nan_tot,
            critical=int(mp < 0.5 * golden_map),
            sdc_critical=int(verdicts.get("SDC", 0) > 0),
            due_critical=int(verdicts.get("DUE", 0) > 0)))
    crit = 100.0 * sum(m < 0.5 * golden_map for m in maps) / len(maps)
    g = lambda k: agg.get(k, 0)
    print(f"{li:>3} {name:<28}{n:>9,}{sum(maps)/len(maps):>9.4f}{crit:>5.0f}%"
          f"{g('masked'):>6}{g('benign'):>7}{g('SDC'):>6}{g('DUE'):>6}", flush=True)

final_map, _ = eval_map()
drift = abs(final_map - golden_map)
print(f"\ngolden re-check: {final_map:.4f} (drift {drift:.2e}) "
      f"{'OK' if drift < 1e-9 else '*** RESTORE FAILED ***'}")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n = len(rows)
print(f"elapsed {time.time()-t0:.0f}s | {n} injections")
print(f"mAP-critical : {sum(r['critical'] for r in rows)}/{n} = "
      f"{100*sum(r['critical'] for r in rows)/n:.1f}%")
print(f"SDC (silent) : {sum(r['sdc_critical'] for r in rows)}/{n} = "
      f"{100*sum(r['sdc_critical'] for r in rows)/n:.1f}%")
print(f"DUE (loud)   : {sum(r['due_critical'] for r in rows)}/{n} = "
      f"{100*sum(r['due_critical'] for r in rows)/n:.1f}%")
print(f"wrote {OUT}")
