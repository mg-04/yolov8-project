#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Activation fault-injection campaign, two fault modes, three-way taxonomy.

  ./campaign_act.py persistent [samples] [bit]
      Stuck-at fault in an activation buffer: the hook fires on EVERY forward
      pass and corrupts the location for every image in the batch. Reports mAP
      (comparable to campaign.py) plus per-image verdicts.

  ./campaign_act.py transient [samples] [bit]
      True single-event upset: one element corrupted during ONE inference, then
      gone. Dataset mAP is meaningless here (1 image of 128), so this mode
      reports per-image verdicts only.

Verdicts (see fi_lib): masked / benign / SDC (silent: wrong but structurally
clean output) / DUE (loud: output carries NaN or Inf, so it is detectable).

Targets are (channel, y_frac, x_frac) rather than flat indices, because
rect=True letterboxing gives different spatial dims per batch.
"""
import contextlib, csv, io, logging, os, random, sys, time
import torch
from ultralytics import YOLO

import fi_lib

logging.getLogger("ultralytics").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = f"{ROOT}/yolov8n.pt"
DATA = "coco128.yaml"
IMG_DIR = "/home/mgong2/datasets/coco128/images/train2017"
RESULTS_DIR = f"{ROOT}/results"
SDC_IMAGES = 16
SEED = 0

MODE = sys.argv[1] if len(sys.argv) > 1 else "persistent"
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 16
BIT = int(sys.argv[3]) if len(sys.argv) > 3 else 30
if MODE not in ("persistent", "transient"):
    sys.exit("mode must be 'persistent' or 'transient'")

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT = f"{RESULTS_DIR}/act_{MODE}_bit{BIT:02d}_n{SAMPLES}.csv"

model = YOLO(WEIGHTS)
model.model.fuse()
core = model.model.cuda().eval()
detect = fi_lib.make_detector(core)
convs = [(n, m) for n, m in model.model.named_modules() if isinstance(m, torch.nn.Conv2d)]


def make_hook(chan, fy, fx, bit, once, record):
    """Corrupt one element of the layer output.

    `once=True` models a transient upset: fires on the first forward pass only.
    Otherwise it fires on every pass and hits ALL batch elements -- a stuck-at
    buffer fault corrupts that location for every image flowing through it.
    """
    state = {"fired": False}

    def hook(mod, inp, out):
        if once and state["fired"]:
            return out
        if out.dim() != 4:
            return out
        n, c, h, w = out.shape
        y, x = int(fy * h), int(fx * w)
        ch = chan % c
        record["before"] = out[0, ch, y, x].item()
        out[:, ch, y, x] = fi_lib.flip(out[:, ch, y, x], bit)
        record["after"] = out[0, ch, y, x].item()
        state["fired"] = True
        return out

    return hook


def eval_map():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = model.val(data=DATA, device=0, verbose=False, plots=False, batch=32)
    return r.box.map, r.box.map50


def direction(rec):
    b, a = rec.get("before"), rec.get("after")
    if b is None:
        return "none"
    if a != a:  # NaN
        return "nan"
    return "grew" if abs(a) > abs(b) else "shrank"


imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
sdc_imgs = [os.path.join(IMG_DIR, f) for f in imgs[:SDC_IMAGES]]
sdc_gold = {p: detect(p)[0] for p in sdc_imgs}

rng = random.Random(SEED)
rows = []
t0 = time.time()
hdr = f"{'mask':>6}{'benign':>7}{'SDC':>6}{'DUE':>6}"

if MODE == "persistent":
    golden, _ = eval_map()
    print(f"golden mAP50-95={golden:.4f} | mode=persistent bit={BIT} samples={SAMPLES}")
    print(f"verdict subset: {len(sdc_imgs)} images, "
          f"{sum(len(v) for v in sdc_gold.values())} golden detections")
    print(f"total injections={len(convs)*SAMPLES}\n", flush=True)
    print(f"{'#':>3} {'layer':<28}{'meanmAP':>9}{'crit%':>6}{hdr}", flush=True)
    for li, (name, layer) in enumerate(convs):
        maps, agg = [], {}
        for _ in range(SAMPLES):
            chan, fy, fx = rng.randrange(1024), rng.random(), rng.random()
            rec = {}
            h = layer.register_forward_hook(make_hook(chan, fy, fx, BIT, False, rec))
            mp, mp50 = eval_map()
            verdicts, tl, tp, tc, nan_tot = {}, 0, 0, 0, 0
            for pth in sdc_imgs:
                dets, nn, ni = detect(pth)
                v, cat, lost, ph, cf = fi_lib.outcome(sdc_gold[pth], dets, nn, ni)[:5]
                verdicts[v] = verdicts.get(v, 0) + 1
                tl += lost; tp += ph; tc += cf; nan_tot += nn
            h.remove()
            maps.append(mp)
            for k, v in verdicts.items():
                agg[k] = agg.get(k, 0) + v
            rows.append(dict(
                layer=name, layer_idx=li, chan=chan, fy=round(fy, 4), fx=round(fx, 4),
                bit=BIT, map=round(mp, 5), delta=round(mp - golden, 5),
                before=rec.get("before"), after=rec.get("after"), direction=direction(rec),
                sdc_images=len(sdc_imgs),
                v_masked=verdicts.get("masked", 0), v_benign=verdicts.get("benign", 0),
                v_sdc=verdicts.get("SDC", 0), v_due=verdicts.get("DUE", 0),
                lost=tl, phantom=tp, class_flip=tc, out_nan=nan_tot,
                critical=int(mp < 0.5 * golden),
                sdc_critical=int(verdicts.get("SDC", 0) > 0),
                due_critical=int(verdicts.get("DUE", 0) > 0)))
        crit = 100.0 * sum(m < 0.5 * golden for m in maps) / len(maps)
        g = lambda k: agg.get(k, 0)
        print(f"{li:>3} {name:<28}{sum(maps)/len(maps):>9.4f}{crit:>5.0f}%"
              f"{g('masked'):>6}{g('benign'):>7}{g('SDC'):>6}{g('DUE'):>6}", flush=True)

else:  # transient
    print(f"mode=transient bit={BIT} samples={SAMPLES} | one fault per inference")
    print(f"total injections={len(convs)*SAMPLES}\n", flush=True)
    gold_cache = dict(sdc_gold)
    print(f"{'#':>3} {'layer':<28}{hdr}{'lost':>6}{'phant':>6}{'flip':>6}", flush=True)
    for li, (name, layer) in enumerate(convs):
        agg = {}
        tl = tp = tc = 0
        for _ in range(SAMPLES):
            img = os.path.join(IMG_DIR, rng.choice(imgs))
            if img not in gold_cache:
                gold_cache[img] = detect(img)[0]
            chan, fy, fx = rng.randrange(1024), rng.random(), rng.random()
            rec = {}
            h = layer.register_forward_hook(make_hook(chan, fy, fx, BIT, True, rec))
            dets, nn, ni = detect(img)
            h.remove()
            v, cat, lost, ph, cf = fi_lib.outcome(gold_cache[img], dets, nn, ni)[:5]
            agg[v] = agg.get(v, 0) + 1
            tl += lost; tp += ph; tc += cf
            rows.append(dict(
                layer=name, layer_idx=li, image=os.path.basename(img),
                chan=chan, fy=round(fy, 4), fx=round(fx, 4), bit=BIT,
                before=rec.get("before"), after=rec.get("after"),
                fired=int("before" in rec), direction=direction(rec),
                n_golden=len(gold_cache[img]), n_corrupt=len(dets),
                out_nan=nn, out_inf=ni, verdict=v, category=cat,
                lost=lost, phantom=ph, class_flip=cf,
                critical=int(v in ("SDC", "DUE")),
                sdc_critical=int(v == "SDC"), due_critical=int(v == "DUE")))
        g = lambda k: agg.get(k, 0)
        print(f"{li:>3} {name:<28}{g('masked'):>6}{g('benign'):>7}{g('SDC'):>6}"
              f"{g('DUE'):>6}{tl:>6}{tp:>6}{tc:>6}", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n = len(rows)
print(f"\nelapsed {time.time()-t0:.0f}s | {n} injections")
print(f"SDC (silent) : {sum(r['sdc_critical'] for r in rows)}/{n} = "
      f"{100*sum(r['sdc_critical'] for r in rows)/n:.1f}%")
print(f"DUE (loud)   : {sum(r['due_critical'] for r in rows)}/{n} = "
      f"{100*sum(r['due_critical'] for r in rows)/n:.1f}%")
print(f"wrote {OUT}")
