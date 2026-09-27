#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Activation fault-injection campaign, with two distinct fault modes.

  ./campaign_act.py persistent [samples] [bit]
      A stuck-at fault in an activation buffer: the hook fires on EVERY forward
      pass, so the same element is corrupted for every image. Measured by
      dataset mAP, directly comparable to campaign.py's weight results.

  ./campaign_act.py transient [samples] [bit]
      A true single-event upset: the fault corrupts ONE element during ONE
      inference, then is gone. Dataset mAP is the wrong metric here (1 image of
      128 affected barely moves it), so this mode compares the affected image's
      detections against its own golden detections and classifies the outcome.

Indexing is shape-robust: a target is (channel, y_frac, x_frac) rather than a
flat index, because Ultralytics' rect=True letterboxing gives different spatial
dimensions per batch.
"""
import contextlib, csv, io, logging, os, random, sys, time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression

logging.getLogger("ultralytics").setLevel(logging.ERROR)

ROOT = "/home/mgong2/tools/yolov8-project"
WEIGHTS = f"{ROOT}/yolov8n.pt"
DATA = "coco128.yaml"
IMG_DIR = "/home/mgong2/datasets/coco128/images/train2017"
RESULTS_DIR = f"{ROOT}/results"
SEED = 0
SDC_IMAGES = 16  # fixed subset used for per-image SDC classification in persistent mode

MODE = sys.argv[1] if len(sys.argv) > 1 else "persistent"
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 16
BIT = int(sys.argv[3]) if len(sys.argv) > 3 else 30
if MODE not in ("persistent", "transient"):
    sys.exit("mode must be 'persistent' or 'transient'")

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT = f"{RESULTS_DIR}/act_{MODE}_bit{BIT:02d}_n{SAMPLES}.csv"

model = YOLO(WEIGHTS)
model.model.fuse()
convs = [(n, m) for n, m in model.model.named_modules() if isinstance(m, torch.nn.Conv2d)]


def flip(t, bit):
    return (t.view(torch.int32) ^ (1 << bit)).view(torch.float32)


def make_hook(chan, fy, fx, bit, once, record):
    """Corrupt one element of the layer output. `once` -> transient (first call only).

    `record` is a dict the hook fills with the before/after values, so the CSV can
    show whether the flip grew the value, shrank it, or produced NaN directly.
    """
    state = {"fired": False}

    def hook(mod, inp, out):
        if once and state["fired"]:
            return out
        if out.dim() != 4:
            return out
        n, c, h, w = out.shape
        y, x = int(fy * h), int(fx * w)
        # A stuck-at fault in an activation buffer corrupts that location for
        # EVERY image flowing through it -- so hit all batch elements, not out[0].
        before = out[0, chan % c, y, x].item()
        out[:, chan % c, y, x] = flip(out[:, chan % c, y, x], bit)
        after = out[0, chan % c, y, x].item()
        record["before"], record["after"] = before, after
        state["fired"] = True
        return out

    return hook


# predict() wraps the model in AutoBackend, whose internal modules are DIFFERENT
# objects -- hooks registered on model.model are orphaned there. So drive the
# network directly: letterbox -> forward -> NMS. One forward pass, hooks intact.
_core = model.model.cuda().eval()
_lb = LetterBox((640, 640))


def detect_direct(path):
    im = _lb(image=cv2.imread(path))
    x = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)
    with torch.no_grad():
        out = _core(x.unsqueeze(0).cuda())
    t = out[0] if isinstance(out, (list, tuple)) else out
    det = non_max_suppression(t.float(), 0.25, 0.45)[0]
    # boxes stay in letterbox coords; golden and corrupted share the same input
    return [(int(d[5]), float(d[4]), [float(v) for v in d[:4]]) for d in det]


# ---------- persistent mode: dataset mAP, comparable to campaign.py ----------
def eval_map():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = model.val(data=DATA, device=0, verbose=False, plots=False, batch=32)
    return r.box.map, r.box.map50


# ---------- transient mode: per-image detection comparison ----------
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def classify(gold, corr):
    """Compare two detection sets -> SDC category."""
    if not gold and not corr:
        return "masked", 0, 0, 0
    used, matched, cls_flip = set(), 0, 0
    min_iou = 1.0  # tightest match across all pairs -> did any box drift?
    for g in gold:
        best, bj = 0.0, None
        for j, c in enumerate(corr):
            if j in used:
                continue
            v = iou(g[2], c[2])
            if v > best:
                best, bj = v, j
        if bj is not None and best >= 0.5:
            used.add(bj)
            matched += 1
            min_iou = min(min_iou, best)
            if corr[bj][0] != g[0]:
                cls_flip += 1
    lost = len(gold) - matched
    phantom = len(corr) - len(used)
    if lost == 0 and phantom == 0 and cls_flip == 0:
        # all objects found with correct class; did the boxes move at all?
        cat = "masked" if min_iou >= 0.99 else "benign"
    elif cls_flip:
        cat = "class_flip"
    elif lost:
        cat = "object_lost"
    else:
        cat = "phantom"
    return cat, lost, phantom, cls_flip


rng = random.Random(SEED)
rows = []
t0 = time.time()

if MODE == "persistent":
    golden, _ = eval_map()
    # fixed image subset for SDC classification, with golden detections cached
    sdc_imgs = [os.path.join(IMG_DIR, f)
                for f in sorted(x for x in os.listdir(IMG_DIR) if x.endswith(".jpg"))[:SDC_IMAGES]]
    sdc_gold = {p: detect_direct(p) for p in sdc_imgs}
    print(f"golden mAP50-95={golden:.4f} | mode=persistent bit={BIT} samples={SAMPLES}")
    print(f"SDC subset: {len(sdc_imgs)} images, "
          f"{sum(len(v) for v in sdc_gold.values())} golden detections")
    print(f"total injections={len(convs)*SAMPLES}\n", flush=True)
    print(f"{'#':>3} {'layer':<28}{'mean mAP':>10}{'worst':>8}{'crit%':>7}"
          f"{'mask':>6}{'benign':>7}{'lost':>6}{'phant':>6}{'flip':>6}", flush=True)
    for li, (name, layer) in enumerate(convs):
        maps = []
        agg = {}
        for _ in range(SAMPLES):
            chan, fy, fx = rng.randrange(1024), rng.random(), rng.random()
            rec = {}
            h = layer.register_forward_hook(make_hook(chan, fy, fx, BIT, False, rec))
            mp, mp50 = eval_map()
            # hook still armed: classify per-image outcomes on the fixed subset
            cats = {}
            tot_lost = tot_ph = tot_cf = 0
            for pth in sdc_imgs:
                cat, lost, ph, cf = classify(sdc_gold[pth], detect_direct(pth))
                cats[cat] = cats.get(cat, 0) + 1
                tot_lost += lost; tot_ph += ph; tot_cf += cf
            h.remove()
            maps.append(mp)
            for k, v in cats.items():
                agg[k] = agg.get(k, 0) + v
            n_bad = sum(v for k, v in cats.items() if k in ("object_lost", "phantom", "class_flip"))
            rows.append(dict(layer=name, layer_idx=li, chan=chan, fy=round(fy, 4),
                             fx=round(fx, 4), bit=BIT, map=round(mp, 5),
                             delta=round(mp - golden, 5),
                             before=rec.get('before'), after=rec.get('after'),
                             sdc_images=len(sdc_imgs),
                             sdc_masked=cats.get('masked', 0),
                             sdc_benign=cats.get('benign', 0),
                             sdc_object_lost=cats.get('object_lost', 0),
                             sdc_phantom=cats.get('phantom', 0),
                             sdc_class_flip=cats.get('class_flip', 0),
                             lost=tot_lost, phantom=tot_ph, class_flip=tot_cf,
                             critical=int(mp < 0.5 * golden),
                             sdc_critical=int(n_bad > 0)))
        crit = 100.0 * sum(m < 0.5 * golden for m in maps) / len(maps)
        g = lambda k: agg.get(k, 0)
        print(f"{li:>3} {name:<28}{sum(maps)/len(maps):>10.4f}{min(maps):>8.4f}"
              f"{crit:>6.0f}%{g('masked'):>6}{g('benign'):>7}{g('object_lost'):>6}"
              f"{g('phantom'):>6}{g('class_flip'):>6}", flush=True)
    final, _ = eval_map()
    print(f"\ngolden re-check: {final:.4f} (drift {abs(final-golden):.2e}) "
          f"{'OK' if abs(final-golden) < 1e-9 else '*** HOOK LEAKED ***'}")

else:  # transient
    imgs = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    print(f"mode=transient bit={BIT} samples={SAMPLES} | one fault per inference")
    print(f"total injections={len(convs)*SAMPLES}\n", flush=True)
    gold_cache = {}
    print(f"{'#':>3} {'layer':<28}{'masked':>8}{'benign':>8}{'lost':>6}"
          f"{'phant':>7}{'clsflip':>8}", flush=True)
    for li, (name, layer) in enumerate(convs):
        cats = {}
        for _ in range(SAMPLES):
            img = os.path.join(IMG_DIR, rng.choice(imgs))
            if img not in gold_cache:
                gold_cache[img] = detect_direct(img)
            chan, fy, fx = rng.randrange(1024), rng.random(), rng.random()
            rec = {}
            h = layer.register_forward_hook(make_hook(chan, fy, fx, BIT, True, rec))
            corr = detect_direct(img)
            h.remove()
            cat, lost, phantom, cf = classify(gold_cache[img], corr)
            cats[cat] = cats.get(cat, 0) + 1
            b, a = rec.get('before'), rec.get('after')
            rows.append(dict(layer=name, layer_idx=li, image=os.path.basename(img),
                             chan=chan, fy=round(fy, 4), fx=round(fx, 4), bit=BIT,
                             before=b, after=a,
                             fired=int('before' in rec),
                             direction=('nan' if a is not None and a != a else
                                        'grew' if b is not None and abs(a) > abs(b) else
                                        'shrank' if b is not None else 'none'),
                             n_golden=len(gold_cache[img]), n_corrupt=len(corr),
                             category=cat, lost=lost, phantom=phantom, class_flip=cf,
                             critical=int(cat in ("object_lost", "phantom", "class_flip"))))
        g = lambda k: cats.get(k, 0)
        print(f"{li:>3} {name:<28}{g('masked'):>8}{g('benign'):>8}"
              f"{g('object_lost'):>6}{g('phantom'):>7}{g('class_flip'):>8}", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

crit = sum(r["critical"] for r in rows)
print(f"\nelapsed {time.time()-t0:.0f}s | {len(rows)} injections")
print(f"critical: {crit}/{len(rows)} = {100*crit/len(rows):.1f}%")
print(f"wrote {OUT}")
