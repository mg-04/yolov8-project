#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Sweep B: box geometry and SDC event identity.

Two questions the main campaigns cannot answer, because they record counts only:

  1. In benign outcomes, HOW FAR did the boxes move? compare() computes the IoU of
     every matched pair and keeps only the minimum, for a threshold test. Here the
     geometry is recorded: IoU, centre shift (in units of the golden box diagonal,
     so pure translation) and area ratio (pure scale). IoU alone conflates the two.

  2. In SDC outcomes, WHICH objects were lost / invented / mislabelled? Class, size,
     pyramid scale, confidence, and for activation faults the distance from the
     injection point to the object.

Writes two CSVs, joined on inj_id:
  geom_<mode>_n<N>.csv     one row per injection
  events_<mode>_n<N>.csv   one row per lost / phantom / class_flip event

Sweeps a REPRESENTATIVE LAYER SUBSET rather than all 64, because these questions
are about the distribution of damage, not about ranking layers. The budget goes
into samples per layer instead: 8 layers x 128 samples is the same cost as
64 x 16 with 8x the per-layer power.

Usage:
  ./analysis/geometry/geom_sweep.py weight [samples] [bits...]
  ./analysis/geometry/geom_sweep.py act    [samples] [bits...]
  ./analysis/geometry/geom_sweep.py weight 128 30 29 22 --all-layers
"""
import csv, logging, os, random, sys, time

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "src"))
import fi_lib                                                    # noqa: E402
from ultralytics import YOLO                                     # noqa: E402

logging.getLogger("ultralytics").setLevel(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = "coco-val2017-sub1024.yaml"   # held out; also supplies the images (--data overrides)
OUTDIR = os.path.join(ROOT, "results")
IMAGES = None   # None = every image in DATA
SEED = 0
# A destroyed model emits ~128 phantom detections per image, and enumerating them
# all adds nothing (the count is already in the main CSV) while exploding the events
# file. Cap the detailed rows; `lost` and `class_flip` are bounded by the golden
# detection count (~3/image) so they are never capped.
MAX_PHANTOM_ROWS = 10

# Chosen to span depth, weight magnitude, branch type and fault locality.
# Benign events concentrate in large-weight layers, so several are included.
SUBSET = [
    # 16 layers spanning section, depth, parameter count, SDC rate, benign yield and
    # blast radius. Proportional to the network's own section split (27/18/9/9/1).
    # Covers 24.8% of conv parameters; SDC 23-100%, benign 0.0-17.7%, blast 185-8400.
    "model.0.conv",             # top benign 17.7%, largest weights (43% >= 2)
    "model.2.cv2.conv",         # 2nd benign 14.2%
    "model.4.cv2.conv",         # 3rd benign 13.4%, P3 skip source
    "model.5.conv",             # SDC extreme: 100% SDC, 0.1% benign
    "model.6.m.1.cv1.conv",     # mid-backbone bottleneck
    "model.7.conv",             # LARGEST layer, 294,912 params
    "model.9.cv2.conv",         # SPPF, structurally distinct
    "model.12.cv1.conv",        # neck, top-down path
    "model.15.cv2.conv",        # neck, feeds Detect P3
    "model.18.cv1.conv",        # neck, bottom-up; blast 2,000 (P4+P5 only)
    "model.21.cv2.conv",        # neck, feeds Detect P5; blast 400 (P5 only)
    "model.22.cv2.0.0.conv",    # box branch P3, lowest SDC 23%
    "model.22.cv2.2.2",         # box OUTPUT conv, P5; blast 397
    "model.22.cv3.0.2",         # class output P3, 100% under every fault type
    "model.22.cv3.2.2",         # class output P5; blast 185, the most local
    "model.22.dfl.conv",        # 16 params, geometry outlier, bit-29 peak
]

# strip flags AND their values, so a flag argument is never read as a bit number
_VALUED = {"--data", "--slice"}
args, _skip = [], False
for _a in sys.argv[1:]:
    if _skip:
        _skip = False
        continue
    if _a.startswith("--"):
        _skip = _a in _VALUED
        continue
    args.append(_a)
ALL_LAYERS = "--all-layers" in sys.argv
# --slice i/N takes every Nth layer starting at i, so one bit can be split across
# processes. Bit 30 is the only axis-less case: it cannot be parallelised by bit.
SLICE = None
if "--data" in sys.argv:
    DATA = sys.argv[sys.argv.index("--data") + 1]
for a in sys.argv:
    if a.startswith("--slice"):
        i, tot = sys.argv[sys.argv.index(a) + 1].split("/")
        SLICE = (int(i), int(tot))
MODE = args[0] if args else "weight"
SAMPLES = int(args[1]) if len(args) > 1 else 128
BITS = [int(b) for b in args[2:]] or [30, 29, 22]
if MODE not in ("weight", "act"):
    sys.exit("mode must be 'weight' or 'act'")

os.makedirs(OUTDIR, exist_ok=True)
# bits go in the filename so per-bit runs can go in parallel without clobbering
_dtag = os.path.splitext(os.path.basename(DATA))[0].replace("coco-val2017-", "")
tag = (f"{MODE}_n{SAMPLES}_b" + "-".join(str(b) for b in BITS) + f"_{_dtag}"
       + ("_all" if ALL_LAYERS else "")
       + (f"_s{SLICE[0]}of{SLICE[1]}" if SLICE else ""))
OUT_MAIN = os.path.join(OUTDIR, f"geom_{tag}.csv")
OUT_EVENTS = os.path.join(OUTDIR, f"events_{tag}.csv")

model = YOLO(os.path.join(ROOT, "yolov8n.pt"))
model.model.fuse()
core = model.model.cuda().eval()
detect = fi_lib.make_detector(core)
convs = [(n, m) for n, m in core.named_modules() if isinstance(m, torch.nn.Conv2d)]
if not ALL_LAYERS:
    by_name = dict(convs)
    missing = [n for n in SUBSET if n not in by_name]
    if missing:
        sys.exit(f"unknown layers in SUBSET: {missing}")
    convs = [(n, by_name[n]) for n in SUBSET]
if SLICE:
    i, tot = SLICE
    convs = convs[i::tot]

paths = fi_lib.dataset_images(DATA, IMAGES)
GOLD = {p: detect(p)[0] for p in paths}
print(f"{len(convs)} layers x {SAMPLES} samples x {len(BITS)} bits x {len(paths)} images"
      f"  ({DATA})")
print(f"golden detections: {sum(len(v) for v in GOLD.values())}\n")


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


rng = random.Random(SEED)
main, events = [], []
inj_id = 0
t0 = time.time()
bar = tqdm(total=len(convs) * SAMPLES * len(BITS), desc=f"geom/{MODE}",
           unit="inj", ncols=90)

for bit in BITS:
    rng = random.Random(SEED)
    for name, layer in convs:
        for _ in range(SAMPLES):
            inj_id += 1
            if MODE == "weight":
                idx = rng.randrange(layer.weight.numel())
                flat = layer.weight.data.view(-1)
                orig = flat[idx].clone()
                flat[idx] = fi_lib.flip(flat[idx], bit)
                tgt, inj_xy, meta = idx, None, dict(chan="", fy="", fx="")
                h = None
            else:
                chan, fy, fx = rng.randrange(1024), rng.random(), rng.random()

                def hook(mod, i_, o, chan=chan, fy=fy, fx=fx, bit=bit):
                    if o.dim() != 4:
                        return o
                    _n, c, hh, ww = o.shape
                    o[:, chan % c, int(fy * hh), int(fx * ww)] = fi_lib.flip(
                        o[:, chan % c, int(fy * hh), int(fx * ww)], bit)
                    return o

                h = layer.register_forward_hook(hook)
                tgt, inj_xy = "", (fx * 640, fy * 640)
                meta = dict(chan=chan, fy=round(fy, 4), fx=round(fx, 4))

            agg = dict(masked=0, benign=0, SDC=0, DUE=0)
            ious, shifts, areas, nlost, nphan, nflip = [], [], [], 0, 0, 0
            for p in paths:
                dets, nn, ni = detect(p)
                r = fi_lib.compare_detailed(GOLD[p], dets)
                loud = nn > 0 or ni > 0
                cls = "DUE" if loud else ("SDC" if r["category"] not in
                                          ("masked", "benign") else r["category"])
                agg[cls] += 1
                for m in r["matches"]:
                    ious.append(m[0]); shifts.append(m[1]); areas.append(m[2])
                nlost += len(r["lost"]); nphan += len(r["phantom"]); nflip += len(r["flips"])
                for kind, lst in (("lost", r["lost"]),
                                  ("phantom", r["phantom"][:MAX_PHANTOM_ROWS]),
                                  ("class_flip", r["flips"])):
                    for e in lst:
                        cx, cy = centre(e["box"])
                        d = (((cx - inj_xy[0]) ** 2 + (cy - inj_xy[1]) ** 2) ** 0.5
                             if inj_xy else "")
                        events.append(dict(
                            inj_id=inj_id, layer=name, bit=bit, mode=MODE,
                            image=os.path.basename(p), event=kind,
                            cls=e["cls"], cls_to=e["cls_to"], conf=round(e["conf"], 4),
                            area=round(e["area"], 1), area_frac=round(e["area"] / (640 * 640), 5),
                            scale=e["scale"], err_class=cls,
                            n_phantom_total=nphan if kind == "phantom" else "",
                            dist_to_injection=round(d, 1) if d != "" else ""))

            if h is not None:
                h.remove()
            else:
                layer.weight.data.view(-1)[idx] = orig

            mean = lambda v: round(sum(v) / len(v), 5) if v else ""
            main.append(dict(
                inj_id=inj_id, layer=name, bit=bit, mode=MODE, weight_idx=tgt, **meta,
                images=len(paths), n_matched=len(ious),
                min_iou=round(min(ious), 5) if ious else "",
                mean_iou=mean(ious), mean_shift=mean(shifts), mean_area_ratio=mean(areas),
                lost=nlost, phantom=nphan, class_flip=nflip,
                v_masked=agg["masked"], v_benign=agg["benign"],
                v_sdc=agg["SDC"], v_due=agg["DUE"]))
            bar.update(1)
bar.close()

for path, data in ((OUT_MAIN, main), (OUT_EVENTS, events)):
    if not data:
        print(f"(no rows for {path})"); continue
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader(); w.writerows(data)

ben = [r for r in main if r["v_benign"]]
print(f"\nelapsed {time.time()-t0:.0f}s | {len(main):,} injections | {len(events):,} events")
print(f"injections with >=1 benign image: {len(ben):,}")
print(f"wrote {OUT_MAIN}")
print(f"wrote {OUT_EVENTS}")
