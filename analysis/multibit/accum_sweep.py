#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""M1: accumulated multi-bit faults, tested against the independence null.

N faults are placed at uniformly random positions across the WHOLE parameter
store, at uniformly random bit positions in [0, 31]. Two sampling choices
separate this from the per-layer sweeps and both are deliberate:

  parameter-weighted  a random bit in memory lands in a layer in proportion to
                      that layer's size, so model.7.conv takes 9.4% of draws and
                      model.22.dfl.conv 0.0005%. The stratified per-layer
                      sampler used by geom_sweep would wildly over-represent the
                      two small high-magnitude layers and inflate the damage.
  uniform bits        including the mantissa, which is mostly harmless. This is
                      what makes the single-fault rate ~4% rather than ~13%.

The null hypothesis is that faults do not interact:

    P(SDC | N) = 1 - (1 - p)^N

so the output is read as a deviation from that curve, not as a damage level.
Sub-additive means the network masks accumulated faults; super-additive means
they compound.

Usage:
  ./analysis/multibit/accum_sweep.py [samples] [images]
  ./analysis/multibit/accum_sweep.py 128 256
  ./analysis/multibit/accum_sweep.py 128 256 --faults 1,2,4,8
  ./analysis/multibit/accum_sweep.py 1024 32 --model yolov8x --faults 1
"""
import csv, logging, os, random, sys, time

import torch
from tqdm import tqdm
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import fi_lib                                                    # noqa: E402

logging.getLogger("ultralytics").setLevel(logging.ERROR)

a = sys.argv[1:]
pos = [x for x in a if not x.startswith("--")]
SAMPLES = int(pos[0]) if len(pos) > 0 else 128
IMAGES = int(pos[1]) if len(pos) > 1 else 256
FAULTS = ([int(x) for x in a[a.index("--faults") + 1].split(",")]
          if "--faults" in a else [1, 2, 4, 8, 16, 32, 64, 128])
MODEL = a[a.index("--model") + 1] if "--model" in a else "yolov8n"
BITS  = ([int(x) for x in a[a.index("--bits") + 1].split(",")]
         if "--bits" in a else list(range(32)))
LAYER = a[a.index("--layer") + 1] if "--layer" in a else None
MAG   = a[a.index("--mag") + 1] if "--mag" in a else None   # "high" | "low"
DATA = "coco-val2017-sub1024.yaml"
SEED = 0

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
tag = f"{MODEL}_n{SAMPLES}_img{IMAGES}_f{'-'.join(str(f) for f in FAULTS)}"
if len(BITS) < 32: tag += "_b" + "-".join(str(b) for b in BITS)
if LAYER: tag += "_" + LAYER.replace(".", "")
if MAG:   tag += "_" + MAG
OUT = os.path.join(RESULTS, f"accum_{tag}.csv")

model = YOLO(os.path.join(ROOT, f"{MODEL}.pt"))
model.model.fuse()
core = model.model.cuda().eval()
detect = fi_lib.make_detector(core)

# Flat view of every parameter in the fused model: this IS what sits in memory.
# Each entry is (name, flat_tensor, cumulative_end), so a single uniform draw in
# [0, total) selects a parameter with probability proportional to layer size.
PARAMS, total = [], 0
for name, p in core.named_parameters():
    flat = p.data.view(-1)
    total += flat.numel()
    PARAMS.append((name, flat, total))
print(f"parameter store: {total:,} values across {len(PARAMS)} tensors")

# Restricted sampling pool. Default (None) means the whole store, drawn uniformly,
# which weights each layer by its size -- the right model for a random memory upset.
# --layer and --mag narrow it instead, for stratified tests of the mechanism:
#   --mag high  the |w| >= 2 population, damageable at nine bit positions
#   --mag low   the ordinary |w| < 2 population, damageable only at bit 30
POOL = None
if LAYER or MAG:
    idx = []
    base = 0
    for name, flat, end in PARAMS:
        take = (LAYER is None) or (name.startswith(LAYER))
        if take and MAG:
            sel = (flat.abs() >= 2) if MAG == "high" else (flat.abs() < 2)
            idx.append(sel.nonzero(as_tuple=True)[0].cpu() + base)
        elif take:
            idx.append(torch.arange(flat.numel()) + base)
        base = end
    POOL = torch.cat(idx).tolist() if idx else []
    if not POOL:
        sys.exit(f"empty pool for layer={LAYER} mag={MAG}")
    where = f" in {LAYER}" if LAYER else ""
    cmp = ">= 2" if MAG == "high" else "< 2"
    what = f" with |w| {cmp}" if MAG else ""
    print(f"restricted pool: {len(POOL):,} weights{where}{what}")
print(f"bit pool: {BITS if len(BITS) < 32 else 'all 32'}")


def locate(g):
    """Map a global parameter index to (name, flat_tensor, local_index)."""
    lo, hi = 0, len(PARAMS) - 1
    while lo < hi:                       # first tensor whose end exceeds g
        mid = (lo + hi) // 2
        if PARAMS[mid][2] <= g:
            lo = mid + 1
        else:
            hi = mid
    name, flat, end = PARAMS[lo]
    return name, flat, g - (end - flat.numel())


def inject(sites):
    """Flip every site; return originals for restoration in reverse order."""
    saved = []
    for g, bit in sites:
        _, flat, i = locate(g)
        saved.append((flat, i, flat[i].clone()))   # .clone(): flat[i] is a VIEW
        flat[i] = fi_lib.flip(flat[i], bit)
    return saved


imgs = fi_lib.dataset_images(DATA, IMAGES)
GOLD = {p: detect(p)[0] for p in imgs}
print(f"golden: {sum(len(v) for v in GOLD.values())} detections over {len(imgs)} images")

rows = []
bar = tqdm(total=len(FAULTS) * SAMPLES, unit="inj")
t0 = time.time()
for nf in FAULTS:
    rng = random.Random(SEED + nf)       # reproducible per fault count
    for s in range(SAMPLES):
        sites = [(POOL[rng.randrange(len(POOL))] if POOL else rng.randrange(total),
                  BITS[rng.randrange(len(BITS))]) for _ in range(nf)]
        saved = inject(sites)
        agg = dict(masked=0, benign=0, SDC=0, DUE=0)
        for p in imgs:
            dets, nn, ni = detect(p)
            verdict = fi_lib.outcome(GOLD[p], dets, nn, ni)[0]
            agg[verdict] += 1
        for flat, i, orig in reversed(saved):
            flat[i] = orig
        bits = sorted({b for _, b in sites}, reverse=True)
        rows.append(dict(
            n_faults=nf, sample=s, images=len(imgs),
            v_masked=agg["masked"], v_benign=agg["benign"],
            v_sdc=agg["SDC"], v_due=agg["DUE"],
            max_bit=max(b for _, b in sites),
            has_bit30=int(any(b == 30 for _, b in sites)),
            bits=" ".join(str(b) for b in bits[:12])))
        bar.update(1)
bar.close()

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nelapsed {time.time() - t0:.0f}s   wrote {OUT}")

p1 = None
print(f"\n{'N':>5s} {'SDC':>8s} {'DUE':>8s}   {'indep. SDC':>10s}")
for nf in FAULTS:
    g = [r for r in rows if r["n_faults"] == nf]
    tot = sum(r["images"] for r in g)
    sdc = sum(r["v_sdc"] for r in g) / tot
    due = sum(r["v_due"] for r in g) / tot
    if nf == 1:
        p1 = sdc
    pred = 1 - (1 - p1) ** nf if p1 is not None else float("nan")
    print(f"{nf:5d} {100 * sdc:7.2f}% {100 * due:7.2f}%   {100 * pred:9.2f}%")
