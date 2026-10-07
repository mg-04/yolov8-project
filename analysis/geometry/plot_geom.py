#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Two figures from the geometry sweep, layers in network depth order.

  1  per-layer SDC rate at selected bit positions
  2  per-layer box displacement in benign outcomes

Usage:
  ./analysis/geometry/plot_geom.py
  ./analysis/geometry/plot_geom.py --bits 30 25 26 29 22 --images 1024
"""
import csv, glob, os, re, sys
from collections import defaultdict
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS, FIGURES = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)

a = sys.argv[1:]
IMAGES = int(a[a.index("--images") + 1]) if "--images" in a else 1024
BITS = ([int(x) for x in a[a.index("--bits") + 1:] if x.isdigit()]
        if "--bits" in a else [30, 29, 26, 25, 22])
BITS = sorted(BITS, reverse=True)      # legend and bar order follow bit number

# validated categorical slots 1-5 (dataviz reference palette)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdcd6"

rows = []
for f in sorted(glob.glob(os.path.join(RESULTS, "geom_weight_n16*.csv"))):
    rows += list(csv.DictReader(open(f)))
rows = [r for r in rows if int(r["images"]) == IMAGES]
if not rows:
    sys.exit(f"no rows at images={IMAGES}")

# layer order follows SUBSET in geom_sweep.py, which is network depth order
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geom_sweep.py")).read()
ORDER = re.findall(r'"(model\.[^"]+)"', src[src.index("SUBSET = ["):src.index("]", src.index("SUBSET = ["))])
ORDER = [l for l in ORDER if l in {r["layer"] for r in rows}]
short = [l.replace("model.", "").replace(".conv", "") for l in ORDER]


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(FIGURES, name)
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {p}")


def style(ax, ylab, title, sub):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_ylabel(ylab, color=INK2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=12.5, pad=16, loc="left")
    ax.text(0, 1.015, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(short, rotation=38, ha="right", color=INK2, fontsize=8.5)
    ax.set_xlim(-0.7, len(ORDER) - 0.3)


# ---- figure 1: SDC by layer and bit -------------------------------------
sdc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for r in rows:
    e = sdc[r["layer"]][int(r["bit"])]
    e[0] += int(r["v_sdc"]); e[1] += int(r["images"])

fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
bw = 0.8 / len(BITS)
for j, b in enumerate(BITS):
    vals = [100 * sdc[l][b][0] / sdc[l][b][1] if sdc[l][b][1] else 0 for l in ORDER]
    ax.bar([i + (j - (len(BITS) - 1) / 2) * bw for i in range(len(ORDER))], vals,
           bw * 0.9, color=C[j % len(C)], edgecolor=SURFACE, linewidth=0.7,
           zorder=3, label=f"bit {b}")
style(ax, "Per-Image SDC Rate",
      "YOLOv8n Per-Layer Fault Sensitivity, Weight Faults",
      f"Layers left to right in network depth order  ·  {IMAGES} held-out images, 16 samples/layer")
ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color=INK2, fontsize=9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=len(BITS),
          frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.3)
save(fig, "layer_bit_sensitivity.png")

# ---- figure 2: box drift by layer ---------------------------------------
drift = defaultdict(list)
for r in rows:
    if int(r["v_benign"]) > 0 and r["mean_shift"]:
        drift[r["layer"]].append(float(r["mean_shift"]))
vals = [212 * st.mean(drift[l]) if drift.get(l) else 0 for l in ORDER]   # px on a 212px diagonal

fig, ax = plt.subplots(figsize=(13, 4.6), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.bar(range(len(ORDER)), vals, 0.62, color=C[0], edgecolor=SURFACE, linewidth=0.7, zorder=3)
for i, v in enumerate(vals):
    if v > 0.3:
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", color=INK2, fontsize=8.5)
style(ax, "Centre Shift (px on a 150px Object)",
      "Box Displacement by Layer, Benign Outcomes",
      "CAVEAT: averaged over ALL images in each injection, not benign ones only, "
      "so masked images dilute it downward")
ax.set_ylim(0, max(vals) * 1.18)
for t in ax.get_yticks():
    pass
ax.set_yticklabels([f"{t:.0f}" for t in ax.get_yticks()], color=INK2, fontsize=9)
save(fig, "layer_box_drift.png")

# ---- figure 3: confidence CDF -------------------------------------------
# For each confidence value x, what fraction of detections score at or below x.
# Shows the whole distribution rather than a median, and answers directly:
# "could a higher confidence threshold filter out corruption-induced detections?"
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
sys.path.insert(0, os.path.join(ROOT, "src"))
import fi_lib                                                    # noqa: E402
from ultralytics import YOLO                                     # noqa: E402

ev = []
for f in sorted(glob.glob(os.path.join(RESULTS, "events_weight_n16*.csv"))):
    ev += list(csv.DictReader(open(f)))
keys = {(r["layer"], r["bit"]) for r in rows}
ev = [x for x in ev if (x["layer"], x["bit"]) in keys]

_m = YOLO(os.path.join(ROOT, "yolov8n.pt"))
_m.model.fuse()
_det = fi_lib.make_detector(_m.model.cuda().eval())
cfg = {127: "coco-val2017-sub127.yaml", 1024: "coco-val2017-sub1024.yaml"}[IMAGES]
gold_conf = [d[1] for p in fi_lib.dataset_images(cfg) for d in _det(p)[0]]

series = [("Golden detections", gold_conf, C[0], "-"),
          ("Lost objects", [float(x["conf"]) for x in ev if x["event"] == "lost"], C[1], "-"),
          ("Phantom detections", [float(x["conf"]) for x in ev if x["event"] == "phantom"], C[2], "-")]

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for lbl, v, col, ls in series:
    if not v:
        continue
    v = sorted(v)
    y = [100 * (i + 1) / len(v) for i in range(len(v))]
    ax.plot(v, y, color=col, linewidth=2, linestyle=ls, zorder=3,
            label=f"{lbl}  (n={len(v):,})")
ax.set_axisbelow(True)
ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(length=0, colors=INK2, labelsize=9)
ax.set_xlim(0.25, 1.0); ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
ax.set_xlabel("Detection Confidence", color=INK2, fontsize=9.5)
ax.set_ylabel("Cumulative % of Detections At or Below", color=INK2, fontsize=9.5)
ax.set_title("Confidence Distribution: Can a Threshold Filter Corruption?",
             color=INK, fontsize=12.5, pad=16, loc="left")
ax.text(0, 1.015, "A curve further right means higher confidence. "
        "Phantoms sitting right of golden means no threshold separates them.",
        transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
save(fig, "confidence_cdf.png")

# ---- figure 4: SDC vs the magnitude predictor ---------------------------
# For each (layer, bit): x = fraction of that layer's weights where flipping the
# bit GROWS the value (i.e. the bit is currently 0), y = measured SDC rate.
# The magnitude theory says damage needs the growth direction, so points should
# rise with x, and separate by how large the multiplier is.
import torch                                                     # noqa: E402
MOD = dict(_m.model.named_modules())
MULT = {b: 2.0 ** (1 << (b - 23)) for b in range(23, 31)}   # exponent bits only;
# bit 31 is the sign and bit 22 the mantissa MSB, neither scales by a power of two,
# so this figure is about the exponent field alone

pts = []
for lay in ORDER:
    W = MOD[lay].weight.data.flatten()
    I = W.view(torch.int32)
    for b in sorted({int(r["bit"]) for r in rows}):
        if b not in MULT:        # skip sign (31) and mantissa (22)
            continue
        s = sdc[lay][b]
        if not s[1]:
            continue
        grow = ((I >> b) & 1 == 0).float().mean().item()
        pts.append((grow, 100 * s[0] / s[1], b, lay))

fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
# Bit position is ORDINAL (ordered by multiplier), not categorical, so use a
# sequential one-hue ramp instead of cycling categorical slots -- 8 bits through
# 5 categorical colours would give two bits the same hue.
BLIST = sorted({p[2] for p in pts}, reverse=True)
RAMP = plt.get_cmap("viridis")
for j, b in enumerate(BLIST):
    P = [p for p in pts if p[2] == b]
    lab = (f"bit {b}  (x{MULT[b]:.0f})" if MULT[b] < 1e6
           else f"bit {b}  (x2^{1 << (b - 23)})")
    ax.scatter([p[0] for p in P], [p[1] for p in P], s=58, alpha=0.9,
               color=RAMP(j / max(len(BLIST) - 1, 1)), edgecolor=SURFACE,
               linewidth=0.8, zorder=3, label=lab)
for p in pts:
    if (p[1] > 35 and p[2] != 30) or (p[1] > 90 and p[3].endswith("dfl.conv")):
        ax.annotate(p[3].replace("model.", "").replace(".conv", ""),
                    (p[0], p[1]), fontsize=7.5, color=INK2,
                    xytext=(5, 3), textcoords="offset points")
ax.set_axisbelow(True); ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
ax.tick_params(length=0, colors=INK2, labelsize=9)
ax.set_xlabel("Fraction of Layer's Weights Where the Flip Grows the Value", color=INK2, fontsize=9.5)
ax.set_ylabel("Per-Image SDC Rate", color=INK2, fontsize=9.5)
ax.set_yticks(range(0, 101, 20)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
ax.set_title("Damage Requires the Growth Direction", color=INK, fontsize=12.5, pad=16, loc="left")
ax.text(0, 1.015, "One point per (layer, bit). Larger multiplier and more weights in the growth "
        "direction both raise SDC.", transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", ncol=2)
save(fig, "sdc_vs_magnitude.png")

# ---- figure 5: class-flip transition LIFT --------------------------------
# Raw counts just rediscover the dataset: person is 35% of objects, so any
# person -> X transition looks large. Lift = observed / expected-if-independent
# removes that, and has a natural midpoint at 1.0 (chance), so the scale is
# diverging on log2(lift): blue = suppressed, red = favoured.
import numpy as np                                               # noqa: E402
from collections import Counter                                  # noqa: E402
names = _m.model.names
flips = [x for x in ev if x["event"] == "class_flip"]
src = Counter(int(x["cls"]) for x in flips)
dst = Counter(int(x["cls_to"]) for x in flips)
pair = Counter((int(x["cls"]), int(x["cls_to"])) for x in flips)
N = len(flips)

MIN_EV = 40          # below this, lift is too noisy to plot
cand = Counter()
for (i, j), n in pair.items():
    if n >= MIN_EV:
        cand[i] += n; cand[j] += n
top = [c for c, _ in cand.most_common(16)]

# Order the axes so similar classes sit next to each other, derived from the data
# rather than assumed: build the symmetric lift matrix, then greedily seriate by
# repeatedly appending whichever remaining class has the highest lift against the
# one just placed. If flips really do follow visual similarity, related classes
# fall into contiguous blocks along the diagonal; if not, the order looks random.
def _lift(i, j):
    e = src[i] * dst[j] / N
    return (pair.get((i, j), 0) / e) if e > 0 else 0.0

def _aff(i, j):                       # symmetric: a <-> b affinity
    return _lift(i, j) + _lift(j, i)

rest, order = set(top), [max(top, key=lambda c: cand[c])]
rest.discard(order[0])
while rest:
    nxt = max(rest, key=lambda c: _aff(order[-1], c))
    order.append(nxt); rest.discard(nxt)
top = order

idx = {c: k for k, c in enumerate(top)}

M = np.full((len(top), len(top)), np.nan)
for (i, j), n in pair.items():
    if i in idx and j in idx and n >= MIN_EV:
        exp = src[i] * dst[j] / N
        if exp > 0:
            M[idx[i]][idx[j]] = np.log2(n / exp)

fig, ax = plt.subplots(figsize=(9.2, 7.6), dpi=160)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor("#efefec")
lim = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
for i in range(len(top)):
    for j in range(len(top)):
        if not np.isnan(M[i][j]) and abs(M[i][j]) > 1.6:
            v = 2 ** M[i][j]
            # sub-1 lifts need decimals: "%.0f" turned 0.19x into a misleading "0x"
            lab = f"{v:.0f}x" if v >= 2 else f"{v:.2f}x"
            ax.text(j, i, lab, ha="center", va="center",
                    fontsize=7.5, color="#ffffff" if abs(M[i][j]) > 3 else INK)
ax.set_xticks(range(len(top))); ax.set_yticks(range(len(top)))
ax.set_xticklabels([names[c] for c in top], rotation=45, ha="right", fontsize=8.5, color=INK2)
ax.set_yticklabels([names[c] for c in top], fontsize=8.5, color=INK2)
ax.set_xlabel("Became This Class", color=INK2, fontsize=9.5)
ax.set_ylabel("Was This Class", color=INK2, fontsize=9.5)
ax.set_title("Class Flips Favour Visually Similar Classes", color=INK,
             fontsize=12.5, pad=34, loc="left")
ax.text(0, 1.012, f"Lift = observed / expected if flips ignored class.  "
        f"{N:,} events, pairs with >= {MIN_EV} shown; grey = too few.",
        transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.044)
cb.set_ticks([-lim, 0, lim])
cb.set_ticklabels([f"{2**-lim:.2f}x", "1x (chance)", f"{2**lim:.0f}x"])
cb.set_label("Lift", color=INK2, fontsize=9); cb.ax.tick_params(length=0, colors=INK2, labelsize=8)
save(fig, "class_flip_heatmap.png")
