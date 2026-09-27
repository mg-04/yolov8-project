#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Grouped bar chart of per-block fault sensitivity.

X axis: conv blocks in network depth order (head branches last).
Four bars per block, a 2x2 of fault site x error class:

    weight     SDC   solid blue
    weight     DUE   solid orange
    activation SDC   hatched blue
    activation DUE   hatched orange

Colour carries the error class, texture carries the fault site -- a composite
encoding, so only two hues are needed and the 2x2 stays readable.

Usage:
  ./plot_layers.py                 bit 30 -> results/layer_sensitivity_bit30.png
  ./plot_layers.py --bit 29
  ./plot_layers.py --bit 30 --out /tmp/x.png
"""
import csv, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

args = sys.argv[1:]
BIT_ARG = args[args.index("--bit") + 1] if "--bit" in args else "30"
ALL_BITS = [31, 30, 29, 24, 23, 22]
# "--bit all" gives the EXPECTED rate for a uniform-random single-bit flip:
# the six swept bits summed and divided by all 32 positions. The 26 unswept
# positions are mantissa bits below 22, which change a value by < 0.1% and
# measured 0.0-0.5% at bit 22, so treating them as zero is a slight underestimate.
BIT = None if BIT_ARG == "all" else int(BIT_ARG)
_tag = "all" if BIT is None else f"{BIT:02d}"
OUT = (args[args.index("--out") + 1] if "--out" in args
       else os.path.join(RESULTS, f"layer_sensitivity_bit{_tag}.png"))

# validated 2-hue categorical pair (dataviz reference palette slots 1 and 2)
C_SDC, C_DUE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdcd6"

def fnames(bit):
    return {"weight": f"campaign_bit{bit:02d}_n16.csv",
            "act": f"act_persistent_bit{bit:02d}_n16.csv"}


def block_of(layer):
    """Group key: block index, or the head branch name."""
    if layer.startswith("model.22."):
        return "dfl" if "dfl" in layer else ("cv2" if "cv2" in layer else "cv3")
    return layer.split(".")[1]


def rates(fname):
    """-> {block: (sdc%, due%)} using the per-image rate."""
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return None
    agg = defaultdict(lambda: [0, 0, 0])
    with open(path) as f:
        for r in csv.DictReader(f):
            a = agg[block_of(r["layer"])]
            a[0] += int(r["v_sdc"]); a[1] += int(r["v_due"])
            a[2] += int(r["sdc_images"])
    return {k: (100 * v[0] / v[2], 100 * v[1] / v[2]) for k, v in agg.items()}


def expected_rates(which):
    """Sum the six swept bits, divide by 32 -> per-random-bit-flip expectation."""
    tot = defaultdict(lambda: [0.0, 0.0])
    for b in ALL_BITS:
        r = rates(fnames(b)[which])
        if r is None:
            sys.exit(f"missing results for bit {b}; run the full sweep first")
        for k, (sdc, due) in r.items():
            tot[k][0] += sdc; tot[k][1] += due
    return {k: (v[0] / 32, v[1] / 32) for k, v in tot.items()}


if BIT is None:
    W, A = expected_rates("weight"), expected_rates("act")
else:
    W = rates(fnames(BIT)["weight"])
    A = rates(fnames(BIT)["act"])
    if W is None or A is None:
        sys.exit(f"missing results for bit {BIT}; run the campaigns first")

# depth order: numeric blocks ascending, then the three head branches
numeric = sorted((int(k) for k in W if k.isdigit()))
order = [str(k) for k in numeric] + [b for b in ("cv2", "cv3", "dfl") if b in W]
labels = [b if not b.isdigit() else b for b in order]

x = range(len(order))
bw = 0.20          # bar width; 4 bars + a gap fit inside 1.0
off = [-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw]

fig, ax = plt.subplots(figsize=(14, 5.2), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

series = [
    ([W[b][0] for b in order], C_SDC, None,  "weight · SDC (silent)"),
    ([W[b][1] for b in order], C_DUE, None,  "weight · DUE (loud)"),
    ([A[b][0] for b in order], C_SDC, "///", "activation · SDC (silent)"),
    ([A[b][1] for b in order], C_DUE, "///", "activation · DUE (loud)"),
]
for (vals, color, hatch, _label), dx in zip(series, off):
    ax.bar([i + dx for i in x], vals, bw * 0.92, color=color, hatch=hatch,
           edgecolor=SURFACE, linewidth=0.8, zorder=3)

ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
ax.xaxis.grid(False)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, color=INK2, fontsize=9)
ax.set_xlim(-0.6, len(order) - 0.4)
ymax = 100 if BIT is not None else 5 * (1 + int(max(
    max(v) for d in (W, A) for v in d.values()) // 5))
ax.set_ylim(0, ymax)
step = 20 if ymax > 40 else 1 if ymax <= 6 else 5
ticks = list(range(0, ymax + 1, step))
ax.set_yticks(ticks)
ax.set_yticklabels([f"{v}%" for v in ticks], color=INK2, fontsize=9)
ax.tick_params(length=0)

ax.set_xlabel("conv block, in network depth order  →  head branches last",
              color=INK2, fontsize=9.5, labelpad=26)
ax.set_ylabel("per-image rate" if BIT is not None
              else "expected rate per random bit flip", color=INK2, fontsize=9.5)
_title = (f"YOLOv8n per-block fault sensitivity, bit {BIT}" if BIT is not None
          else "YOLOv8n per-block fault sensitivity, averaged over all 32 bit positions")
ax.set_title(f"{_title}  ·  colour = error class, hatch = activation fault",
             color=INK, fontsize=12, pad=14, loc="left")

# section spans: faint dividers plus a centred name under each run
n_bb = sum(1 for b in order if b.isdigit() and int(b) <= 9)
n_neck = sum(1 for b in order if b.isdigit() and int(b) > 9)
spans = [("backbone", 0, n_bb - 1),
         ("neck (PAN-FPN)", n_bb, n_bb + n_neck - 1),
         ("head 22", n_bb + n_neck, len(order) - 1)]
for _name, lo, _hi in spans[1:]:
    ax.axvline(lo - 0.5, color=GRID, linewidth=1.0, zorder=1)
for name, lo, hi in spans:
    if hi < lo:
        continue
    ax.annotate(name, xy=((lo + hi) / 2, -0.075), xycoords=("data", "axes fraction"),
                ha="center", va="top", color=INK2, fontsize=9,
                annotation_clip=False)

ax.legend(handles=[Patch(facecolor=c, hatch=h, edgecolor=SURFACE, label=l)
                   for _v, c, h, l in series],
          loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False,
          fontsize=9, labelcolor=INK2, handlelength=1.4)

fig.tight_layout()
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {OUT}")

# text table alongside, so the numbers are readable without the image
print(f"\n{'block':>6}{'W SDC':>8}{'W DUE':>8}{'A SDC':>8}{'A DUE':>8}")
for b in order:
    print(f"{b:>6}{W[b][0]:>7.0f}%{W[b][1]:>7.0f}%{A[b][0]:>7.0f}%{A[b][1]:>7.0f}%")
