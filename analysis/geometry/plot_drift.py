#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Mean and worst-case box geometry change per layer, from the geometry sweep.

  1  layer_area_drift.png     box area change, signed (shrink vs grow)
  2  layer_displacement.png   box centre displacement

Both read `mean_area_ratio` / `mean_shift`, which are already averaged over the
images of one injection. So the "worst" end of each range is the worst
INJECTION, not the worst individual box; single boxes move further than this.

Usage:
  ./analysis/geometry/plot_drift.py
  ./analysis/geometry/plot_drift.py --images 1024
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
DIAG = 212.0          # diagonal of a 150x150 object; mean_shift is in diagonal units

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdcd6"

rows = []
for f in sorted(glob.glob(os.path.join(RESULTS, "geom_weight_n16*.csv"))):
    rows += list(csv.DictReader(open(f)))
# Every injection with at least one surviving box. Do NOT filter on v_benign:
# that drops the SDC-dominated injections, which are exactly the ones that move
# boxes, and makes the class heads look geometrically inert when they are not.
rows = [r for r in rows if int(r["images"]) == IMAGES and r["n_matched"] != "0"]
if not rows:
    sys.exit(f"no rows at images={IMAGES}")
BITS = sorted({int(r["bit"]) for r in rows}, reverse=True)

# layer order follows SUBSET in geom_sweep.py, which is network depth order
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "geom_sweep.py")).read()
ORDER = re.findall(r'"(model\.[^"]+)"', src[src.index("SUBSET = ["):src.index("]", src.index("SUBSET = ["))])
ORDER = [l for l in ORDER if l in {r["layer"] for r in rows}]
short = [l.replace("model.", "").replace(".conv", "") for l in ORDER]


def sdc_frac(r):
    return (int(r["v_sdc"]) + int(r["v_due"])) / int(r["images"])


def collect(field, transform, keep):
    """Per-layer list of per-injection values, for injections passing `keep`."""
    d = defaultdict(list)
    for r in rows:
        if not r[field] or not keep(r):
            continue
        v = float(r[field])
        if field == "mean_area_ratio" and v == 0:      # no matched box to measure
            continue
        d[r["layer"]].append(transform(v))
    return d


# The sweep records geometry pooled over all images of an injection, with no
# per-verdict breakdown, so verdict is separated at injection granularity:
# did this injection corrupt most of its images, or almost none of them?
GROUPS = [("Mostly masked or benign injections", lambda r: sdc_frac(r) < 0.5, C[0]),
          ("SDC-dominated injections (>50% of images)", lambda r: sdc_frac(r) >= 0.5, C[1])]


def figure(field, transform, ylab, title, name, fmt, zero_line=False):
    """Range bar from min to max per layer and outcome group, mean marked."""
    fig, ax = plt.subplots(figsize=(13.5, 5.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    series, off = [], 0.17
    for k, (lbl, keep, col) in enumerate(GROUPS):
        d = collect(field, transform, keep)
        series.append((d, col, lbl, (k - (len(GROUPS) - 1) / 2) * 2 * off))

    allv = [v for d, *_ in series for l in ORDER for v in d.get(l, [])]
    span = max(allv) - min(allv + [0])

    for d, col, _lbl, dx in series:
        for i, l in enumerate(ORDER):
            v = d.get(l)
            if not v:
                continue
            a_, b_, m = min(v), max(v), st.mean(v)
            ax.plot([i + dx, i + dx], [a_, b_], color=col, linewidth=7, alpha=0.30,
                    solid_capstyle="round", zorder=3)
            ax.plot([i + dx], [m], marker="o", markersize=6.5, color=col,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)
            far = b_ if abs(b_) >= abs(a_) else a_
            if abs(far) > 0.07 * span:
                ax.text(i + dx, far + (0.025 if far >= 0 else -0.025) * span,
                        fmt(far), ha="center",
                        va="bottom" if far >= 0 else "top", color=col, fontsize=7.5)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    if zero_line:
        ax.axhline(0, color=INK2, linewidth=1.0, zorder=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_visible(not zero_line)
    ax.tick_params(length=0)
    ax.set_ylabel(ylab, color=INK2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=12.5, pad=16, loc="left")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(short, rotation=38, ha="right", color=INK2, fontsize=8.5)
    ax.set_xlim(-0.7, len(ORDER) - 0.3)
    pad = 0.16 * span
    ax.set_ylim(min(allv + [0]) - pad, max(allv) + pad)
    ax.set_yticklabels([fmt(t) for t in ax.get_yticks()], color=INK2, fontsize=9)

    # legend: one entry per outcome group, plus the glyph key
    for d, col, lbl, _ in series:
        ax.plot([], [], color=col, linewidth=7, alpha=0.30,
                label=f"{lbl}  (n={sum(len(d.get(l, [])) for l in ORDER):,})")
    ax.plot([], [], marker="o", markersize=6.5, color=INK2, linestyle="none",
            markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="Mean  ·  bar spans min to max")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
              fontsize=8.5, labelcolor=INK2, handlelength=1.6, columnspacing=1.6)

    fig.tight_layout()
    p = os.path.join(FIGURES, name)
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {p}")


print(f"{IMAGES} images, bits {BITS}, measured on matched boxes only (IoU >= "
      f"{0.5}); IoU matching bounds what is representable at area ratio "
      f"[0.50, 2.00] and displacement <= s/3, so both tails are censored.")

figure("mean_area_ratio", lambda v: 100 * (v - 1),
       "Box Area Change",
       "Box Area Drift by Layer and Outcome",
       "layer_area_drift.png",
       lambda v: f"{v:+.0f}%" if abs(v) >= 1 else f"{v:+.1f}%",
       zero_line=True)

figure("mean_shift", lambda v: DIAG * v,
       "Centre Displacement (px on a 150px Object)",
       "Box Displacement by Layer and Outcome",
       "layer_displacement.png",
       lambda v: f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}")
