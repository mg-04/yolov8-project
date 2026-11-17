#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Cost vs. protection vs. geometric damage, for every bound configuration.

Four panels, one shared axis of configurations ordered by deployed metadata:

  1  metadata actually deployed, KB -- one bound per UNIT, not the expanded
     per-element tensors the harness keeps so the check vectorises
  2  SDC retired, relative to the undefended campaign on the same fault sites
  3  DUE retired, same basis
  4  box drift of SURVIVING detections, in IoU points (0 = boxes did not move)

Weights and biases are plotted as separate series because they are separate
campaigns over separate site populations -- biases are 0.18% of parameters, so
a uniform sample would draw one roughly every 550 injections and say nothing
about them. Their behaviour genuinely differs: an exponent bound cannot see a
sign flip, and bias sign flips are the single largest bias failure mode.

Error bars are 95% Wilson intervals on the DEFENDED residual rate, mapped
through retired = 1 - p_def/p_undef. They do not include uncertainty in the
undefended denominator, so they understate total uncertainty somewhat.

Usage:  ./plot_tradeoff.py [weight_csv] [bias_csv]
"""
import collections, csv, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W_CSV = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/sentry_weight_n48_img24.csv"
B_CSV = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/results/sentry_bias_n64_img24.csv"
OUT = f"{ROOT}/figures/sentry_tradeoff.png"

# ---- design tokens (validated: CVD dE 24.7, normal-vision dE 33.6) ----------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e8e5"
S_WEIGHT, S_BIAS, NEUTRAL = "#2a78d6", "#eb6834", "#8a8984"

ORDER = ["exp 1-sided per-chan", "exp 2-sided per-chan", "hybrid exp-w signed-b",
         "WS signed per-chan", "signed + floor per-chan", "WS signed K=128"]
SHORT = {"exp 1-sided per-chan": "exp 1-sided", "exp 2-sided per-chan": "exp 2-sided",
         "hybrid exp-w signed-b": "hybrid  exp\u2009/\u2009signed",
         "WS signed per-chan": "signed per-chan", "signed + floor per-chan": "signed + floor",
         "WS signed K=128": "signed K=128"}
# BRAM36 blocks at natural packing width, from the metadata derivation
BRAM = {"exp 1-sided per-chan": 4, "exp 2-sided per-chan": 5, "WS signed per-chan": 18,
        "signed + floor per-chan": 18, "WS signed K=128": 55,
        "hybrid exp-w signed-b": 8}


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(path):
    """-> {config: dict of aggregated counts}, summing over bit positions."""
    agg = collections.defaultdict(lambda: collections.Counter())
    drift = collections.defaultdict(lambda: [0.0, 0])      # sum(pts*n), sum(n)
    for r in csv.DictReader(open(path)):
        c = r["config"]
        for f in ("masked", "benign", "sdc", "due"):
            agg[c][f] += int(r[f])
        agg[c]["meta_kb"] = float(r["meta_kb"])
        nm = int(r.get("n_matched", 0) or 0)
        drift[c][0] += float(r.get("drift_pts", 0) or 0) * nm
        drift[c][1] += nm
    out = {}
    for c, a in agg.items():
        n = a["masked"] + a["benign"] + a["sdc"] + a["due"]
        out[c] = dict(n=n, sdc=a["sdc"], due=a["due"], meta_kb=a["meta_kb"],
                      drift=(drift[c][0] / drift[c][1]) if drift[c][1] else 0.0)
    return out


def retired(cfg, base, field):
    """(point, lo, hi) percent of `field` retired vs the undefended campaign."""
    if base[field] == 0:
        return float("nan"), float("nan"), float("nan")
    pu = base[field] / base["n"]
    lo, hi = wilson(cfg[field], cfg["n"])
    r = lambda p: 100.0 * (1.0 - p / pu)
    return r(cfg[field] / cfg["n"]), r(hi), r(lo)


def rbar(ax, y, val, h, color, r_px=4):
    """Thin bar with rounded data-end, anchored at x=0."""
    if val <= 0:
        return
    rad = min(h / 2.2, abs(val) / 2.2)
    ax.add_patch(FancyBboxPatch(
        (0, y - h / 2), max(val - rad, 1e-9), h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        linewidth=0, facecolor=color, mutation_aspect=1, clip_on=False, zorder=3))


def style(ax, title, xlabel):
    ax.set_title(title, fontsize=10.5, color=INK, pad=10, loc="left", )
    ax.set_xlabel(xlabel, fontsize=8.8, color=INK2, labelpad=6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="x", labelsize=8.4, colors=INK2, length=0)
    ax.tick_params(axis="y", labelsize=9.4, colors=INK, length=0)


def main():
    W, B = load(W_CSV), load(B_CSV)
    cfgs = [c for c in ORDER if c in W and c in B]
    wb, bb = W["undefended"], B["undefended"]

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.5), facecolor=SURFACE)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.775, bottom=0.235, wspace=0.22)

    ys = list(range(len(cfgs)))[::-1]          # cheapest at the top
    H, GAPH = 0.46, 0.165                      # single-series / grouped heights

    # ---- 1. deployed metadata --------------------------------------------
    ax = axes[0]
    for y, c in zip(ys, cfgs):
        rbar(ax, y, W[c]["meta_kb"], H, NEUTRAL)
        ax.text(W[c]["meta_kb"] + 5, y, f"{W[c]['meta_kb']:.1f} KB  ·  {BRAM[c]} BRAM36",
                va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 300)
    style(ax, "Deployed metadata", "KB  (one bound per unit)")

    # ---- 2, 3, 4. retired fractions, as dot plots -------------------------
    # Dots, not bars: each point carries a confidence interval, and a CI drawn
    # inside a filled bar reads as an artefact rather than as uncertainty.
    # All three share a 0-100 axis so they are directly comparable -- which also
    # makes the weight configs collapse onto each other, correctly: their
    # differences are not statistically resolvable at this sample size.
    def dot(ax, y, p, lo, hi, col, draw_ci=True):
        if p != p:
            return
        if draw_ci and hi - lo > 0.8:
            ax.plot([lo, hi], [y] * 2, color=col, lw=1.3, alpha=0.45,
                    zorder=3, solid_capstyle="round")
        ax.plot([p], [y], marker="o", ms=8.5, color=col, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=2, ls="")

    PANELS = (("sdc", "SDC retired", True), ("due", "DUE retired", True))
    for ax, (field, lab, ci) in zip(axes[1:3], PANELS):
        for y, c in zip(ys, cfgs):
            for off, (D, base, col) in ((+GAPH, (W[c], wb, S_WEIGHT)),
                                        (-GAPH, (B[c], bb, S_BIAS))):
                dot(ax, y + off, *retired(D, base, field), col, draw_ci=ci)
        for y, c in zip(ys, cfgs):
            v = retired(B[c], bb, field)[0]
            if v == v and v < 99.0:                 # label the shortfalls only
                ax.text(v, y - GAPH - 0.17, f"{v:.1f}%", va="top", ha="center",
                        fontsize=8.3, color=S_BIAS)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        style(ax, lab, "% of undefended, same sites")

    # ---- 4. box drift, same "retired" basis so the panels compare ---------
    ax = axes[3]
    for y, c in zip(ys, cfgs):
        for off, (D, base, col) in ((+GAPH, (W[c], wb, S_WEIGHT)),
                                    (-GAPH, (B[c], bb, S_BIAS))):
            r = 100.0 * (1.0 - D["drift"] / base["drift"]) if base["drift"] else float("nan")
            dot(ax, y + off, r, 0, 0, col, draw_ci=False)
        rb = 100.0 * (1.0 - B[c]["drift"] / bb["drift"])
        if rb < 99.0:
            ax.text(rb, y - GAPH - 0.17, f"{rb:.0f}%", va="top", ha="center",
                    fontsize=8.3, color=S_BIAS)
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    style(ax, "Box drift removed", "% of undefended drift, same sites")

    for ax in axes:
        ax.set_yticks(ys)
    axes[0].set_yticklabels([SHORT[c] for c in cfgs])
    for ax in axes[1:]:
        ax.set_yticklabels([])
    for ax in axes:
        ax.set_ylim(-0.6, len(cfgs) - 0.4)

    h = [plt.Line2D([], [], marker="s", ms=8, ls="", color=S_WEIGHT, label="weight faults"),
         plt.Line2D([], [], marker="s", ms=8, ls="", color=S_BIAS, label="bias faults")]
    leg = fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.995, 0.985),
                     frameon=False, fontsize=9.2, ncol=2, handletextpad=0.4,
                     columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.105, 0.945, "WeightSentry bound configurations: cost, protection, and residual geometric damage",
             fontsize=12.4, color=INK, ha="left")
    fig.text(0.105, 0.900,
             f"yolov8n · bits 31-23, identical injection sites across configurations · "
             f"weights {wb['n']:,} inferences/config, biases {bb['n']:,} · "
             f"bars and dots are percentages OF the undefended campaign shown below",
             fontsize=8.8, color=INK2, ha="left")
    fig.text(0.105, 0.862,
             f"undefended baseline — SDC {100*wb['sdc']/wb['n']:.2f}% / {100*bb['sdc']/bb['n']:.2f}%  ·  "
             f"DUE {100*wb['due']/wb['n']:.2f}% ({wb['due']} events) / {100*bb['due']/bb['n']:.2f}% ({bb['due']}) ·  "
             f"box drift {wb['drift']:.2f} / {bb['drift']:.2f} IoU points      (weights / biases)",
             fontsize=8.5, color=INK2, ha="left")
    fig.text(0.105, 0.072,
             "Dots are point estimates; lines are 95% Wilson intervals on the defended residual rate. No interval on drift — per-sample IoU variance is not recorded.",
             fontsize=7.9, color=INK2, ha="left")
    fig.text(0.105, 0.030,
             "Every configuration retires 100% of DUE, but the undefended weight DUE baseline is 6 events, so that panel's weight interval runs from 36% — the bias series (528 events) is the informative one.",
             fontsize=7.9, color=INK2, ha="left")

    os.makedirs(f"{ROOT}/figures", exist_ok=True)
    fig.savefig(OUT, dpi=190, facecolor=SURFACE)
    print(f"-> {OUT}")

    # exact values, since the panels carry only selective labels
    print(f"\n{'config':<18}{'KB':>8}{'BRAM':>6}", end="")
    for k in ("SDC ret%", "DUE ret%", "drift"):
        print(f"{k+' w':>11}{k+' b':>11}", end="")
    print()
    for c in cfgs:
        print(f"{SHORT[c]:<18}{W[c]['meta_kb']:>8.2f}{BRAM[c]:>6}", end="")
        for f in ("sdc", "due"):
            print(f"{retired(W[c], wb, f)[0]:>11.2f}{retired(B[c], bb, f)[0]:>11.2f}", end="")
        print(f"{W[c]['drift']:>11.3f}{B[c]['drift']:>11.3f}")
    print(f"{'undefended':<18}{0.0:>8.2f}{0:>6}{'-':>11}{'-':>11}{'-':>11}{'-':>11}"
          f"{wb['drift']:>11.3f}{bb['drift']:>11.3f}")


if __name__ == "__main__":
    main()
