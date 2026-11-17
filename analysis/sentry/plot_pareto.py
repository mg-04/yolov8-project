#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Cost vs residual damage, log-log. The Pareto view of the bound configurations.

Plotted as RESIDUAL rate, not "retired %": every configuration retires 98-100%,
so a linear retired-axis stacks them all against 100 and hides the differences
that remain. Residual on a log axis spreads them and keeps the undefended
baseline on the same scale as a reference line.

Configurations with ZERO observed residual events cannot be placed on a log
axis and cannot honestly be called zero either -- 0/13,824 has a 95% Wilson
upper bound of 0.027%. Those are drawn as UPPER LIMITS (downward arrows at the
bound), the standard treatment for a null measurement.

Down and to the left is better.

Usage:  ./plot_pareto.py [weight_csv] [bias_csv]
"""
import csv, collections, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W_CSV = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/sentry_weight_n48_img24.csv"
B_CSV = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/results/sentry_bias_n64_img24.csv"
OUT = f"{ROOT}/figures/sentry_pareto.png"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e8e5"
S_SDC, S_DUE = "#2a78d6", "#eb6834"

SHORT = {"exp 1-sided per-chan": "exp 1-sided", "exp 2-sided per-chan": "exp 2-sided",
         "hybrid exp-w signed-b": "hybrid", "WS signed per-chan": "signed per-chan",
         "signed + floor per-chan": "signed + floor", "WS signed K=128": "signed K=128"}
# deployed metadata, KB -- recomputed from the current code, not the stale CSV column
KB = {"exp 1-sided per-chan": 11.19, "exp 2-sided per-chan": 16.78,
      "hybrid exp-w signed-b": 27.97, "WS signed per-chan": 67.13,
      "signed + floor per-chan": 72.73, "WS signed K=128": 214.41}


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h) * 100, min(1.0, c + h) * 100


def load(path):
    agg = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(path)):
        for f in ("masked", "benign", "sdc", "due"):
            agg[r["config"]][f] += int(r[f])
    out = {}
    for c, a in agg.items():
        n = sum(a[f] for f in ("masked", "benign", "sdc", "due"))
        out[c] = dict(n=n, sdc=a["sdc"], due=a["due"])
    return out


def main():
    W, B = load(W_CSV), load(B_CSV)
    cfgs = [c for c in KB if c in W and c in B]
    cfgs.sort(key=lambda c: KB[c])

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.235, wspace=0.17)

    for ax, (D, tag) in zip(axes, ((W, "weight faults"), (B, "bias faults"))):
        base = D["undefended"]
        # DUE is not plotted: every configuration records zero events in both
        # streams, so the series is twelve identical upper limits carrying no
        # information. It is also zero STRUCTURALLY, not just empirically --
        # in_range() is a positive test and NaN/Inf fail every comparison, so a
        # non-finite weight can never pass. Stated in the subtitle instead.
        for field, col, lab in (("sdc", S_SDC, "SDC"),):
            # undefended reference line
            b = 100 * base[field] / base["n"]
            if b > 0:
                ax.axhline(b, color=col, lw=1.0, alpha=0.4, zorder=1)
                side = dict(x=288, ha="right") if field == "sdc" else dict(x=8.8, ha="left")
                ax.text(side["x"], b * 1.16, f"undefended {lab}  {b:.2f}%",
                        fontsize=7.8, color=col, va="bottom", ha=side["ha"])
            xs, ys, labelled = [], [], [1e-9]   # x of the last annotated zero point
            for c in cfgs:
                k, n = D[c][field], D[c]["n"]
                lo, hi = wilson(k, n)
                x = KB[c]
                if k == 0:                       # null result -> upper limit
                    # Annotated with the trial count, because a bare arrow reads
                    # as "not measured" when it means the opposite: the campaign
                    # ran in full and recorded no events at all.
                    ax.plot([x], [hi], marker="v", ms=9, color=col, ls="",
                            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5)
                    ax.plot([x, x], [hi, hi * 0.33], color=col, lw=1.2,
                            alpha=0.5, zorder=4)
                    if x / labelled[0] > 1.3:        # skip a near-duplicate
                        ax.text(x * 1.10, hi * 1.55, f"0 of\n{n:,}", fontsize=7.4,
                                color=col, ha="left", va="center", linespacing=1.15)
                        labelled[0] = x
                    continue
                y = 100 * k / n
                xs.append(x); ys.append(y)
                ax.plot([x, x], [max(lo, 1e-3), hi], color=col, lw=1.3,
                        alpha=0.45, zorder=4, solid_capstyle="round")
                ax.plot([x], [y], marker="o", ms=8.5, color=col, ls="",
                        markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(8.5, 300); ax.set_ylim(0.012, 95)
        # name each configuration on the axis itself rather than inside the
        # plot: at these costs several points sit within 10% of each other on a
        # log axis, so in-plot labels collide with each other and with the
        # reference lines.
        # "signed per-chan" (67.1 KB) and "signed + floor" (72.7 KB) are 8%
        # apart -- unresolvable on a log axis -- and score identically in both
        # panels, so they share one label at the geometric midpoint.
        ticks, labels, skip = [], [], False
        for i, c in enumerate(cfgs):
            if skip:
                skip = False
                continue
            nxt = cfgs[i + 1] if i + 1 < len(cfgs) else None
            if nxt and KB[nxt] / KB[c] < 1.25:
                ticks.append((KB[c] * KB[nxt]) ** 0.5)
                labels.append(f"{SHORT[c]}\n+ floor\n{KB[c]:.0f}\u2013{KB[nxt]:.0f} KB")
                skip = True
            else:
                ticks.append(KB[c]); labels.append(f"{SHORT[c]}\n{KB[c]:.1f} KB")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=34, ha="right", rotation_mode="anchor")
        for t in ax.get_xticklabels():
            t.set_fontsize(8.0)
        ax.set_yticks([0.02, 0.05, 0.1, 0.5, 1, 5, 10, 50])
        ax.set_yticklabels(["0.02", "0.05", "0.1", "0.5", "1", "5", "10", "50"])
        ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.7)
        ax.tick_params(labelsize=8.4, colors=INK2, length=0)
        ax.set_xlabel("deployed metadata  (log scale)", fontsize=9, color=INK2, labelpad=42)
        ax.set_title(tag, fontsize=10.5, color=INK, loc="left", pad=10)
    axes[0].set_ylabel("residual rate  (% of inferences, log)", fontsize=9, color=INK2)

    h = [plt.Line2D([], [], marker="o", ms=8, ls="", color=S_SDC,
                    label="residual SDC (95% CI)"),
         plt.Line2D([], [], marker="v", ms=8, ls="", color=S_SDC,
                    label="no events — upper limit")]
    leg = fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.985, 0.985),
                     frameon=False, fontsize=8.6, ncol=2, handletextpad=0.4,
                     columnspacing=1.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.075, 0.945, "Cost versus residual damage — down and left is better",
             fontsize=12.6, color=INK, ha="left")
    fig.text(0.075, 0.903,
             f"yolov8n \u00b7 bits 31-23, identical injection sites across configurations \u00b7 "
             f"weights {W['undefended']['n']:,} inferences/config, biases {B['undefended']['n']:,}",
             fontsize=8.6, color=INK2, ha="left")
    fig.text(0.075, 0.862,
             "DUE is not plotted \u2014 every configuration eliminates it entirely, and does so "
             "structurally: in_range() is a positive test, so NaN and Inf can never pass it.",
             fontsize=8.3, color=INK2, ha="left")

    os.makedirs(f"{ROOT}/figures", exist_ok=True)
    fig.savefig(OUT, dpi=190, facecolor=SURFACE)
    print(f"-> {OUT}\n")
    print(f"{'config':<18}{'KB':>8}{'w SDC%':>9}{'w 95% CI':>16}"
          f"{'b SDC%':>9}{'b 95% CI':>16}{'DUE w/b':>12}")
    for c in ["undefended"] + cfgs:
        k = KB.get(c, 0.0)
        ws, bs = 100 * W[c]["sdc"] / W[c]["n"], 100 * B[c]["sdc"] / B[c]["n"]
        wl, wh = wilson(W[c]["sdc"], W[c]["n"]); bl, bh = wilson(B[c]["sdc"], B[c]["n"])
        print(f"{SHORT.get(c, c):<18}{k:>8.2f}{ws:>9.3f}{f'[{wl:.3f},{wh:.3f}]':>16}"
              f"{bs:>9.3f}{f'[{bl:.3f},{bh:.3f}]':>16}"
              f"{100*W[c]['due']/W[c]['n']:>6.3f}/{100*B[c]['due']/B[c]['n']:<6.3f}")


if __name__ == "__main__":
    main()
