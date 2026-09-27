#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Cross-campaign summary tables from results/*.csv.

Joins the three campaigns (weight, activation-persistent, activation-transient)
across all bit positions and prints:

  bits      SDC / DUE per bit per campaign          (the headline table)
  sections  SDC / DUE per structural section        at one bit
  layers    per-layer ranking across campaigns      at one bit
  depth     SDC / DUE by block depth                at one bit
  weighted  parameter-weighted whole-model rate     weight campaign only

Two rate definitions are reported, because they differ substantially:

  any   fraction of injections where ANY of the 16 subset images was SDC/DUE.
        Sensitive, but upward-biased -- this is the `sdc_critical` column.
  img   fraction of the 16*N individual image comparisons that were SDC/DUE.
        The unbiased per-image rate; report this one.

Usage:
  ./summarize.py                    all tables, bit 30 for the per-bit ones
  ./summarize.py --bit 29           use bit 29 where a single bit is needed
  ./summarize.py bits sections      only those tables
  ./summarize.py --csv out.csv      also dump the per-bit table as CSV
"""
import csv, glob, math, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
BITS = [31, 30, 29, 24, 23, 22]
FIELD = {31: "sign", 30: "exp MSB", 29: "exp", 24: "exp", 23: "exp", 22: "mant MSB"}
CAMPAIGNS = {
    "weight": "campaign_bit{b:02d}_n{n}.csv",
    "act-persist": "act_persistent_bit{b:02d}_n{n}.csv",
    "act-transient": "act_transient_bit{b:02d}_n{n}.csv",
}
SECTIONS = ["backbone", "neck", "head:cls", "head:box", "head:DFL"]

args = sys.argv[1:]
BIT = int(args[args.index("--bit") + 1]) if "--bit" in args else 30
CSV_OUT = args[args.index("--csv") + 1] if "--csv" in args else None
N = 16
wanted = [a for a in args if not a.startswith("--") and not a.isdigit()]
if CSV_OUT in wanted:
    wanted.remove(CSV_OUT)
TABLES = wanted or ["bits", "sections", "layers", "depth", "weighted"]


def section(layer):
    if layer.startswith("model.22.dfl"):
        return "head:DFL"
    if layer.startswith("model.22.cv2"):
        return "head:box"
    if layer.startswith("model.22.cv3"):
        return "head:cls"
    return "backbone" if int(layer.split(".")[1]) <= 9 else "neck"


def depth_group(layer):
    b = int(layer.split(".")[1])
    return ("0-4 early" if b <= 4 else "5-9 late bb" if b <= 9
            else "10-21 neck" if b <= 21 else "22 head")


def wilson(k, n, z=1.96):
    """95% Wilson score interval -- stays inside [0,1] near 0% and 100%."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load(campaign, bit, n=N):
    path = os.path.join(RESULTS, CAMPAIGNS[campaign].format(b=bit, n=n))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows or None


def tally(rows):
    """-> (sdc_any, due_any, n_inj, sdc_img, due_img, n_img)

    Transient rows hold one image each (columns `verdict`); weight and persistent
    rows hold 16 (columns `v_sdc` / `v_due`). Handle both.
    """
    sdc_any = due_any = sdc_img = due_img = n_img = 0
    for r in rows:
        sdc_any += int(r.get("sdc_critical", 0) or 0)
        due_any += int(r.get("due_critical", 0) or 0)
        if "v_sdc" in r:                       # 16 images per injection
            sdc_img += int(r["v_sdc"] or 0)
            due_img += int(r["v_due"] or 0)
            n_img += int(r.get("sdc_images", 0) or 0)
        else:                                  # transient: one image per injection
            sdc_img += int(r.get("verdict") == "SDC")
            due_img += int(r.get("verdict") == "DUE")
            n_img += 1
    return sdc_any, due_any, len(rows), sdc_img, due_img, n_img


def pct(k, n):
    return f"{100*k/n:.1f}%" if n else "   -"


# ---------------------------------------------------------------- bits
def table_bits():
    print("=== SDC / DUE by bit position "
          "(any = any of 16 imgs; img = per-image rate) ===\n")
    hdr = f"{'bit':>4} {'field':<9}"
    for c in CAMPAIGNS:
        hdr += f"{c+' SDC':>16}{c+' DUE':>16}"
    print(hdr)
    out = []
    for b in BITS:
        line = f"{b:>4} {FIELD[b]:<9}"
        rec = {"bit": b, "field": FIELD[b]}
        for c in CAMPAIGNS:
            rows = load(c, b)
            if rows is None:
                line += f"{'-':>16}{'-':>16}"
                continue
            sa, da, ni, si, di, nimg = tally(rows)
            line += f"{pct(sa,ni)+' / '+pct(si,nimg):>16}"
            line += f"{pct(da,ni)+' / '+pct(di,nimg):>16}"
            rec[f"{c}_sdc_any"] = round(100 * sa / ni, 2)
            rec[f"{c}_sdc_img"] = round(100 * si / nimg, 2) if nimg else None
            rec[f"{c}_due_any"] = round(100 * da / ni, 2)
            rec[f"{c}_due_img"] = round(100 * di / nimg, 2) if nimg else None
        print(line)
        out.append(rec)
    print("\nDUE is nonzero only where a value can reach the E=255 NaN encoding.")
    return out


# ------------------------------------------------------------ sections
def table_sections():
    print(f"\n=== SDC / DUE by section, bit {BIT} (per-image rate, 95% Wilson CI) ===\n")
    print(f"{'section':<12}" + "".join(f"{c:>30}" for c in CAMPAIGNS))
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for c in CAMPAIGNS:
        rows = load(c, BIT)
        if rows is None:
            continue
        for r in rows:
            a = agg[section(r["layer"])][c]
            s, d, _, si, di, nimg = tally([r])
            a[0] += si; a[1] += di; a[2] += nimg
    for sec in SECTIONS:
        line = f"{sec:<12}"
        for c in CAMPAIGNS:
            si, di, n = agg[sec][c]
            if not n:
                line += f"{'-':>30}"
                continue
            lo, hi = wilson(si, n)
            line += f"  SDC {100*si/n:>5.1f}% [{100*lo:>4.0f},{100*hi:>4.0f}]  DUE {100*di/n:>5.1f}%"
        print(line)


# -------------------------------------------------------------- layers
def table_layers(top=10):
    print(f"\n=== per-layer SDC%, bit {BIT} (per-image rate) ===\n")
    per = defaultdict(dict)
    for c in CAMPAIGNS:
        rows = load(c, BIT)
        if rows is None:
            continue
        a = defaultdict(lambda: [0, 0])
        for r in rows:
            _, _, _, si, _, nimg = tally([r])
            e = a[r["layer"]]; e[0] += si; e[1] += nimg
        for lay, (si, n) in a.items():
            per[lay][c] = 100 * si / n if n else float("nan")
    cols = [c for c in CAMPAIGNS if any(c in v for v in per.values())]
    if not cols:
        print("  no data")
        return
    rank = sorted(per.items(), key=lambda kv: -sum(kv[1].get(c, 0) for c in cols))
    print(f"{'layer':<26}{'section':<11}" + "".join(f"{c:>15}" for c in cols))
    for lay, d in rank[:top]:
        print(f"{lay:<26}{section(lay):<11}" + "".join(f"{d.get(c,0):>14.0f}%" for c in cols))
    print(f"{'  ... least sensitive ...':<37}")
    for lay, d in rank[-top:]:
        print(f"{lay:<26}{section(lay):<11}" + "".join(f"{d.get(c,0):>14.0f}%" for c in cols))
    allzero = [l for l, d in per.items()
               if all(d.get(c, 0) == 0 for c in cols) and len(d) == len(cols)]
    print(f"\nlayers at 0% SDC across all {len(cols)} campaigns: {len(allzero)}")
    for l in sorted(allzero):
        print("  ", l)


# --------------------------------------------------------------- depth
def table_depth():
    print(f"\n=== SDC / DUE by block depth, bit {BIT} (per-image rate) ===\n")
    print(f"{'group':<14}" + "".join(f"{c:>26}" for c in CAMPAIGNS))
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for c in CAMPAIGNS:
        rows = load(c, BIT)
        if rows is None:
            continue
        for r in rows:
            a = agg[depth_group(r["layer"])][c]
            _, _, _, si, di, nimg = tally([r])
            a[0] += si; a[1] += di; a[2] += nimg
    for g in ["0-4 early", "5-9 late bb", "10-21 neck", "22 head"]:
        line = f"{g:<14}"
        for c in CAMPAIGNS:
            si, di, n = agg[g][c]
            line += f"  SDC {100*si/n:>5.1f}%  DUE {100*di/n:>5.1f}%" if n else f"{'-':>26}"
        print(line)


# ------------------------------------------------------------ weighted
def table_weighted():
    rows = load("weight", BIT)
    if rows is None or "params" not in rows[0]:
        print("\n(weighted table needs the weight campaign CSV with a params column)")
        return
    print(f"\n=== whole-model rate, weight campaign, bit {BIT} ===\n")
    a = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        e = a[r["layer"]]
        _, _, _, si, _, nimg = tally([r])
        e[0] += si; e[1] += nimg; e[2] = int(r["params"])
    tot = sum(v[2] for v in a.values())
    plain = sum(v[0] / v[1] for v in a.values()) / len(a)
    weighted = sum((v[2] / tot) * (v[0] / v[1]) for v in a.values())
    print(f"  conv weights                  : {tot:,}")
    print(f"  plain mean of per-layer rates : {100*plain:.1f}%   <- NOT a whole-model rate")
    print(f"  parameter-weighted rate       : {100*weighted:.1f}%   <- correct")
    print("\n  Sampling is uniform per layer, so a plain mean treats a 16-param layer")
    print("  as equally likely to be hit as a 294,912-param one. Weight by params.")


# ----------------------------------------------------------- layerbits
def table_layerbits(campaign="weight", bits=(31, 30, 29), top=8):
    """Per-layer SDC across several bits -- shows that the ranking is bit-dependent."""
    print(f"\n=== per-layer SDC%, {campaign} campaign, by bit (per-image rate) ===\n")
    per = defaultdict(dict)
    overall = {}
    for b in bits:
        rows = load(campaign, b)
        if rows is None:
            continue
        a = defaultdict(lambda: [0, 0])
        tk = tn = 0
        for r in rows:
            _, _, _, si, _, nimg = tally([r])
            e = a[r["layer"]]; e[0] += si; e[1] += nimg
            tk += si; tn += nimg
        overall[b] = 100 * tk / tn if tn else 0
        for lay, (si, nn) in a.items():
            per[lay][b] = 100 * si / nn if nn else 0
    have = [b for b in bits if b in overall]
    if not have:
        print("  no data")
        return
    print("  overall: " + "   ".join(f"bit {b} {overall[b]:.1f}%" for b in have) + "\n")
    for b in have:
        rank = sorted(per.items(), key=lambda kv: -kv[1].get(b, 0))
        print(f"  --- bit {b} top {top} ---")
        print(f"  {'layer':<26}{'section':<11}" + "".join(f"{'bit '+str(x):>10}" for x in have))
        for lay, d in rank[:top]:
            print(f"  {lay:<26}{section(lay):<11}"
                  + "".join(f"{d.get(x,0):>9.0f}%" for x in have))
        print()


RUN = {"bits": table_bits, "sections": table_sections, "layers": table_layers,
       "depth": table_depth, "weighted": table_weighted,
       "layerbits": table_layerbits}

bits_rows = None
for name in TABLES:
    fn = RUN.get(name)
    if fn is None:
        sys.exit(f"unknown table '{name}'; choose from {', '.join(RUN)}")
    res = fn()
    if name == "bits":
        bits_rows = res

if CSV_OUT and bits_rows:
    keys = sorted({k for r in bits_rows for k in r}, key=lambda k: (k != "bit", k))
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(bits_rows)
    print(f"\nwrote {CSV_OUT}")
