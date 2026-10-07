#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Report tables for the geometry / event sweeps (Sweep B).

Reads results/geom_*.csv and results/events_*.csv, concatenating any --slice
outputs automatically, and prints:

  geometry   Q1: how far boxes move in BENIGN outcomes, by bit and by layer
  events     Q2: what gets lost / invented / mislabelled, against a golden control
  summary    error-class rates by bit and by layer

The golden control is recomputed from the dataset the CSV rows name, so the
comparison distributions always match the run rather than a hardcoded baseline.

Usage:
  ./analysis/geometry/geom_report.py                      all tables
  ./analysis/geometry/geom_report.py geometry events
  ./analysis/geometry/geom_report.py --mode act           act runs instead of weight
  ./analysis/geometry/geom_report.py --csv out.csv        dump the per-bit table
"""
import csv, glob, os, statistics as st, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

argv = sys.argv[1:]
MODE = argv[argv.index("--mode") + 1] if "--mode" in argv else "weight"
IMAGES = int(argv[argv.index("--images") + 1]) if "--images" in argv else None
CSV_OUT = argv[argv.index("--csv") + 1] if "--csv" in argv else None
_VALUED = {"--mode", "--csv", "--images"}
TABLES, skip = [], False
for a in argv:
    if skip:
        skip = False
        continue
    if a.startswith("--"):
        skip = a in _VALUED
        continue
    TABLES.append(a)
TABLES = TABLES or ["summary", "geometry", "events"]

# every bit present in the data, high to low, so new bit positions show up
# automatically instead of being silently dropped by a hardcoded list
BIT_ORDER = None   # filled in after the CSVs load


def load(prefix):
    """Concatenate every CSV for this mode, including --slice shards."""
    rows, files = [], sorted(glob.glob(os.path.join(RESULTS, f"{prefix}_{MODE}_*.csv")))
    for f in files:
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows, files


GEOM, GEOM_FILES = load("geom")
EVENTS, EVENT_FILES = load("events")
if not GEOM:
    sys.exit(f"no geom_{MODE}_*.csv in {RESULTS}")

# dedupe: an older run and a newer one can both contain the same (layer, bit, inj)
seen, uniq = set(), []
for r in GEOM:
    k = (r["layer"], r["bit"], r.get("weight_idx", ""), r.get("chan", ""),
         r.get("fy", ""), r.get("fx", ""), r.get("images", ""))
    if k not in seen:
        seen.add(k)
        uniq.append(r)
GEOM = uniq

# Different image-set sizes are different experiments, so RAW COUNTS must never be
# pooled. Per-image RATES are normalised and can sit in one table, provided each row
# says which set it came from. So: keep every bit, but for any bit measured more than
# once keep only its largest image set, and report the count per row.
if IMAGES is not None:
    GEOM = [r for r in GEOM if int(r["images"]) == IMAGES]
else:
    best = {}
    for r in GEOM:
        b, im = int(r["bit"]), int(r["images"])
        if b not in best or im > best[b]:
            best[b] = im
    dropped = sorted({(int(r["bit"]), int(r["images"])) for r in GEOM
                      if int(r["images"]) != best[int(r["bit"])]})
    GEOM = [r for r in GEOM if int(r["images"]) == best[int(r["bit"])]]
    if dropped:
        print("!! superseded runs ignored (bit, images): " +
              ", ".join(f"{b}@{i}" for b, i in dropped))
    if len(set(best.values())) > 1:
        print("!! bits measured on different image sets: " +
              ", ".join(f"bit {b}={i} imgs" for b, i in sorted(best.items())))
        print("!! rates are per-image so they are comparable; raw counts are not\n")
# events carry no image-set column, so key them to the (layer, bit) pairs that
# survived the filter -- otherwise a run at a different image count leaks in
_keys = {(r["layer"], r["bit"]) for r in GEOM}
EVENTS = [x for x in EVENTS if (x["layer"], x["bit"]) in _keys]

print(f"mode={MODE}  {len(GEOM):,} injections, {len(EVENTS):,} events")
BIT_ORDER = sorted({int(r["bit"]) for r in GEOM}, reverse=True)
print(f"layers={len({r['layer'] for r in GEOM})}  "
      f"bits={sorted({int(r['bit']) for r in GEOM})}\n")


def fnum(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None


# --------------------------------------------------------------- summary
def table_summary():
    print("=== error-class rate by bit (per image) ===\n")
    print(f"{'bit':>4}{'imgs':>7}{'injections':>12}{'masked':>9}{'benign':>9}{'SDC':>9}{'DUE':>9}")
    agg = defaultdict(Counter)
    for r in GEOM:
        a = agg[int(r["bit"])]
        for k in ("v_masked", "v_benign", "v_sdc", "v_due"):
            a[k] += int(r[k])
        a["n"] += 1
        a["img"] += int(r["images"])
        a["imgset"] = int(r["images"])
    for b in [x for x in BIT_ORDER if x in agg]:
        a = agg[b]
        t = a["img"]
        print(f"{b:>4}{a['imgset']:>7}{a['n']:>12,}" + "".join(
            f"{100*a[k]/t:>8.2f}%" for k in ("v_masked", "v_benign", "v_sdc", "v_due")))
    print()


# -------------------------------------------------------------- geometry
def table_geometry():
    print("=== Q1: box displacement in BENIGN outcomes ===")
    print("    shift is centre displacement / golden box diagonal, so 0.001 = 0.1% of box size\n")
    print(f"{'bit':>4}{'imgs':>7}{'inj w/ benign':>15}{'mean IoU':>10}{'min IoU':>9}"
          f"{'shift/diag':>12}{'area ratio':>12}")
    for b in BIT_ORDER:
        rows = [r for r in GEOM if int(r["bit"]) == b and int(r["v_benign"]) > 0
                and fnum(r, "mean_iou") is not None]
        if not rows:
            continue
        mi = [fnum(r, "mean_iou") for r in rows]
        mn = [fnum(r, "min_iou") for r in rows if fnum(r, "min_iou") is not None]
        sh = [fnum(r, "mean_shift") for r in rows if fnum(r, "mean_shift") is not None]
        ar = [fnum(r, "mean_area_ratio") for r in rows if fnum(r, "mean_area_ratio") is not None]
        print(f"{b:>4}{int(rows[0]['images']):>7}{len(rows):>15,}"
              f"{st.mean(mi):>10.4f}{st.mean(mn):>9.4f}"
              f"{st.mean(sh):>12.4f}{st.mean(ar):>12.4f}")

    print("\n=== box displacement by layer (bits pooled, benign only) ===\n")
    L = defaultdict(list)
    for r in GEOM:
        if int(r["v_benign"]) > 0 and fnum(r, "mean_shift") is not None:
            L[r["layer"]].append((fnum(r, "mean_iou"), fnum(r, "mean_shift")))
    print(f"{'layer':<26}{'n':>6}{'mean IoU':>10}{'shift/diag':>12}")
    for k, v in sorted(L.items(), key=lambda kv: -st.mean([s for _, s in kv[1]])):
        print(f"{k:<26}{len(v):>6}{st.mean([i for i, _ in v]):>10.4f}"
              f"{st.mean([s for _, s in v]):>12.4f}")
    print()


# ---------------------------------------------------------------- events
def golden_control():
    """Recompute the golden detection distribution for the dataset the rows used."""
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    import fi_lib
    from ultralytics import YOLO
    sz = max({int(r["images"]) for r in GEOM})
    cfg = {127: "coco-val2017-sub127.yaml",
           1024: "coco-val2017-sub1024.yaml"}.get(sz)
    if cfg is None:
        return None, None
    m = YOLO(os.path.join(ROOT, "yolov8n.pt"))
    m.model.fuse()
    det = fi_lib.make_detector(m.model.cuda().eval())
    gold = [d for p in fi_lib.dataset_images(cfg) for d in det(p)[0]]
    return gold, cfg


def table_events():
    if not EVENTS:
        print("(no events files)\n")
        return
    import fi_lib
    gold, cfg = golden_control()
    print(f"=== Q2: what gets lost / invented / mislabelled ===")
    print(f"    golden control recomputed from {cfg}\n")
    print(f"{'':<14}{'n':>10}{'small':>8}{'med':>8}{'large':>8}{'med conf':>10}")
    if gold:
        gs = Counter(fi_lib.scale_of(d[2]) for d in gold)
        gt = sum(gs.values())
        print(f"{'golden':<14}{gt:>10,}" + "".join(
            f"{100*gs[k]/gt:>7.0f}%" for k in ("P3", "P4", "P5")) +
            f"{st.median([d[1] for d in gold]):>10.3f}")
    for kind in ("lost", "phantom", "class_flip"):
        ev = [x for x in EVENTS if x["event"] == kind]
        if not ev:
            continue
        s = Counter(x["scale"] for x in ev)
        t = sum(s.values())
        print(f"{kind:<14}{len(ev):>10,}" + "".join(
            f"{100*s[k]/t:>7.0f}%" for k in ("P3", "P4", "P5")) +
            f"{st.median([float(x['conf']) for x in ev]):>10.3f}")

    ph = [float(x["conf"]) for x in EVENTS if x["event"] == "phantom"]
    if len(ph) > 100:
        q = st.quantiles(ph, n=100)
        print("\nphantom confidence percentiles: " +
              "  ".join(f"p{p}={q[p-1]:.3f}" for p in (25, 50, 75, 90, 99)))

    print("\nevents by bit:")
    print(f"{'bit':>4}{'lost':>10}{'phantom':>10}{'class_flip':>12}")
    by = defaultdict(Counter)
    for x in EVENTS:
        by[int(x["bit"])][x["event"]] += 1
    for b in [x for x in BIT_ORDER if x in by]:
        c = by[b]
        print(f"{b:>4}{c['lost']:>10,}{c['phantom']:>10,}{c['class_flip']:>12,}")
    print()


RUN = {"summary": table_summary, "geometry": table_geometry, "events": table_events}
for name in TABLES:
    if name not in RUN:
        sys.exit(f"unknown table '{name}'; choose from {', '.join(RUN)}")
    RUN[name]()

if CSV_OUT:
    rows = []
    agg = defaultdict(Counter)
    for r in GEOM:
        a = agg[int(r["bit"])]
        for k in ("v_masked", "v_benign", "v_sdc", "v_due"):
            a[k] += int(r[k])
        a["img"] += int(r["images"])
    for b, a in sorted(agg.items()):
        rows.append(dict(bit=b, **{k: round(100 * a[k] / a["img"], 3)
                                   for k in ("v_masked", "v_benign", "v_sdc", "v_due")}))
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CSV_OUT}")
