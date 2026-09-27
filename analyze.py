#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Summarize a fault-injection campaign: per-section and per-layer vulnerability.

Reports Wilson score intervals on the critical-SDC rate, since a bare
percentage from N=16 samples carries no useful precision on its own.
"""
import csv, math, sys
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "/home/mgong2/tools/yolov8-project/campaign_results.csv"
GOLDEN = 0.4451  # coco128 mAP50-95, yolov8n


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def section(layer):
    """Which structural part of YOLOv8 does this layer belong to?"""
    i = int(layer.split(".")[1])
    if layer.startswith("model.22.dfl"):
        return "head: DFL"
    if layer.startswith("model.22.cv2"):
        return "head: box (cv2)"
    if layer.startswith("model.22.cv3"):
        return "head: class (cv3)"
    if i <= 9:
        return "backbone"
    return "neck"


rows = list(csv.DictReader(open(CSV)))
for r in rows:
    r["map"] = float(r["map"])
    r["critical"] = int(r["critical"])
    r["section"] = section(r["layer"])

print(f"{len(rows)} injections | golden mAP50-95 = {GOLDEN:.4f}\n")

# ---- by structural section ----
by_sec = defaultdict(list)
for r in rows:
    by_sec[r["section"]].append(r)

order = ["backbone", "neck", "head: box (cv2)", "head: class (cv3)", "head: DFL"]
print(f"{'section':<20}{'n':>6}{'mean mAP':>10}{'median':>9}{'crit%':>8}  {'95% CI':>16}")
for sec in order:
    rs = by_sec.get(sec, [])
    if not rs:
        continue
    maps = sorted(r["map"] for r in rs)
    k = sum(r["critical"] for r in rs)
    lo, hi = wilson(k, len(rs))
    med = maps[len(maps) // 2]
    print(f"{sec:<20}{len(rs):>6}{sum(maps)/len(maps):>10.4f}{med:>9.4f}"
          f"{100*k/len(rs):>7.0f}%  [{100*lo:>5.1f}, {100*hi:>5.1f}]")

# ---- most and least vulnerable individual layers ----
by_layer = defaultdict(list)
for r in rows:
    by_layer[r["layer"]].append(r)

stats = []
for name, rs in by_layer.items():
    maps = [r["map"] for r in rs]
    k = sum(r["critical"] for r in rs)
    stats.append((sum(maps) / len(maps), name, k, len(rs), int(rs[0]["params"])))
stats.sort()

print(f"\n{'--- 8 most vulnerable layers ---':<48}")
print(f"{'layer':<28}{'params':>9}{'mean mAP':>10}{'crit':>8}")
for mean, name, k, n, p in stats[:8]:
    print(f"{name:<28}{p:>9,}{mean:>10.4f}{f'{k}/{n}':>8}")

print(f"\n{'--- 8 most resilient layers ---':<48}")
print(f"{'layer':<28}{'params':>9}{'mean mAP':>10}{'crit':>8}")
for mean, name, k, n, p in stats[-8:][::-1]:
    print(f"{name:<28}{p:>9,}{mean:>10.4f}{f'{k}/{n}':>8}")

# ---- outcome distribution ----
print("\n--- outcome distribution ---")
buckets = [
    ("destroyed  (mAP < 0.01)", lambda m: m < 0.01),
    ("severe     (0.01-0.50x)", lambda m: 0.01 <= m < 0.5 * GOLDEN),
    ("degraded   (0.50-0.95x)", lambda m: 0.5 * GOLDEN <= m < 0.95 * GOLDEN),
    ("masked     (> 0.95x)", lambda m: m >= 0.95 * GOLDEN),
]
for label, pred in buckets:
    n = sum(1 for r in rows if pred(r["map"]))
    bar = "#" * int(60 * n / len(rows))
    print(f"{label:<26}{n:>5} {100*n/len(rows):>5.1f}%  {bar}")
