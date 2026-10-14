#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Census of high-magnitude weights (|w| >= 2) across model architectures.

The fault-injection results reduce to one static predictor: weights with
|w| >= 2 have the FP32 exponent MSB set, which makes them damageable at nine
bit positions instead of one. That needs no injection to measure -- it is a
property of the checkpoint -- so this surveys architectures cheaply to decide
where injection time is worth spending.

Usage:
  ./analysis/multibit/magnitude_census.py yolov8n yolov8x yolo11x rtdetr-l
"""
import logging, sys
logging.getLogger("ultralytics").setLevel(logging.ERROR)
import torch
from ultralytics import YOLO, RTDETR

DEFAULT = ["yolov8n", "yolov8x", "yolo11x", "yolov9e", "yolov10x", "rtdetr-l"]

print(f"{'model':12s} {'params':>12s} {'tensors':>8s} {'|w|>=2':>7s} {'share':>9s} "
      f"{'layers':>7s} {'max|w|':>8s} {'top-1':>7s}")
for name in sys.argv[1:] or DEFAULT:
    try:
        cls = RTDETR if name.startswith("rtdetr") else YOLO
        m = cls(f"{name}.pt")
        try:
            m.model.fuse()          # RT-DETR and some heads do not support fuse
        except Exception:
            pass
        core = m.model.eval()
    except Exception as e:
        print(f"{name:12s} skipped: {str(e)[:60]}")
        continue

    P = sum(p.numel() for p in core.parameters())
    nt = sum(1 for _ in core.parameters())
    # weight tensors only: 4-D conv kernels and 2-D linear matrices
    rows = []
    for n, p in core.named_parameters():
        if p.dim() not in (2, 4):
            continue
        k = int((p.abs() >= 2).sum())
        if k:
            rows.append((k, n, p.numel(), float(p.abs().max())))
    rows.sort(reverse=True)
    tot = sum(r[0] for r in rows)
    mx = max((r[3] for r in rows), default=0.0)
    top = 100 * rows[0][0] / tot if tot else 0.0
    print(f"{name:12s} {P:12,} {nt:8d} {tot:7d} {100*tot/P:8.4f}% {len(rows):7d} "
          f"{mx:8.2f} {top:6.1f}%")
    for k, n, t, mxl in rows[:3]:
        print(f"             {n:40s} {k:5d}/{t:<10,} {100*k/t:6.2f}%  max {mxl:.2f}")
