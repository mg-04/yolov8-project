#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Inspect YOLOv8n structure.

  ./structure.py              summary + per-block shapes (measured at 640x640)
  ./structure.py --convs      all 64 nn.Conv2d layers with shapes
  ./structure.py --detailed   Ultralytics per-layer table (every leaf module)
  ./structure.py --block 9    full module tree for one top-level block
  ./structure.py --raw        torch's own repr of the whole model
"""
import os
import logging, sys
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

logging.getLogger("ultralytics").setLevel(logging.ERROR)
args = sys.argv[1:]

m = YOLO(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8n.pt"))
m.model.fuse()
m.model.eval()
core = m.model

if "--raw" in args:
    print(core)
    sys.exit()

if "--block" in args:
    i = int(args[args.index("--block") + 1])
    print(f"=== block {i}: {type(core.model[i]).__name__} ===")
    print(core.model[i])
    n = sum(p.numel() for p in core.model[i].parameters())
    print(f"\nparameters: {n:,}")
    sys.exit()

if "--detailed" in args:
    model_info(core, detailed=True, verbose=True)
    sys.exit()

if "--convs" in args:
    print(f"{'#':>3} {'name':<28}{'out':>5}{'in':>5}{'k':>3}{'params':>10}")
    for i, (name, mod) in enumerate(
        (n, x) for n, x in core.named_modules() if isinstance(x, torch.nn.Conv2d)
    ):
        o, ci, kh, _ = mod.weight.shape
        print(f"{i:>3} {name:<28}{o:>5}{ci:>5}{kh:>3}{mod.weight.numel():>10,}")
    sys.exit()

# default: summary + per-block output shapes from a real forward pass
logging.getLogger("ultralytics").setLevel(logging.INFO)  # info() logs at INFO
core.info()
logging.getLogger("ultralytics").setLevel(logging.ERROR)
print()
shapes = {}
hooks = [
    blk.register_forward_hook(lambda mo, i_, o, k=i: shapes.__setitem__(k, o))
    for i, blk in enumerate(core.model)
]
with torch.no_grad():
    core(torch.zeros(1, 3, 640, 640))
for h in hooks:
    h.remove()

NOTE = {4: "P3 skip source", 6: "P4 skip source", 9: "P5 skip source",
        15: "-> Detect P3", 18: "-> Detect P4", 21: "-> Detect P5"}
print(f"{'#':>3} {'block':<12}{'output shape':<22}{'stride':>7}{'params':>10}  note")
for i, blk in enumerate(core.model):
    o = shapes[i]
    t = o[0] if isinstance(o, (list, tuple)) else o
    shape = str(tuple(t.shape)) if torch.is_tensor(t) else "?"
    stride = 640 // t.shape[2] if torch.is_tensor(t) and t.dim() == 4 else ""
    p = sum(x.numel() for x in blk.parameters())
    print(f"{i:>3} {type(blk).__name__:<12}{shape:<22}{stride:>7}{p:>10,}  {NOTE.get(i,'')}")
