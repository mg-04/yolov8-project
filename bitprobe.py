#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Probe: how much does a single weight bit-flip perturb the value, by bit position?"""
import torch, logging
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

m = YOLO("/home/mgong2/tools/yolov8-project/yolov8n.pt")
m.model.fuse()
core = m.model

convs = [(n, mod) for n, mod in core.named_modules() if isinstance(mod, torch.nn.Conv2d)]
name, layer = convs[len(convs) // 2]  # a mid-network conv
w = layer.weight.data.flatten()
idx = 1234
orig = w[idx].item()

print(f"layer: {name}  shape {tuple(layer.weight.shape)}  |W|={w.numel()}")
print(f"weight[{idx}] = {orig:.8f}")
print(f"weight stats: mean {w.mean():.5f}  std {w.std():.5f}  absmax {w.abs().max():.5f}")
print()
print(f"{'bit':>4} {'field':>9} {'flipped value':>22} {'|ratio|':>14}")
for b in range(31, -1, -1):
    as_int = torch.tensor([orig], dtype=torch.float32).view(torch.int32)
    flipped = (as_int ^ (1 << b)).view(torch.float32).item()
    if b == 31:
        field = "sign"
    elif b >= 23:
        field = "exponent"
    else:
        field = "mantissa"
    ratio = abs(flipped / orig) if orig != 0 else float("inf")
    print(f"{b:>4} {field:>9} {flipped:>22.6e} {ratio:>14.3e}")
