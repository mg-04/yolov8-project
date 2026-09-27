#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Activation (feature-map) bit-flips vs weight bit-flips, same layer.

A weight fault is persistent and reused across every spatial position;
an activation fault corrupts one value in one feature map for one image.
"""
import contextlib, io, logging, torch
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)
WEIGHTS = "/home/mgong2/tools/yolov8-project/yolov8n.pt"
LAYER = "model.15.cv2.conv"


def load():
    m = YOLO(WEIGHTS)
    m.model.fuse()
    return m


def evaluate(m):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = m.val(data="coco8.yaml", device=0, verbose=False, plots=False)
    return r.box.map, r.box.map50


def flip(t, bit):
    return (t.view(torch.int32) ^ (1 << bit)).view(torch.float32)


def act_stats():
    """Distribution of activations at the target layer, vs weights."""
    m = load()
    seen = []
    layer = dict(m.model.named_modules())[LAYER]
    h = layer.register_forward_hook(lambda mod, i, o: seen.append(o.detach().flatten().cpu()))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        m.val(data="coco8.yaml", device=0, verbose=False, plots=False)
    h.remove()
    a = torch.cat(seen)
    w = layer.weight.data.flatten().cpu()
    for name, t in (("activations", a), ("weights", w)):
        bit30_clear = ((t.view(torch.int32) >> 30) & 1 == 0).float().mean().item()
        print(
            f"{name:<12} n={t.numel():>9,}  mean {t.mean():>8.4f}  "
            f"absmax {t.abs().max():>9.4f}  bit30-clear {bit30_clear*100:5.2f}%"
        )
    return a


def run_act(bit, idx=1234):
    """Flip one element of this layer's output, every forward pass."""
    m = load()
    layer = dict(m.model.named_modules())[LAYER]

    # register callback function
    def hook(mod, inp, out):
        flat = out.view(-1)
        flat[idx] = flip(flat[idx], bit)
        return out

    h = layer.register_forward_hook(hook)
    r = evaluate(m)
    h.remove()
    return r


def run_weight(bit, idx=1234):
    m = load()
    layer = dict(m.model.named_modules())[LAYER]
    flat = layer.weight.data.view(-1)
    flat[idx] = flip(flat[idx], bit)
    return evaluate(m)


print(f"=== distributions at {LAYER} ===")
act_stats()
print()

g = evaluate(load())
print(f"=== mAP50-95 impact (golden = {g[0]:.4f}) ===")
print(f"{'bit':>5}  {'weight flip':>14}  {'activation flip':>18}")
for bit in (31, 30, 29, 27, 24, 23, 20, 10):
    w = run_weight(bit)[0]
    a = run_act(bit)[0]
    print(f"{bit:>5}  {w:>14.4f}  {a:>18.4f}")
