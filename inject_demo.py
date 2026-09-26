#!/afs/ece.cmu.edu/usr/mgong2/.pyenv/versions/3.11.9/bin/python3
"""Single-weight bit-flip -> mAP, as a function of which bit was hit."""
import contextlib, io, logging, torch
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)
WEIGHTS = "/home/mgong2/tools/yolov8-project/yolov8n.pt"
LAYER, IDX = "model.15.cv2.conv", 1234


def run(bit=None):
    m = YOLO(WEIGHTS)
    m.model.fuse()
    if bit is not None:
        layer = dict(m.model.named_modules())[LAYER]
        flat = layer.weight.data.view(-1)
        as_int = flat[IDX].view(torch.int32) ^ (1 << bit)
        flat[IDX] = as_int.view(torch.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = m.val(data="coco8.yaml", device=0, verbose=False, plots=False)
    return r.box.map, r.box.map50, r.box.mp, r.box.mr


g = run(None)
print(f"{'case':<22}{'mAP50-95':>10}{'mAP50':>9}{'prec':>8}{'recall':>8}   {'delta mAP':>10}")
print(f"{'golden':<22}{g[0]:>10.4f}{g[1]:>9.4f}{g[2]:>8.4f}{g[3]:>8.4f}   {'--':>10}")
for bit in (30, 29, 24, 23, 22, 18, 10, 2):
    c = run(bit)
    field = "sign" if bit == 31 else ("exp" if bit >= 23 else "mant")
    print(
        f"{'flip bit ' + str(bit) + ' (' + field + ')':<22}"
        f"{c[0]:>10.4f}{c[1]:>9.4f}{c[2]:>8.4f}{c[3]:>8.4f}   {c[0] - g[0]:>+10.4f}"
    )
