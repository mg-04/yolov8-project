"""Shared fault-injection primitives for the weight and activation campaigns.

Keeping these in one place means campaign.py and campaign_act.py classify
outcomes identically, so their results are directly comparable.

The central idea here is the three-way reliability taxonomy:

  masked / benign  output unchanged (or only boxes drifted)
  DUE              wrong AND the output carries NaN/Inf -> DETECTABLE at runtime
  SDC              wrong AND the output is structurally clean -> SILENT

Separating DUE from SDC matters because a DUE is recoverable (check for NaN,
re-run the inference) while an SDC is not detectable from the output alone.
"""
import cv2
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression

CONF, IOU_NMS = 0.25, 0.45
MATCH_IOU = 0.50      # two boxes are the same object above this
UNCHANGED_IOU = 0.99  # below this a matched box counts as having drifted

_lb = LetterBox((640, 640))


def flip(t, bit):
    """Reinterpret as int32, toggle one bit, reinterpret back. NOT a conversion."""
    return (t.view(torch.int32) ^ (1 << bit)).view(torch.float32)


def make_detector(core, device="cuda"):
    """Return a detect(path) that drives `core` directly.

    Deliberately avoids YOLO.predict(): predict() wraps the model in AutoBackend
    whose internal modules are DIFFERENT objects, which silently orphans any
    forward hook registered on the original modules.

    Returns (detections, n_nan, n_inf) so callers can tell DUE from SDC.
    """
    # Preprocessing (imread + letterbox + HWC->CHW) costs ~60ms and is identical
    # for every injection, so cache the ready-to-run GPU tensor per image. At
    # 4.9MB each, all 128 coco128 images fit in well under 1GB.
    cache = {}

    def detect(path):
        x = cache.get(path)
        if x is None:
            im = _lb(image=cv2.imread(path))
            x = (torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy())
                 .float().div(255).unsqueeze(0).to(device))
            cache[path] = x
        with torch.no_grad():
            out = core(x)
        t = out[0] if isinstance(out, (list, tuple)) else out
        n_nan = int(torch.isnan(t).sum())
        n_inf = int(torch.isinf(t).sum())
        det = non_max_suppression(t.float(), CONF, IOU_NMS)[0]
        dets = [(int(d[5]), float(d[4]), [float(v) for v in d[:4]]) for d in det]
        return dets, n_nan, n_inf

    return detect


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare(gold, corr):
    """Greedy IoU match of two detection sets -> (category, lost, phantom, class_flip)."""
    if not gold and not corr:
        return "masked", 0, 0, 0
    used, matched, cls_flip = set(), 0, 0
    min_iou = 1.0
    for g in gold:
        best, bj = 0.0, None
        for j, c in enumerate(corr):
            if j in used:
                continue
            v = iou(g[2], c[2])
            if v > best:
                best, bj = v, j
        if bj is not None and best >= MATCH_IOU:
            used.add(bj)
            matched += 1
            min_iou = min(min_iou, best)
            if corr[bj][0] != g[0]:
                cls_flip += 1
    lost = len(gold) - matched
    phantom = len(corr) - len(used)
    if lost == 0 and phantom == 0 and cls_flip == 0:
        cat = "masked" if min_iou >= UNCHANGED_IOU else "benign"
    elif cls_flip:
        cat = "class_flip"
    elif lost:
        cat = "object_lost"
    else:
        cat = "phantom"
    return cat, lost, phantom, cls_flip


def outcome(gold, corr, n_nan, n_inf):
    """Full three-way classification.

    Returns (verdict, category, lost, phantom, class_flip, loud) where verdict is
    one of masked / benign / DUE / SDC.

    A 'loud' output (NaN or Inf present) is DUE regardless of what happened to the
    detections: the fault announced itself, so a runtime monitor catches it. Only a
    structurally clean but wrong output is a true SDC.
    """
    cat, lost, phantom, cf = compare(gold, corr)
    loud = (n_nan > 0) or (n_inf > 0)
    if loud:
        verdict = "DUE"
    elif cat in ("masked", "benign"):
        verdict = cat
    else:
        verdict = "SDC"
    return verdict, cat, lost, phantom, cf, loud


VERDICTS = ("masked", "benign", "SDC", "DUE")
CATEGORIES = ("masked", "benign", "object_lost", "phantom", "class_flip")
