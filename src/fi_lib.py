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


# ---------------------------------------------------------------------------
# Detailed comparison, used by the geometry / event sweeps.
#
# compare() above returns counts only, which is all the original campaigns
# needed. These add the per-match geometry and per-event identity that the
# benign-shift and SDC-characterisation studies require. Kept separate so the
# existing campaign CSVs stay reproducible.
# ---------------------------------------------------------------------------

def box_geom(a, b):
    """Geometry of one matched pair -> (iou, centre shift / diagonal, area ratio)."""
    v = iou(a, b)
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    diag = ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5 or 1.0
    shift = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / diag
    area_a = (a[2] - a[0]) * (a[3] - a[1]) or 1.0
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return v, shift, area_b / area_a


def scale_of(box, imgsz=640):
    """Which pyramid level would own an object this size: P3 small, P4 medium, P5 large.

    COCO's own small/medium/large split is 32^2 and 96^2 pixels, which maps onto
    the three detection strides closely enough to reuse.
    """
    area = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    side = area ** 0.5
    return "P3" if side < 32 else ("P4" if side < 96 else "P5")


def compare_detailed(gold, corr):
    """Greedy IoU match, returning geometry per match and identity per event.

    Returns a dict:
      category                  as compare()
      lost / phantom / flips    lists of per-event dicts, not just counts
      matches                   [(iou, centre_shift, area_ratio, class_kept), ...]
      min_iou / mean_iou        over matched pairs, None when nothing matched
      mean_shift                mean centre shift, in units of the golden box diagonal
    """
    used, matches, flips, lost = set(), [], [], []
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
            c = corr[bj]
            matches.append(box_geom(g[2], c[2]) + (c[0] == g[0],))
            if c[0] != g[0]:
                flips.append(dict(cls=g[0], cls_to=c[0], conf=g[1],
                                  area=(g[2][2] - g[2][0]) * (g[2][3] - g[2][1]),
                                  scale=scale_of(g[2]), box=g[2]))
        else:
            lost.append(dict(cls=g[0], cls_to=-1, conf=g[1],
                             area=(g[2][2] - g[2][0]) * (g[2][3] - g[2][1]),
                             scale=scale_of(g[2]), box=g[2]))
    phantom = [dict(cls=c[0], cls_to=-1, conf=c[1],
                    area=(c[2][2] - c[2][0]) * (c[2][3] - c[2][1]),
                    scale=scale_of(c[2]), box=c[2])
               for j, c in enumerate(corr) if j not in used]

    ious = [m[0] for m in matches]
    if not lost and not phantom and not flips:
        cat = "masked" if (not ious or min(ious) >= UNCHANGED_IOU) else "benign"
    elif flips:
        cat = "class_flip"
    elif lost:
        cat = "object_lost"
    else:
        cat = "phantom"
    return dict(
        category=cat, matches=matches, lost=lost, phantom=phantom, flips=flips,
        n_matched=len(matches),
        min_iou=min(ious) if ious else None,
        mean_iou=sum(ious) / len(ious) if ious else None,
        mean_shift=sum(m[1] for m in matches) / len(matches) if matches else None,
    )
