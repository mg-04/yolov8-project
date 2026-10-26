# Sweep B: box geometry and SDC event identity

The main campaigns record *counts* — how many images were SDC. This sweep records
**what the damage looked like**: how far each surviving box moved, and which objects
were lost, invented or mislabelled.

**Read [../../README.md](../../README.md) first** for the fault model and metrics.

## Scripts

| script | what it does |
|---|---|
| `geom_sweep.py` | the sweep. Writes `geom_<mode>_n<N>.csv` (one row per injection) and `events_<mode>_n<N>.csv` (one row per lost / phantom / class_flip event) |
| `geom_report.py` | tables from those CSVs, with a golden control recomputed from the dataset the rows name |
| `plot_geom.py` | per-layer SDC rate and per-layer box displacement, in depth order |
| `plot_drift.py` | box area drift (signed) and centre displacement per layer |

```bash
./analysis/geometry/geom_sweep.py weight 128 30 29 22
./analysis/geometry/geom_report.py                   # all tables
./analysis/geometry/geom_report.py geometry events
./analysis/geometry/geom_report.py --mode act
./analysis/geometry/plot_geom.py --bits 30 25 26 29 22 --images 1024
./analysis/geometry/plot_drift.py --images 1024
```

## Two design choices worth knowing

**A representative layer subset, not all 64.** 8 layers × 128 samples costs the same
as 64 × 16 but gives 8× the per-layer power. The subset spans depth, weight magnitude,
branch type and fault locality — so these numbers describe the *distribution* of
damage and are **not** a basis for ranking layers. For rankings see
[../README.md](../README.md).

**Geometry is recorded, not thresholded.** The main harness computes IoU per matched
pair and keeps the minimum for a pass/fail test. Here IoU, centre shift (normalised by
the golden box diagonal, so pure translation) and area ratio (pure scale) are all
kept, because IoU alone conflates translation with resizing.

---

## What the Damage Actually Looks Like

> `./analysis/geometry/geom_sweep.py weight 128 30 29 22`. A separate sweep over 8
> representative layers x 128 samples x 3 bits = 3,072 injections, 16 images each,
> recording per-match box geometry and per-event object identity rather than counts.
> 162,652 events. The layer subset spans depth, weight magnitude, branch type and
> fault locality; it is not a basis for ranking layers.

Golden baseline on the 16-image subset: **50 detections, 3.12 per image**, median
object side 150px.

### Benign outcomes are sub-pixel wobble

| bit | injections | mean IoU | min IoU | centre shift / diagonal | area ratio |
|---|---|---|---|---|---|
| 22 | 260 | 0.975 | 0.860 | 0.0057 | 1.016 |
| 29 | 284 | 0.986 | 0.908 | 0.0032 | 0.995 |
| 30 | 199 | 0.944 | 0.688 | 0.0151 | 0.966 |

Centre shift is normalized by the golden box diagonal, so 0.003-0.015 means the box
moved **0.3-1.5% of its own size**. On a median 150px object that is **0.6-3.2
pixels**. Area ratios sit at ~1.0, so it is translation, not resizing: measuring IoU
alone would have conflated the two.

So the outcome distribution is sharply bimodal. A fault either nudges a box by a
pixel or two, or it destroys/invents a detection. There is almost nothing between.

**One layer breaks the pattern.** Per-layer box shift at bit 29:

| layer | n | mean IoU | shift / diagonal |
|---|---|---|---|
| **`model.22.dfl.conv`** | 11 | **0.895** | **0.0211** |
| `model.0.conv` | 69 | 0.978 | 0.0053 |
| `model.2.cv2.conv` | 90 | 0.990 | 0.0025 |
| `model.4.cv2.conv` | 65 | 0.995 | 0.0010 |
| `model.15.cv2.conv` | 40 | 0.997 | 0.0005 |
| `model.5.conv` | 5 | 0.999 | 0.0003 |

`model.22.dfl.conv` distorts geometry 4-7x more than any other layer. That is
mechanically exact: it is the fixed projection that decodes box distributions into
coordinates, so corrupting it perturbs box position *directly*, while every other
layer can only affect boxes through the detection pipeline.

### Object loss is unbiased; phantoms are not

Each event compared against the golden distribution it was drawn from:

| | small | medium | large | median conf |
|---|---|---|---|---|
| **golden** (control) | 22% | 18% | 60% | 0.555 |
| lost (40,966) | 23% | 19% | 58% | 0.548 |
| phantom (120,716) | 34% | 52% | 15% | **1.000** |
| class_flip (970) | 29% | 25% | 46% | 0.447 |

**Object loss carries no size or confidence bias.** Every column is within a point
or two of golden. Corruption does not pick off small or marginal objects first, it
removes them indiscriminately. This contradicts the obvious prediction.

**Phantoms arrive at confidence 1.000.** Median and near-maximal mean. This is the
most safety-relevant result here: corruption-induced false detections cannot be
filtered by a confidence threshold, because they are *more* confident than any real
detection. They also cluster at medium size (52% vs 18% in golden) and almost never
appear large (15% vs 60%).

**Class flips hit marginal detections** at confidence 0.447 against a 0.555 baseline
— the one place where "the uncertain ones break first" holds.

### Caveats

- The size buckets are labelled `P3`/`P4`/`P5` in the CSVs but are assigned by
  **object size** (COCO's 32px / 96px thresholds), not by which detection head
  emitted the box. "Corrupted boxes are medium-sized" is supported; "the P4 head
  produces the phantoms" is not. Real provenance needs NMS to return surviving
  anchor indices, which it does not.
- All 162,652 events come from corrupting the same **50 golden objects**, so
  class-level counts mostly reflect that 20 of the 50 are people. Scale and
  confidence distributions are sounder, since each is compared against its own
  golden baseline.
- 8 layers, chosen to span the structural axes. Not a layer ranking.

## Additional Observations


### Transient outcome by flip direction (bit 30)

| direction | n | outcome |
|---|---|---|
| nan | 133 | 67 object_lost, 66 masked |
| grew | 262 | 58 lost, 41 phantom, 16 class_flip, 147 masked |
| shrank | 117 | 24 phantom, 93 masked, **never loses an object** |

`nan` arises when `|a| ∈ [1,2)`. `grew` is the only direction producing class flips.
`shrank` means bit 30 was already set, so the flip zeroes the activation,
effectively a dropout event, which never loses an object.

### DUE does not imply a wrong answer

Transient bit 30, error class against the underlying detection outcome:

```
DUE   object_lost   167     detected AND failed
DUE   masked        128     detected, output actually fine   <- 42% of DUE rows
DUE   phantom        14
DUE   benign          6
DUE   class_flip      4
```

The NaN check scans the whole `84 × 8400` output, but NMS discards most anchors, so
NaN frequently lands somewhere that never reaches a detection. **42% of DUE rows had
correct detections.** The DUE column is therefore a *detection* rate, not a failure
rate; a strict "detected and failed" reading of transient bit 30 is closer to 18%
DUE / 22% SDC / 60% masked.

