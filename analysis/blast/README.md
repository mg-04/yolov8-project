# Sweep A: blast radius

How much of the Detect output does **one** fault change? Purely structural — no NMS,
no golden detections, no mAP. One extra forward pass per injection (~20 ms, against
~1.6 s for the full detection pipeline), so it is cheap enough to sweep all 64 layers.

**Read [../../README.md](../../README.md) first** for the fault model.

```bash
./analysis/blast/blast_sweep.py weight [samples] [bits...]
./analysis/blast/blast_sweep.py act    [samples] [bits...]
```

Defaults: 16 samples, bits `31 30 29 24 23 22`, 4 images. Writes
`results/blast_<mode>_n<samples>.csv`.

## What it measures

The Detect output is `84 × 8400`, where the 8,400 are candidate positions across three
scales — P3 80×80 = 6,400, P4 40×40 = 1,600, P5 20×20 = 400. An anchor counts as
**changed** if any of its 84 values differs from golden by more than `THRESH = 1e-3`,
or became NaN/Inf.

Because every anchor has a known grid position, changed anchors have known pixel
locations. For activation faults, injected at a known `(fy, fx)`, that gives the
spatial spread of the fault directly.

| column | meaning |
|---|---|
| `n_changed`, `frac_changed` | anchors changed, of 8400 |
| `frac_p3`, `frac_p4`, `frac_p5` | the same, per pyramid scale |
| `median_dist` | median pixel distance from the injection point to a changed anchor (activation mode only) |
| `chan`, `fy`, `fx` | activation injection site; blank in weight mode |
| `weight_idx` | flat index into the weight tensor; blank in activation mode |

## Results so far

**Weight faults are global.** 6,144 injections (64 layers × 16 samples × 6 bits):

| bit | median frac changed | mean | p90 |
|---|---|---|---|
| 31 | 0.9976 | 0.6194 | 1.0000 |
| 30 | **1.0000** | 0.6520 | 1.0000 |
| 29 | 0.9966 | 0.6153 | 1.0000 |
| 24 | 0.9970 | 0.6167 | 1.0000 |
| 23 | 0.9960 | 0.6126 | 1.0000 |
| 22 | 0.9912 | 0.6069 | 0.9999 |

A single weight flip typically perturbs **every anchor in the output**, and the bit
position barely matters. That is the structural reason weight faults are so much more
damaging than activation faults: one weight is reused at every spatial position of its
output feature map, so there is no such thing as a local weight fault.

Note the gap between median (~1.0) and mean (~0.62). The distribution is not spread
out, it is split:

| | injections | share |
|---|---|---|
| < 1% of anchors changed | 780 | 12.7% |
| > 99% changed | 3,312 | 53.9% |
| in between | 2,052 | 33.4% |

So a weight fault either does nothing measurable or moves the whole output. This is
the same bimodality the geometry sweep finds in box coordinates, but measured on the
raw tensor with no detection pipeline in the way — which rules out NMS as its cause.

**Activation faults are local.** Bit 30, 64 injections (1 per layer — thin, see below):

| | |
|---|---|
| median frac changed | **0.0355** |
| mean | 0.2916 |
| p90 | 0.9056 |
| median distance to a changed anchor | 81 px |

A median of 3.6% against the weight sweep's ~100% is a **28× difference in blast
radius** at the same bit. The 81 px median distance puts a number on the locality: a
corrupted activation damages a neighbourhood, not the frame. The p90 of 0.91 says the
tail still reaches the whole output — a fault early enough in the backbone has the
depth to spread.

**Damage concentrates in P3** under both fault types — 0.52 / 0.42 mean fraction
changed at P3 against 0.21–0.29 at P5. P3 is 76% of the anchors, so this is partly
just its share, but the per-scale columns are there to separate that out with a longer
run.

## Caveats

- The activation numbers rest on **1 sample per layer at one bit** (`blast_act_n1.csv`).
  Treat them as a direction, not a measurement. `./analysis/blast/blast_sweep.py act 16 30`
  would put them on the same footing as the weight sweep.
- `IMAGES = 4`, and blast radius is image-dependent for activation faults.
- `THRESH = 1e-3` is absolute, not relative, so it is more permissive on the large
  class-logit values than on box coordinates.
- `IMG_DIR` is **hardcoded** to `/home/mgong2/datasets/coco128/images/train2017`,
  unlike every other script here, which derives paths from its own location. It will
  not run elsewhere without an edit.
- This sweep is not written up in the main findings. The headline numbers above are
  computed from the two CSVs in `results/`.
