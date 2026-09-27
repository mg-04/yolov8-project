# Implementation notes

Harness details for [Hardware Fault Injection in YOLOv8](README.md) — code layout,
correctness guarantees, bugs found along the way, and how to reproduce the runs.
The fault model, method and results live in the README.

## Scripts

| script | role |
|---|---|
| `fi_lib.py` | Shared primitives, imported by both campaigns so they classify identically: `flip`, `iou`, `compare`, `make_detector`, `outcome`. |
| `campaign.py` | Weight campaign, all 64 conv layers. `./campaign.py [samples] [bit]` |
| `campaign_act.py` | Activation campaign. `./campaign_act.py {persistent\|transient} [samples] [bit]` |
| `analyze.py` | Summarizes a CSV: per-section rates + Wilson intervals, per-layer ranking, outcome buckets. |
| `summarize.py` | Cross-campaign tables from `results/*.csv`: per-bit, per-section, per-layer, depth, parameter-weighted. Reports both rate definitions. `./summarize.py [bits\|sections\|layers\|depth\|weighted] [--bit N] [--csv out.csv]` |
| `structure.py` | Model inspection. `--convs` lists the 64 targets in campaign order. |
| `bitprobe.py` | No inference; flips all 32 bits of one weight. Establishes the encoding asymmetry. |
| `inject_demo.py`, `inject_act.py` | Single-layer demos, superseded. `inject_act.py`'s hook fires every pass. |
| `get_coco_val.sh` | Downloads COCO val2017 without the 19 GB train set. |

Outputs: `results/campaign_bit<BB>_n<N>.csv`,
`results/act_{persistent,transient}_bit<BB>_n<N>.csv`, plus `.log` files.

---

## Reproducing

```bash
./campaign.py 16 30                 # weights,   ~29 min (bit 30), ~9 min others
./campaign_act.py transient 16 30   # SEU,       ~40 s
./campaign_act.py persistent 16 30  # stuck-at,  ~12 min
./summarize.py                      # all cross-campaign tables
./analyze.py results/campaign_bit30_n16.csv    # single-CSV detail
./structure.py --convs              # the 64 targets, in campaign order
```

Full sweep, 3 campaigns × 6 bits, ~2.5 h:

```bash
screen -dmS sweep bash -c '
  for b in 31 30 29 24 23 22; do ./campaign.py 16 $b > results/campaign_bit${b}_n16.log 2>&1; done
  for b in 31 30 29 24 23 22; do ./campaign_act.py transient 16 $b > results/act_transient_bit${b}_n16.log 2>&1; done
  for b in 31 30 29 24 23 22; do ./campaign_act.py persistent 16 $b > results/act_persistent_bit${b}_n16.log 2>&1; done
'
```

## Correctness guarantees

All load-bearing; the last two were bugs first.

| guard | what it prevents |
|---|---|
| `orig = flat[idx].clone()` | `flat[idx]` is a **view sharing storage**. Without the clone you restore the corrupted value onto itself and corruption compounds. |
| golden re-check at end | Detects restore failure as nonzero drift. Measured `0.00e+00` over 1,024 injections. |
| `h.remove()` after hooks | A leaked hook keeps corrupting every later evaluation. |
| `random.Random(SEED)` | Local RNG — Ultralytics seeds the global one for augmentation. |
| bypass `YOLO.predict()` | `predict()` builds an `AutoBackend` wrapper holding **different module objects**, silently orphaning every forward hook. `val()` is unaffected. |
| `out[:, ...]` not `out[0, ...]` | Corrupts all batch elements. With `batch=32` over 128 images, `out[0]` reaches 4 images and the effect vanishes. |

## Bugs found and fixed

Each produced plausible-looking wrong results rather than failing.

1. **Hooks silently orphaned by `predict()`** — `predictor.model.model is
   model.model` is `False`. The *first* `predict()` call does fire hooks (before the
   wrapper exists), which masked it. Symptom: transient campaign reporting 0%
   critical with `fired=0` on all 512 rows. Fix: `fi_lib.make_detector` drives the
   model directly (letterbox → forward → NMS).
2. **Persistent mode corrupted only batch element 0** — 4 of 128 images. Symptom:
   bit 31 reporting *exactly* 0.4451 for all 64 layers. Fix: `out[:, ...]`.
3. **Unreachable `benign` category** — `lost == 0 and phantom == 0` forces equal
   counts, so the original test could never fire. Now keyed on `min_iou < 0.99`.


### Runtime scales with corruption

| bit | val() | 16×detect | 16×classify | avg dets/img |
|---|---|---|---|---|
| 22 | 0.853s | 0.059s | 0.000s | 3.1 |
| 30 | 1.840s | 0.151s | 0.003s | **127.6** |

A corrupted model emits ~128 detections per image instead of 3.1, so `val()` runs NMS
and AP over ~40× more boxes. Classification costs 3 ms. The expensive faults
(huge-but-finite) are the same ones that are silent.

### Where mAP fails

mAP-critical (`mAP < 0.5 × golden`) versus per-image SDC, weight campaign:

| bit | mAP-critical | SDC |
|---|---|---|
| 31 | 0.4% | 2.8% |
| 30 | 80.1% | 83.0% |
| 29 | 1.7% | 2.8% |
| 24 | 0.4% | 2.4% |
| 23 | 0.1% | 1.2% |
| 22 | 0.0% | 0.5% |

mAP tracks SDC well at bit 30 and understates it by 3–7× everywhere else, because
losing one object out of 50 barely moves a dataset average. It also cannot separate
SDC from DUE at all — "NaN destroyed everything" and "huge finite values produced
garbage boxes" both read `0.0000`.

# Known limitations

1. **mAP is not significance-tested.** `delta` is recorded but never tested; the 50%
   cutoff is a convention. `val()` is deterministic so deltas are reproducible — but
   measured on 128 specific images, and five random 128-image subsets of val2017
   spanned **0.4008–0.4574** (±0.057). The comparison is *paired*, so a paired
   bootstrap or Wilcoxon signed-rank over per-image AP is the proper test.
2. **Two rate definitions differ by 4–12×.** `sdc_critical` fires if *any* of the 16
   subset images shows an SDC; the per-image rate from `v_sdc` is unbiased. Weight bit
   22 is 6.2% by the first and 0.5% by the second. All results above use per-image;
   `summarize.py` prints both.
3. **Rates are conditional on the injected bit**, not field rates. A uniform-random
   single-bit weight flip gives `P(critical) ≈ (1/32) × 0.9995 × 0.911 ≈ 2.8%`
   (~1 upset in 36) — bit 30 only, so a lower bound.

4. **DUE is a detection rate, not a failure rate.** The NaN check scans the whole
   output tensor, and 42% of DUE rows had correct detections because NMS discarded the
   affected anchors. A stricter monitor would check after NMS. `fi_lib.outcome` does
   not currently separate detected-and-failed from detected-but-benign.
4. **coco128 is neither held out nor large.** All 128 images are in COCO train2017
   (verified 128/128). Golden 0.4451 vs 0.3684 held out. The inflation is mostly a
   small-sample artifact rather than contamination: five *held-out* 128-image subsets
   averaged 0.4300, because a 128-image sample holds only 71–74 of 80 classes and
   13–14 classes with ≤ 2 instances, which inflates a per-class mean.
   Under the SDC metric `model.22.cv3.2.2` is 98%/94%/94% critical, so the earlier
   claim that it looked immune due to coco128 having few large objects described an
   artifact that does not exist.
5. **FP32-specific.** Edge deployments run FP16 or INT8; INT8 bounds worst-case error
   at 128× rather than 10³⁸, so quantized models should be markedly more
   fault-tolerant.
6. **Confidence degradation is invisible** — `compare()` uses class and box only, so
   a detection whose confidence fell 0.91 → 0.26 counts as `masked`.
7. **Single-bit only.** A single particle can flip adjacent cells; multi-bit upsets
   defeat simple SEC ECC and are not modeled.
8. **n=1 per layer is unreliable** — an n=1 run showed `model.0.conv` apparently
   immune at mAP 0.4296; at n=16 it was 9/16 critical, mean 0.1888.

9. **Per-layer results are reported at bit 30 and for activations only at bit 30.**
   The cross-bit weight table shows the ranking inverts between small-weight and
   large-weight layers, so the bit-30 per-section and per-block figures should not be
   read as a bit-independent vulnerability profile.

10. **Mixing metrics across campaigns produced a wrong conclusion once.** An earlier
   per-layer table used mAP for weights and SDC for activations, which made eight
   box-branch layers look "0% critical across all fault types." On a consistent SDC
   metric no layer is at 0%. Always confirm which column a cross-campaign table used.

# Next steps

- Harden `model.22.dfl.conv` (**16 parameters**) — most sensitive layer in the network
  at bits 29/24/31, and triplicating 16 values is free. Highest value per byte by a
  wide margin.
- Harden the three `cv3.*.2` class-output convs (19,200 params, 0.6%) — highest
  absolute sensitivity at bit 30, across all three fault types
- Extend the per-layer ranking to bits 24/23/22 and to the activation campaigns, to
  see whether the large-weight/small-weight split holds throughout
- Split DUE into detected-and-failed vs detected-but-benign; move the NaN check after
  NMS for a harm-based variant
- Evaluate Ranger-style activation clipping — bounding activations to their golden
  range should reclaim the huge-but-finite (silent) cases, exactly the band a NaN
  monitor misses
- Add a paired bootstrap CI on the mAP delta
- Re-run headline configurations on val2017
- Compare FP32 / FP16 / INT8
- Extend to compute faults (NVBitFI) — stored weights are the part most likely to be
  ECC-protected already

# Glossary

| term | meaning |
|---|---|
| **mAP** | mean Average Precision. Per class, area under the precision-recall curve, averaged over classes. `mAP50-95` averages again over IoU 0.5:0.05:0.95 (COCO primary); `mAP50` uses IoU 0.5 only. |
| **IoU** | Intersection over Union — overlap / union area of two boxes. Scale-invariant. Thresholds here: 0.50 = same object, 0.99 = unchanged. |
| **SDC** | Silent Data Corruption — wrong output, no indication anything went wrong. |
| **DUE** | Detected Unrecoverable Error — wrong output that announces itself (here NaN/Inf). Recoverable by re-running. |
| **object_lost / phantom / class_flip** | Object vanished / spurious object / wrong label on a matched box. |
| **SEU** | Single Event Upset — one bit flipped by a particle strike. |
| **rad-hard** | A property of *hardware* (SEU-resistant cells, ECC, TMR), not of an environment. |