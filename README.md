# Hardware Fault Injection in YOLOv8

Measuring how single-bit hardware errors affect object detection, and building the
cheapest defence that covers the dominant failure mode. The software campaigns are
complete; current work is the RTL for a radiation-hardened edge-ML inference test
chip.

**Target** YOLOv8n, 3,151,904 parameters after BatchNorm fusion, 64 `nn.Conv2d`
layers, 100,860,928 bits. Conv weights (the injection targets) are 3,146,176.

| golden baseline | images | mAP50-95 |
|---|---|---|
| `coco128` | 128 | 0.4451 |
| `val2017` | 5,000 | **0.3684** |

Sweeps use coco128 for speed (~1.5 s per evaluation vs 26 s); 0.3684 is the number
to report.

## Pages

| page | what it is |
|---|---|
| [RadHard Edge ML Inference](https://claude.ai/artifact/MDFHVxUWta1jqd9MKk34Kv) | the project overview: goals, the chip, progress, where we need people, related work |
| [WeightSentry Setup](https://claude.ai/artifact/PaQ535jDTSBX49g3Jr6jzm) | this repo's setup guide as a page, same content as GETTING_STARTED.md |

## Setup

See **[GETTING_STARTED.md](GETTING_STARTED.md)**, or the page above. Short version:

```bash
./setup.sh --install     # once: create .venv, install pinned deps
source setup.sh          # every new shell: PATH, PYTHONPATH, tool licences
```

## Layout

```
src/                 injection harness   fi_lib.py  campaign.py  campaign_act.py  shield.py
hw/                  RTL, verification, synthesis    <- current work
analysis/            main campaign       summarize.py  plot_layers.py  analyze.py
analysis/blast/      blast radius: how much of the output one fault changes
analysis/geometry/   box geometry and SDC event identity
analysis/multibit/   accumulated faults, magnitude stratification, cross-model census
analysis/partition/  splitting the model toward FPGA fabric
demos/               superseded one-off scripts from the exploratory phase
results/             campaign CSVs and logs  (gitignored)
figures/             plots  (tracked)
```

Each `analysis/` subdirectory has its own README with the method, the reproduction
commands and the full tables for that sweep. Harness details are in
**[IMPLEMENTATION.md](IMPLEMENTATION.md)**; the hardware flow is in
**[hw/README.md](hw/README.md)**.

```bash
./src/campaign.py 16 30                 # weight campaign at bit 30
./src/campaign_act.py persistent 16 30  # activation campaign
./analysis/summarize.py                 # all cross-campaign tables
cd hw && make test && make test-read    # RTL equivalence and read path
```

---

# Current work: the weight read path in silicon

The defence is a one-sided exponent bound per output channel: a weight is rejected
when its exponent exceeds the largest exponent that channel legitimately holds. The
chip exists to measure that check against SECDED under real upsets, including
multi-bit upsets, which simulation cannot produce.

## Architecture

```
S0  address generate          weight address, bound address
S1  SRAM read                 22-bit word: FP16[15:0] + Hamming[20:16] + parity[21]
S2  inject + check            address-gated XOR, then SECDED decode || bound check
S3  MAC                       8x FP16, accumulate
```

Both checkers see the same faulted read. SECDED is the oracle, not a competitor:
its syndrome says what actually happened, so every bound-check verdict can be
scored as caught / missed / false / clean. The injector is gated on address *and*
region, because weights and activations share one SRAM macro and a mistyped
address must not score as a weight fault.

| block | cells (FP16, Yosys generic) |
|---|---|
| `sentry_check` — the check itself | **28** |
| `secded_encode` | 36 |
| `secded_decode` | 118 |
| `fault_inject` (address + region gated) | 124 |
| `bound_table` (DEPTH=72, TMR) | **4,765** |
| `weight_read_check` total | 5,095 |

## Results

**Storage is the headline, not gates.** Per-channel exponent bounds against SECDED
check bits, for the same coverage of the dominant failure mode:

| 64x64x3x3 layer (36,864 weights, 64 channels) | metadata |
|---|---|
| SECDED, 6 check bits per word | 27.0 kB |
| exponent 1-sided bound, 1 B per channel | **64 B** |

**432x less metadata**, and 864x on the largest layer (128x256x3x3). Gate count is
the weaker argument: nobody tapes out to save 180 gates.

**Bound choice, detection only (`REPAIR=0`):**

| bound | metadata | FP32 | FP16 |
|---|---|---|---|
| exp 1-sided | 1 B/unit | 46 | **28** |
| exp 2-sided | 2 B/unit | 116 | 67 |
| signed | 8 B/unit | 421 | 209 |
| signed + floor | 9 B/unit | 491 | 248 |

Exponent against signed: **9.2x fewer gates in FP32, 7.5x in FP16.**

**Algorithm 1's candidate search is 85-90% of the block and provably does nothing**
— the software result showed the first candidate always wins. FP16 1-sided goes
28 -> 189 gates to add it. Dropping it is the largest single area saving available,
and it follows from a software finding rather than any hardware work.

**The bound table dominates everything else.** At `DEPTH=72` with TMR it is 4,765
cells — 93% of the read path, and 170x the 28-gate comparator it feeds. `DEPTH=72`
matches no yolov8n `Cout` (16/32/64/80/128/256); it is inherited from an old
figure. Sizing it to the *slice* depth and reloading per slice, with parity instead
of TMR, gives roughly 370 cells, about **13x smaller**, at no cost — the weight SRAM
is already reloaded 128 times for the largest layer.

| DEPTH | single bank | TMR x3 |
|---|---|---|
| 16 | 338 | 1,115 |
| 72 (current) | 1,568 | 4,765 |
| 256 | — | 17,162 |

## Verification

| suite | tool | coverage |
|---|---|---|
| `make test` | Verilator | **7,680 vectors, 0 mismatches** — 1,920 per bound mode, real weights with real per-unit bounds, against `weightSentry.Bounds` itself |
| `make test-read` | VCS | read path, TMR bound vote under a forced bank, address and region gating, **all 22 single-bit and all 924 double-bit flip patterns** against SECDED |

Known gaps, in order: the equivalence vectors are FP32 while the chip is FP16
(`WS.Bounds` hardcodes the FP32 exponent field); only `flag` is compared, never the
repaired value, so `REPAIR=1` has no equivalence proof; and the TMR vote is
one-ninth covered.

---

# Established by the software campaigns

18,432 injections across three campaigns, six bit positions and 64 layers, plus
focused magnitude, accumulation, geometry and cross-model sweeps. The full tables
live with the code that produces them. Condensed:

1. **Risk concentrates in one bit position.** Sampling uniformly over the parameter
   store, `p = (1/32)(95.49%) + (31/32)(0.21%) = 3.18%`. **Bit 30 — the exponent
   MSB — is 93% of random-fault risk.** An injection containing a bit-30 flip
   corrupts ~95% of images; one without corrupts 0.2%.

2. **The mechanism is FP32 magnitude duality.** A flip grows or shrinks a weight
   depending on whether the bit is already set. Almost every weight has `|w| < 2`,
   so bit 30 is clear and flipping it scales by 2^128. The 453 weights with
   `|w| >= 2` (0.0144%) invert this and are damageable at nine positions instead of
   one, confirmed by magnitude-stratified sampling at 85x. Those weights never
   produce NaN — **the most vulnerable weights are also the most silent.**

3. **Corruption is overwhelmingly silent.** DUE = 0.00% at 9 of 10 bit positions.
   Only bit 30 produces NaN/Inf, in 6.65% of cases. At 16 accumulated faults the
   model is wrong on 41% of images and announces it 6.5% of the time.

4. **Accumulated faults compose independently.** `P(corrupt|N) = 1-(1-0.0320)^N`
   fits across a 32x span, lift 1.00 ± 0.06. **N50 = 21 faults.** Single-fault
   characterisation fully predicts multi-fault behaviour.

5. **Damage is bimodal, not graded.** Benign injections move boxes 0.02 px median;
   SDC injections 16.23 px. A monitor watching box stability sees nothing until an
   object is simply gone.

6. **A range check suffices for random faults.** A bit-30 flip on a normal weight
   yields `|w| ~ 1e36` against a largest legitimate conv weight of 17.494. One
   comparison per weight removes 93% of risk with zero false positives — which is
   what the chip implements. Adversarial faults are a different threat model and
   need partitioning; see [analysis/partition/](analysis/partition/).

7. **What does not work:** NaN/Inf detection (blind to 9 of 10 bits, and 42% of the
   DUE it does raise had correct detections anyway); confidence thresholding (77% of
   phantom detections sit at confidence exactly 1.000); box-geometry monitoring
   (bimodal, per 5).

8. **Larger models are ~20x less reliable.** Bit 30 is exactly size-invariant while
   memory footprint grows linearly, so scale dilutes only the ~7% of risk that was
   never the problem. A census across yolov8n/x, yolo11x, yolov9e, yolov10x and
   rtdetr-l finds every CNN detector carries the same signature — the first conv
   holds 35-38% of its weights at `|w| >= 2`, making it a **BatchNorm-fusion
   artifact, not a YOLOv8 quirk.** RT-DETR breaks the pattern: no small dense layer
   to shield, so partitioning does not transfer to transformer detectors.

| where | what it answers |
|---|---|
| [analysis/](analysis/) | per-bit and per-layer sensitivity, all three campaigns |
| [analysis/blast/](analysis/blast/) | weight faults are global (~100% of anchors), activation faults local (median 3.6%) |
| [analysis/geometry/](analysis/geometry/) | how far boxes move, which objects are lost or invented |
| [analysis/multibit/](analysis/multibit/) | accumulation, magnitude stratification, cross-model census |
| [analysis/partition/](analysis/partition/) | the trust boundary and the ladder toward FPGA fabric |

## Limitations

- The 16-layer sweep omits `model.22.cv3.{0,1,2}.1` and `cv2.0.1`, which hold 208 of
  the 453 high-magnitude weights. Their behaviour at bits 29/28/27 is inferred from
  the mechanism, not measured.
- Per-layer rates rest on 16 weight samples per (layer, bit), roughly ±12 pp. Images
  within one injection are near-perfectly correlated (ICC ~ 1.00), so effective
  sample size is the injection count.
- Single model, single dataset. The magnitude-duality mechanism is a property of
  IEEE-754 and should generalise; the specific layer rankings will not.
- Out of scope throughout: compute-unit faults (needs NVBitFI), control-path and
  addressing faults, and input-image faults. Multi-bit upsets are out of scope *in
  simulation* — measuring them is what the chip is for.
- Synthesis numbers are generic gates, not LUTs or µm². The ratios are sound;
  absolute area and any Fmax need a standard-cell flow and place-and-route.
