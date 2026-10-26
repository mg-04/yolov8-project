# Accumulated faults and magnitude stratification

Two tools that sample the parameter store differently from the per-layer campaigns,
which is what lets them answer questions the per-layer sweeps structurally cannot.

**Read [../../README.md](../../README.md) first.** Conclusions 1, 2, 4 and 8 there are
produced by these scripts; this file documents how to run them and why the sampling
differs.

## Scripts

| script | what it does |
|---|---|
| `accum_sweep.py` | places N faults at random positions and random bit positions, then tests the result against the independence null |
| `magnitude_census.py` | counts `\|w\| >= 2` across architectures. Static — no injection, seconds per model |
| `status.sh` | progress of a long sweep |

## Why the sampling differs

`accum_sweep.py` makes two choices that separate it from `geom_sweep.py` and the main
campaigns, and both are deliberate.

**Parameter-weighted, not per-layer.** A random bit in memory lands in a layer in
proportion to that layer's size, so `model.7.conv` takes 9.4% of draws and
`model.22.dfl.conv` takes 0.0005%. The stratified per-layer sampler would give those
two tiny high-magnitude layers equal weight and wildly inflate the measured damage.
Use this script for whole-model rates; use the per-layer campaigns for rankings.

**Uniform bits over [0, 31], mantissa included.** Most mantissa bits are harmless,
which is exactly why they belong in the draw. This is what makes the single-fault rate
~3% rather than the ~13% you get from sweeping only the six interesting bits.

## The independence null

The null hypothesis is that faults do not interact:

```
P(SDC | N) = 1 - (1 - p)^N
```

So read the output as a **deviation from that curve**, not as a damage level.
Sub-additive means the network masks accumulated faults; super-additive means they
compound. Measured: lift 1.00 ± 0.06 with no trend across a 32× span in N
(14,464 injections) — they compose independently, so single-fault characterisation
fully predicts multi-fault behaviour.

```bash
./analysis/multibit/accum_sweep.py 1024 256 --faults 1,2
./analysis/multibit/accum_sweep.py 1024 256 --faults 4,6,8,12,16,24,32
```

Full table in [../../figures/accumulation_results.txt](../../figures/accumulation_results.txt).

## Stratifying the parameter store

Four filters narrow the target pool, and they compose. Each appends to the output tag
so runs do not overwrite each other.

| flag | pool |
|---|---|
| `--bits B [B...]` | restrict which bit positions are drawn |
| `--mag high\|low` | split by `\|w\| >= 2` vs `\|w\| < 2` |
| `--kind bias\|weight` | split by parameter role |
| `--layer PREFIX` | one layer, or any dotted prefix |
| `--model NAME` | a different checkpoint (downloads on first use) |

Pool sizes for yolov8n, so you know whether a cell has the samples to say anything:

| | `\|w\| >= 2` | `\|w\| < 2` |
|---|---|---|
| `.bias` | 1,028 | 4,700 |
| `.weight` | 453 | 3,145,723 |

**Biases hold 69.4% of the high-magnitude population while being 0.18% of the store** —
roughly 1,200× denser in targets than weights. Any analysis restricted to `.weight`
undercounts the target set by 3.3×.

### Isolating the magnitude mechanism

```bash
for MAG in high low; do
  for B in 30 29 28 27 26 25; do
    ./analysis/multibit/accum_sweep.py 256 64 --faults 1 --bits $B --mag $MAG
  done
done
```

This is the run behind Conclusion 2: the two populations swap which bits kill them, by
a factor of 85× at bits 29–27. See the table there.

### Does width dilute a fault?

```bash
for M in yolov8n yolov8x; do
  for L in model.0.conv model.7.conv; do
    for B in 30 29 25; do
      ./analysis/multibit/accum_sweep.py 256 64 --model $M --faults 1 --bits $B --layer $L
    done
  done
done
```

Short answer: catastrophic upsets (bit 30) do not dilute at all; moderate ones
(bit 25) dilute strongly. Numbers and the caveat on the bit-25 point estimate are in
Conclusion 8.

## Cross-architecture census

```bash
./analysis/multibit/magnitude_census.py yolov8n yolov8x yolo11x yolov9e yolov10x rtdetr-l
```

Reports params, tensors, the `|w| >= 2` count and share, how many layers carry them,
`max |w|` and the top-1 layer's share. Cheap enough to run before committing injection
time to a new architecture. **Counts `.weight` only** — see the pool table above.

`max |w|` is the number a range-check threshold has to clear, and it is
model-specific: 17.49 for yolov8n up to 91.72 for yolo11x.

## Open runs

Gaps that would tighten the current claims:

```bash
# bias vs weight residual, which gates the residual-risk analysis
for K in bias weight; do for B in 26 25 24 31; do
  ./analysis/multibit/accum_sweep.py 256 64 --faults 1 --bits $B --kind $K
done; done

# the bit-25 dilution figure rests on 1 event in 256 injections
./analysis/multibit/accum_sweep.py 2048 64 --model yolov8x --faults 1 --bits 25 --layer model.7.conv

# magnitude-stratified at scale
./analysis/multibit/accum_sweep.py 256 64 --model yolov8x --faults 1 --bits 30 --mag high
```
