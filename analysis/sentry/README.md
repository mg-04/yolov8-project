# Range-check defences: WeightSentry, and what replaces it

A published defence (WeightSentry, Abumandour et al., HASP 2025) reimplemented against
this harness, measured on YOLOv8n, and then modified where the measurement showed it
does not do what the paper claims.

**Read [../../README.md](../../README.md) first** for the fault model, sampling design
and metric definitions. This file assumes them.

## Scripts

| script | what it does |
|---|---|
| `sentry_sweep.py` | defended-vs-undefended injection campaign; same sites for every configuration |
| `coverage_census.py` | exhaustive single-flip detection rate, stratified by magnitude; no inference, no sampling |
| `batch_overhead.py` | wall-clock cost of the check vs batch size, and the repair path |
| [`../../src/weightSentry.py`](../../src/weightSentry.py) | the defence itself: bound tables, Algorithm 1, forward-pre hooks |

```bash
./analysis/sentry/sentry_sweep.py 48 24 weight 31 30 29 28 27 26 25 24 23
./analysis/sentry/sentry_sweep.py 64 24 bias   31 30 29 28 27 26 25 24 23
./analysis/sentry/coverage_census.py
./analysis/sentry/batch_overhead.py 1,2,4,8,16,32,64 50
```

Arguments are `samples_per_bit`, `n_images`, `kind`, then bit positions.

## What the defence is

Two phases, and no training data at any point.

**Pre-deployment**, compute a bound over each *weight unit* — a group of weights that
will share one bound — directly from the checkpoint. That table is the only metadata.

**At inference**, check every weight against its unit's bound before use. A trained
model never contains out-of-range values, so anything outside is a fault. Repair by
toggling exponent bits MSB-first and accepting the first candidate that lands back in
range; failing that negate; failing that clamp (the paper's Algorithm 1).

The check sits on the weight *fetch* path, which is why `attach()` uses a
`forward_pre_hook`. Checking after the conv would be activation range restriction, a
different defence with worse properties: weight bounds are exact and have provably zero
false positives, while activation bounds must be profiled and misfire on
out-of-distribution inputs.

## Why the paper's CNN path never fires here

WeightSentry uses per-output-channel bounds **only if the kernel is larger than 3x3**,
and groups smaller kernels into fixed units of K weights. YOLOv8n:

| kernel | layers | weights | share | WS path |
|---|---|---|---|---|
| 1x1 | 25 | 761,104 | 24.2% | K-grouped |
| 3x3 | 39 | 2,385,072 | 75.8% | K-grouped |

No kernel exceeds 3x3, so every weight falls to K-grouping and the per-channel
mechanism the paper leads with is never exercised. At the paper's default K=128 that is
**24,581 units x 8 B = 192 KB** for YOLOv8n — 7% of the 2.7 MB model, against the
48-228 KB of SRAM per SM the paper budgets for.

That figure also does not reconcile with the paper's Table 3, which reports 2.64 KB for
ResNet-18. At ~11.2M parameters and K=128 that should be ~87,500 units, ~700 KB — 265x
apart. The LLM rows *do* reconcile (Llama-3.2-3B: 1.98 KB / 8 B = 247 tensors, matching
per-tensor bounds), so the discrepancy is confined to the CNN column. Resolve it before
citing those numbers.

## Granularity

`mode` selects how a layer is carved. A group never spans layers.

- `chan` — one bound per output channel (one filter). `K` is ignored.
- `k` — one bound per K consecutive weights in flat memory order.
- `tensor` — one bound for the whole layer.

Per-channel is not uniformly finer or coarser than K=128; it depends on filter size:

```
model.0.conv        (16,3,3,3)      chan =    27 w/group   K=128 =   108
model.2.cv1.conv    (32,32,1,1)     chan =    32 w/group   K=128 =   128
model.22.cv3.2.0    (80,256,3,3)    chan = 2,304 w/group   K=128 =   128
```

The useful way to compare them is **exponent slack** — the gap between a typical weight
and its group's ceiling, which is the room a fault has before the bound notices. Each
bit adds a known number of exponent steps when flipped (bit 23 → +1, 24 → +2, 25 → +4,
26 → +8, 27 → +16, 28 → +32, 29 → +64, 30 → +128):

| mode | units | mean slack | median | amplifying flips caught | sign flips |
|---|---|---|---|---|---|
| tensor | 64 | 5.41 | 5 | 39.72% | 0.01% |
| k/K=1024 | 3,081 | 3.92 | 4 | 52.76% | 0.30% |
| **chan** | **5,729** | **3.73** | **3** | **54.87%** | **0.50%** |
| k/K=128 | 24,581 | 3.10 | 3 | 61.36% | 2.45% |
| k/K=16 | 196,636 | 2.14 | 2 | 71.30% | 19.53% |

Median slack 3 at `chan` predicts exactly the measured per-bit coverage — bit 23 (+1)
and bit 24 (+2) sit inside the slack, bit 25 (+4) clears it:

| bit | amplification | global | chan | K=128 | K=16 |
|---|---|---|---|---|---|
| 30 | 3.4e38 | 100.00% | 100.00% | 100.00% | 100.00% |
| 29 | 1.8e19 | 97.42% | 97.42% | 97.42% | 97.42% |
| 27 | 65,536 | 20.26% | 76.48% | 84.93% | 93.02% |
| 26 | 256 | 0.10% | 69.85% | 81.65% | 91.67% |
| 25 | 16 | 0.02% | 53.44% | 71.20% | 89.56% |
| 24 | 4 | 0.00% | 12.30% | 17.07% | 33.35% |
| 23 | 2 | 0.00% | 1.59% | 4.18% | 15.43% |

Weighted by measured low-magnitude SDC rates, the whole granularity axis is nearly flat:

| scheme | storage | risk retired |
|---|---|---|
| global constant | 0 B | 91.24% |
| **per-channel** | **5.59 KB** | **94.61%** |
| K=128 | 192 KB | 95.55% |
| K=16 | 1.5 MB | 96.87% |

34,000x more storage buys 5.6 points. `chan` is the knee, and the extra coverage finer
granularity buys lands at bits 24 and 23, which amplify by only 4x and 2x.

## The global threshold is not global

A single hard-wired `exp < 132` false-positives on yolov8m and larger. Max exponent over
**all** fused parameters, weights and biases both:

| model | max \|w\| weight | max \|w\| bias | threshold |
|---|---|---|---|
| yolov8n / yolov8s | 17.494 / 26.039 | 23.999 / 25.344 | **132** |
| yolov8m | 58.808 | 21.982 | **133** |
| yolov8l / x / yolo11x | 77.665 / 84.248 / 91.720 | 15.060 / 12.977 / 13.967 | **134** |
| rtdetr-l | 10.656 | 25.902 | **132** |

So Tier 1 is one byte in a config register written at model load, not zero storage.
Note also that max weight grows with model size (17 → 92) while max **bias shrinks**
(24 → 13).

## Algorithm 1 does not restore values

Measured over single flips, with a pristine copy held aside to classify each repair:

| bit | WS signed K=128: detected | **exact** | exp 2-sided chan: detected | **exact** |
|---|---|---|---|---|
| 30 | 142/150 | 142 | 150/150 | 150 |
| 29 | 7/150 | **0** | 150/150 | **150** |
| 27 | 9/150 | **0** | 148/150 | **148** |
| 25 | 96/150 | **0** | 74/150 | 10 |
| 23 | 13/150 | **0** | 1/150 | 0 |

WeightSentry achieves **0% exact repair at every bit except 30**, at any granularity:

```
original   corrupted    repaired        unit bound
 -2.054   -3.79e+19   -1.114e-19   [-8.735, 8.204]
  6.907    1.274e+20    3.744e-19   [-8.735, 8.204]
```

Because `[min, max]` brackets zero, clearing the exponent MSB *always* lands in range.
Being the first candidate tried, it is always the one accepted. The paper's 100%
accuracy retention follows from crushing weights to ~1e-19 being benign, not from
recovery.

**The fix is one more byte per unit.** Bound magnitude from both sides, so the weight's
exponent must land inside its unit's observed span. The crush-to-zero candidate is
rejected and the loop continues to the bit that actually flipped. The bound becomes an
annulus rather than an interval, which is the right shape: the fault moves magnitude in
either direction, and an interval containing zero is blind to one of them.

`two_sided=True`. The other deviation is `signed=False`, which keeps only the unit's
maximum exponent field — one byte, an 8-bit unsigned compare instead of two FP32
compares. It cannot see sign flips, which costs 0.50% of them at `chan`.

## Measured: defended vs undefended, weight faults

48 injections per bit, 9 bits, 24 images, identical sites across configurations
([`results/sentry_n48_img24.csv`](../../results/sentry_n48_img24.csv)). Storage figures
are weight-side only:

| config | metadata | SDC | rate | retired |
|---|---|---|---|---|
| undefended | 0 | 1056 / 10,368 | 10.185% | — |
| WS signed per-chan | 44.76 KB | 8 | 0.077% | 99.2% |
| WS signed K=128 | 192.04 KB | 6 | 0.058% | 99.4% |
| **exp 2-sided per-chan** | **11.19 KB** | **14** | **0.135%** | **98.7%** |

Undefended SDC by bit: **bit 30 carries 1017 of 1056 events** (plus all 6 DUE), bit 25
carries 30, bit 31 carries 5, bits 26 and 24 carry 3 and 1. Every configuration takes
bit 30 to exactly zero with 48/48 exact repairs.

Three results worth keeping, two of which cut against the section above:

**The two-sided fix bought no measurable damage reduction.** It wins decisively on
repair quality at bits 29/28/27, but those bits produce **0.00% SDC undefended** —
flipping bit 29 on a typical weight clears an already-set bit, shrinking the value to
~1e-22, which is a pruning event. The floor catches faults that were never going to
hurt. Repair quality is real; its value here is not.

**Crush-to-zero is an effective mitigation, not merely a defect.** At bit 25 every
repair in every configuration was false, and SDC still fell from 2.60% to 0.00-0.61%.
Zeroing a corrupted weight removes the damage whether or not the right value came back.

**Exponent-only is genuinely looser than signed, exactly where the residual lives.** At
bit 25: 21 flagged vs 27, 0.61% SDC vs 0.09%. The exponent ceiling admits everything up
to 2^(emax+1) while the signed bound stops at the actual maximum — up to 2x slack.

Caveat on all of it: sites were drawn uniformly over 3.15M weights, so the 453
high-magnitude weights (0.014%) were essentially never hit. **This sweep says nothing
about the |w| >= 2 population.** Testing it needs magnitude-stratified sampling, as in
[`../multibit/accum_sweep.py`](../multibit/) with `--mag high`.

## Biases

WeightSentry covers weights only — biases appear nowhere in the paper. That matters
here because of where the high-magnitude population actually lives:

```
weights  3,146,176 x4B = 12.00 MB   median 0.0345   |x|>=2:  0.014%
biases       5,728 x4B = 22.38 KB   median 0.8645   |x|>=2: 17.947%
```

Biases are **25x larger at the median** — the BN-fusion term `B - g*mu/sqrt(s^2+e)` —
and nearly one in five exceeds 2. The split holds across architectures:

| model | params | \|w\|>=2 | share | in weights | in biases | % of biases |
|---|---|---|---|---|---|---|
| yolov8n | 3.15M | 1,481 | 0.0470% | 453 | **1,028** | 17.9% |
| yolov8s | 11.2M | 2,215 | 0.0199% | 687 | 1,528 | 14.6% |
| yolov8m | 25.9M | 2,543 | 0.0098% | 816 | 1,727 | 10.2% |
| yolov8x | 68.2M | 2,899 | 0.0043% | 888 | 2,011 | 6.8% |
| yolo11x | 56.9M | 2,498 | 0.0044% | 993 | 1,505 | 3.2% |
| rtdetr-l | 32.1M | 5,755 | 0.0179% | 1,560 | **4,195** | 6.2% |

Biases are 59-73% of the population everywhere. Note that the population is roughly
**constant in absolute terms** (~1.5k-3k) while parameters grow 22x, so its share
collapses with scale — 0.0470% to 0.0043%. Under a random-fault model the
high-magnitude population contributes on the order of **0.1% of total risk**; the 85x
is a per-value vulnerability multiplier, not a risk contribution, and should not be
presented as one. It earns its place through adversarial selection (an attacker picks
the weight, so population size is irrelevant), through setting the detector's threshold,
and as a correctness claim against Hong 2019's `[3.05e-5, 2]` assumption.

**The bias region is cheap to protect wholesale.** One bias per output channel means
`chan` grouping puts each bias in its own unit, so the bound degenerates to a copy of
that value's own exponent. Two consequences: it detects a flip in *either* direction
with no floor needed (`emin == emax`, so `two_sided` is free on biases), and it pins the
repair to exactly one candidate, removing the false-repair mode entirely.

| region | signed bound | exponent bound |
|---|---|---|
| weights, per-channel | 44.76 KB | 11.19 KB |
| biases, per-channel | 22.38 KB | 5.59 KB |

For a degenerate unit the signed bound stops being a range: `lo == hi`, so you store the
value once (4 B) and test equality. That is plain duplication of the bias array, and
`_per_unit()` accounts for it rather than charging 8 B for two identical copies.

### Measured: bias faults behave nothing like weight faults

64 injections per bit, 9 bits, 24 images
([`results/sentry_bias_n64_img24.csv`](../../results/sentry_bias_n64_img24.csv)).
Undefended SDC, bias faults against weight faults at the same bit:

| bit | 31 | 30 | 29 | 28 | 27 | 26 | 25 | 24 | 23 |
|---|---|---|---|---|---|---|---|---|---|
| **bias** | 24.15% | 18.68% | 17.84% | 14.97% | 18.49% | 21.42% | 14.97% | 19.66% | 13.22% |
| **weight** | 0.43% | 88.28% | 0.00% | 0.00% | 0.00% | 0.26% | 2.60% | 0.09% | 0.00% |

**The "93% of risk sits in one bit" result is weight-only.** Bias faults are roughly
uniformly damaging across every exponent bit *and* the sign bit — 13-24% everywhere,
18.16% overall. The reason is magnitude and position: a bias is 25x larger at the median
and enters as a constant offset on every output pixel of its channel, with no
attenuation by an activation (measured median |a| = 0.2742) and no dilution across the
Cin*k^2 accumulation terms. A bit-23 flip doubles a ~0.86 bias and shifts the entire
feature map by that amount; the same flip on a 0.0345 weight is lost in the sum.

Bit 30 also behaves differently in *kind*: 528 of 1536 bias faults were **DUE** (34.4%),
against 6 DUE in the entire weight campaign. A bias at ~1e38 overflows its channel to
Inf and announces itself; a weight at ~1e38 is scaled by activations into a large finite
number and stays silent.

| config | bias metadata | SDC | rate | exact repairs |
|---|---|---|---|---|
| undefended | — | 2,510 / 13,824 | 18.157% | — |
| **WS signed per-chan** | **22.38 KB** | **0** | **0.000%** | **64/64 at every bit** |
| exp 2-sided per-chan | 5.59 KB | 371 | 2.684% | 64/64 except bit 31 |

**This is the one place the exponent-only bound fails outright.** It is blind to sign,
and sign flips are the *largest* single bias failure mode at 24.15% — it flags nothing
at bit 31 and leaves that rate untouched. The signed bound catches it because at
degenerate granularity `lo == hi` makes any change detectable, sign included.

So the region wants duplication, not range checking. Once each unit holds one value the
range machinery has degenerated into a comparison against a stored copy, at which point
storing the full 4 B and getting sign and mantissa coverage free is strictly better than
storing 1 exponent byte and losing a quarter of the failure modes.

**Cheaper still, and not yet implemented:** SECDED over the full 32-bit bias word needs
6 parity bits, **5,728 x 6 bits = 4.20 KB** — below even the exponent bound, and it
*corrects* single faults in any bit including sign and mantissa without assuming the
metadata itself is trustworthy. That is the recommended mechanism for this region.

Keep the denominator honest when quoting any of this. These are per-bias-fault rates.
Biases are 0.182% of parameters, so their contribution to total random-fault risk is
`0.182% x 18.16% = 0.033%`, against `99.818% x 10.185% = 10.167%` for weights — about
**0.32% of total risk**. Biases are far more dangerous per event and far less likely to
be hit; protection is worth doing because 4.20 KB is cheap, not because the region
dominates.

## Cost: the check is per-forward, so it amortizes over the batch

`batch_overhead.py`, RTX 4070, yolov8n fused, forward pass only (no NMS), 50 timed
iterations after 10 warmup. The check runs once per conv per forward call over a weight
tensor whose size does not depend on how many images are in flight, so its **absolute**
cost should be flat in batch while the baseline grows. It is:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| undefended ms/fwd | 2.28 | 3.26 | 5.40 | 10.75 | 22.65 | 53.68 | 115.11 |
| WS signed per-chan **+ms** | 5.20 | 5.00 | 4.61 | 4.39 | 4.39 | 4.28 | 4.87 |
| WS signed per-chan overhead | 228.2% | 153.7% | 85.3% | 40.9% | 19.4% | 8.0% | **4.2%** |
| WS signed K=128 overhead | 224.1% | 151.6% | 84.9% | 41.5% | 19.1% | 8.0% | 4.4% |
| exp 1-sided per-chan overhead | 354.9% | 234.1% | 133.2% | 59.1% | 26.8% | 11.2% | 5.3% |
| exp 2-sided per-chan overhead | 514.1% | 334.6% | 183.1% | 81.8% | 36.3% | 14.8% | 6.9% |

The added milliseconds are constant to within ±10% across a **64x** span in batch. That
is the whole mechanism behind the paper's numbers: 1.1% at batch 64 and 2.8% at batch 16
are not a property of the defence, they are a property of the denominator. **Batch 1 --
the latency-critical single-camera case -- pays 54x more, and it is the only regime a
reactive edge deployment can use**, because batching *k* frames from one camera adds
(k-1) frame intervals of latency. Multi-camera deployments batch across cameras instead
and get the amortization for free; that, not frame batching, is the realistic way an
edge box reaches batch 8.

Note also that undefended throughput is best at **batch 4-8** (1.34-1.35 ms/img) and gets
*worse* by batch 64 (1.80 ms/img) on 12 GB. Defended throughput peaks at batch 16
(1.69 ms/img). So batch 64 is not even the right operating point here -- quoting a
batch-64 overhead describes a configuration you would not deploy.

### Two reasons these are an upper bound, and what they tell us

`Bounds` stores `lo`/`hi`/`emax` **expanded to full tensor shape** so the check
vectorises, so a signed check streams 36 MB (w + lo + hi) where a deployed one streams
12 MB of weights against a 67 KB table. And `in_range()` is several eager elementwise
ops, each its own CUDA kernel, adding ~5 launches x 64 convs per forward. Both costs are
batch-independent, so neither distorts the *shape* of the curve, but both inflate its
magnitude.

Which one dominates is testable, and the answer **refutes the memory-traffic
explanation**: `exp 1-sided` reads *less* than `WS signed` (24 MB vs 36 MB) but is
**53% slower** (8.09 ms vs 5.20 ms at batch 1). Exponent-only costs more instructions and
more kernels (shift, mask, compare, isfinite vs two float compares) and that is what the
clock follows. On this card the overhead is launch- and instruction-bound, not
bandwidth-bound. Granularity is also free at runtime -- K=128 times identically to
per-chan (224.1% vs 228.2%) despite 3.2x the metadata -- so **granularity costs storage,
not time**.

### Repair cost

One *detectable* bit-30 flip in `model.7.conv` (9.4% of the model), batch 1:

| config | clean | 1 fault | delta |
|---|---|---|---|
| WS signed per-chan | 7.51 ms | 7.93 ms | +5.7% |
| WS signed K=128 | 7.45 ms | 7.92 ms | +6.2% |
| exp 1-sided per-chan | 10.50 ms | 11.13 ms | +6.0% |
| exp 2-sided per-chan | 13.89 ms | 14.68 ms | +5.7% |

The site must be low-magnitude or nothing is being timed -- see the census below. This
~6% is a *simulation artifact worth understanding*: the vectorised loop runs all eight
candidate exponents over the **whole tensor** regardless of how many elements are bad, so
one fault costs the same as a million. Scalar or hardware repair touches only the
faulting element, so the real figure is ~8 extra operations on 1 weight out of 3.1M.

### Datapath cost -- ANALYTICAL, nothing synthesized yet

No HLS or RTL exists, so there are no measured cycle counts. What the structure implies:

| | signed | exp 1-sided | exp 2-sided |
|---|---|---|---|
| compare logic per weight lane | 2 x fp32 compare | 1 x uint8 compare | 2 x uint8 compare |
| added throughput cost | 0 cycles/weight | 0 | 0 |
| added pipeline latency | 1 stage | 1 stage | 1 stage |
| bound storage | 67.13 KB | 11.19 KB | 16.78 KB |

A comparator is combinational, so in a pipelined weight-fetch path the check sustains one
weight per cycle and costs one stage of latency, paid once per layer rather than per
weight. The repair loop becomes a priority encoder over all eight candidate exponents
evaluated in parallel -- 1 cycle, not 8 -- and only on the faulting lane. This is the
sense in which the GPU's batch-dependence is an artifact of the platform: a
weight-stationary array amortizes the check over every activation streamed through the
loaded weights, which a single 640x640 image already supplies thousands of.

**Confirming any of this requires synthesis.** The exponent-only variant exists precisely
because an 8-bit unsigned compare is the cheap thing to put in that datapath, and that
claim is currently an argument, not a number.

## Exhaustive detection census, stratified by magnitude

`coverage_census.py` enumerates **every** parameter and asks only "would the check flag
this flip?" -- no inference, no sampling, exact counts over all 3,146,176 weights and
5,728 biases. It isolates detection from the network's own fault tolerance, which the SDC
sweeps necessarily conflate.

The stratification is the finding. A bit-30 flip does **opposite** things to the two
magnitude populations:

```
|x| < 2    exponent MSB is 0 -> flip SETS it   -> value explodes to ~1e19
|x| >= 2   exponent MSB is 1 -> flip CLEARS it -> value collapses to ~1e-40
```

A signed `[min,max]` bound brackets zero, so it catches the explosion and is
**structurally blind to the collapse**. Weight detection rate at bit 30, by stratum:

| config | `\|w\| < 2` | `\|w\| >= 2` |
|---|---|---|
| WS signed per-chan | 100.00% | **0.00%** |
| WS signed K=128 | 100.00% | **0.00%** |
| exp 1-sided per-chan | 100.00% | **0.00%** |
| exp 2-sided per-chan | 100.00% | **99.78%** |

Not approximately zero -- **exactly** zero, all 453 high-magnitude weights, both signed
granularities. This closes the open `--mag high` question in the gaps list below, and it
is the same interval-vs-annulus defect documented in `weightSentry.py`, seen from the
fault side rather than the repair side.

It is also why the earlier SDC sweep showed the two-sided floor buying nothing: the faults
it newly catches are the ones that *collapse* a weight toward zero, and collapsing one
weight is benign. The floor's detection gain at bit 30 is real and its damage reduction is
nil, and both statements belong in the same sentence.

Where the floor does change the picture is bits 29-27, which signed misses almost
entirely across the whole weight population:

| bit | WS signed per-chan | WS signed K=128 | exp 2-sided per-chan |
|---|---|---|---|
| 29 | 0.01% | 0.01% | **100.00%** |
| 28 | 0.01% | 0.01% | **100.00%** |
| 27 | 0.06% | 0.06% | **98.78%** |
| 26 | 11.70% | 12.82% | 25.65% |
| 25 | 54.28% | 65.18% | 41.60% |

Those bits cause 0% SDC undefended on *low*-magnitude weights, which is why the sweep
could not see the difference -- but Conclusion 2 in the top-level README has the
high-magnitude population dying at bits 29-27 by a factor of 85x. Signed catches 100% of
high-magnitude flips at 29-27; the exposure is confined to the collapse direction.

Coverage is **non-monotonic** in bit position (exp 2-sided: 98.78% at bit 27, 25.65% at
26, 41.60% at 25) because detection depends on whether the flip moves the exponent *out*
of the unit's observed span, and the direction of the move depends on whether that bit is
currently set -- which varies with the weight's own exponent.

### Why every bias row reads 100%

Under `chan` grouping there is one bias per output channel, so each bias is its own unit
and the bound **degenerates**: signed `lo == hi == the value`, which is exact duplication,
not a range. Any flip whatsoever falls outside `[v, v]`, so signed bias detection is
100.00% at every bit including the sign bit. Exponent-only degenerates to `emin == emax`,
pinning the exponent exactly, so `exp 2-sided` is also 100% at bits 30-23 -- and **0.00%
at bit 31**, because it cannot see sign at all. That single gap is the entire 2.68%
residual SDC in the bias sweep, and it is why the recommendation for biases is duplication
or SECDED rather than a range.

`exp 1-sided` on biases reads 82.05% at bit 30 for the same arithmetic reason: 1,028 of
5,728 biases are high-magnitude (17.95%), their bit-30 flips collapse, and a ceiling-only
bound cannot see a collapse. 100% - 17.95% = 82.05%, exactly.

## What is and is not new here

The per-channel bound is **WeightSentry's, published**. Not ours to claim. What survives
as distinct: the random-fault characterisation and the magnitude duality; bias coverage;
activations; the exponent-only integer comparator as against two FP32 compares; and the
demonstration that Algorithm 1 is zero-substitution rather than restoration.

The threat models differ and both are correct. The paper's S-BFA sets a loss threshold
specifically to *avoid* exponent-MSB flips, defending against an adversary who dodges
the cheap check. The 93%-in-one-bit result here is about random faults, which do not
dodge.

## Known gaps

- `campaign.py` still filters to `layer.weight` and has never injected a bias, so every
  per-layer and per-bit table outside this folder remains weight-only. `sentry_sweep.py
  ... bias` is currently the only bias injection path in the repo.
- SECDED on the bias word (4.20 KB, corrects sign and mantissa too) is recommended above
  but not implemented; `bias_mode` only offers range/duplication bounds.
- The magnitude-stratified question is now answered for **detection** by
  `coverage_census.py` (signed bounds are exactly 0% on `|w| >= 2` at bit 30), but not for
  **damage**: no injection sweep has been run restricted to the 453 high-magnitude weights,
  so the SDC consequence of that blind spot is still inferred rather than measured.
- No HLS or RTL exists, so the datapath cycle and area figures above are analytical. The
  claim that an 8-bit compare is cheaper than two FP32 compares is unsynthesized.
- Activation-side range checking is unimplemented. Conv activations are 15,131,200
  values at 640x640 batch 1, 4.8x the weight check and scaling with resolution and
  batch, against a bound table of the same 5,729 units.
- Repair under *multi-bit* faults is untested; `correct()` assumes a single flip and
  will accept a wrong candidate when several bits move.
