# Hardware Fault Injection in YOLOv8

Measuring how single-bit hardware errors affect object detection quality. Injects
into **weights** (memory faults) and **activations** (buffer / compute faults), and
separates **silent** corruption from **detectable** corruption.

Target: YOLOv8n, 3,151,904 parameters after BatchNorm fusion, 64 `nn.Conv2d`
layers, 100,860,928 bits. Conv weights — the injection targets — are 3,146,176.

Python 3.11.9 (pyenv) · Ultralytics 8.4.156 · PyTorch 2.5.1+cu121 · RTX 4070

| golden baseline | images | held out? | mAP50-95 |
|---|---|---|---|
| `coco128` | 128 | **no** — all 128 are in COCO train2017 | 0.4451 |
| `val2017` | 5,000 | yes | **0.3684** |

Sweeps use coco128 for speed (~1.5 s per evaluation vs 26 s); 0.3684 is the number
to report.

Harness details — scripts, correctness guarantees, bugs found, and how to reproduce
the runs — are in **[IMPLEMENTATION.md](IMPLEMENTATION.md)**.

---

# Method

## Fault model

Single-bit flips in FP32 values, three injection sites:

| site | models | injection | gap between model and reality |
|---|---|---|---|
| **weights** | bit rot in weight memory (no ECC) | write to `layer.weight.data` | **none** — weights are FP32 in memory, so a flipped tensor bit *is* a flipped memory bit |
| **activations, persistent** | stuck-at fault in an activation buffer | forward hook, every pass, all batch elements | corruption persists and hits every image, which is right. But a fixed tensor index is assumed to map to a fixed physical cell — plausible under PyTorch's caching allocator at fixed shapes, but not established |
| **activations, transient** | single-event upset during one inference | forward hook, fires once | one inference is corrupted, which is right. But a real SASS fault corrupts one *thread's* output register, and which tensor element that is depends on the kernel's thread mapping; here the element is chosen directly |

Out of scope:

- compute-unit faults (needs NVBitFI)
- control-path and addressing faults
- multi-bit upsets from a single particle
- input-image / frame-buffer faults 

ECC on weight memory would correct
exactly the weight faults modeled here, which argues for pushing toward the compute
path next.

## Bit position sensitivity

FP layout  is `[31] sign | [30:23] exponent | [22:0] mantissa`. Damage is governed by
**magnitude, not bit position alone**, because a flip's direction depends on whether
the bit is already set:

- `|v| < 2` → exponent < 128 → bit 30 **clear**, so flipping *sets* it: ×2¹²⁸,
  catastrophic. Bits 23–29 are mostly already `1`, so flipping them shrinks the value.
- `|v| ≥ 2` → bit 30 **set**, so flipping it shrinks; but bit 29 is now clear (all weights are less than 2^65), and
  flipping that multiplies by 2⁶⁴ — catastrophic.
- `|v| ∈ [1,2)` → exponent exactly 127 → setting bit 30 gives 255, the **NaN
  encoding**. NaN appears directly, with no overflow step.

| population | n | mean \|v\| | max \|v\| | \|v\| ≥ 2 | bit 30 clear |
|---|---|---|---|---|---|
| all conv weights | 3,146,176 | 0.062 | 17.49 | 0.01% | 99.95% |
| `model.0.conv` | 432 | — | 17.49 | **43.3%** | 56.7% |
| activations @ `model.15.cv2.conv` | 2.2M | 0.162 | 7.60 | 4.7% | 95.3% |

`model.0.conv` is a magnitude outlier — BatchNorm fusion scales first-layer weights
up, so bit 29 is catastrophic there and essentially nowhere else among weights.
Activations are more exposed at bit 29 than weights (4.7% vs 0.01% at `|v| ≥ 2`).

Bits swept: **31, 30, 29, 24, 23, 22** — sign, exponent MSB, the
magnitude-dependent bit, two mid-exponent bits, mantissa MSB. Mantissa bits below
22 change a value by < 0.1%.

## Injection mechanism

```python
# weights — mutate the parameter tensor
flat = layer.weight.data.view(-1)
orig = flat[idx].clone()                              # .clone() is essential
flat[idx] = (flat[idx].view(torch.int32) ^ (1 << bit)).view(torch.float32)
...
flat[idx] = orig                                      # restore

# activations — intercept with a forward hook
def hook(mod, inp, out):
    n, c, h, w = out.shape
    y, x = int(fy * h), int(fx * w)
    out[:, chan % c, y, x] = flip(out[:, chan % c, y, x], bit)   # ALL batch elements
    return out
```

`Tensor.view(dtype)` is a **reinterpret cast**. XOR with
`1 << bit` toggles one position.

Injection happens **after `model.fuse()`**
- `model.fuse()` *folds* each BatchNorm layer into the convolution before it
- Compare both results after normalization, more tracable

Activation targets are `(channel, y_frac, x_frac)` rather than flat indices.

## Sampling design
Inject one bit corruption each time:

Each run tests:
- 64 conv layers × 
- 16 randomly chosen targets (`SEED = 0`) x
- Six possible bit positions × 
- Three campaigns (weight, persistent activation, transient activation)

= **18,432 injections**.

The seed is fixed and the RNG constructed after it, so every bit tested hits the
identical target set — differences are attributable to bit position alone.

Sampling is **uniform per layer, not per parameter**: `model.22.dfl.conv` (16
params) gets the same 16 injections as `model.7.conv` (294,912). 
- That gives every
layer equal statistical power, but means a whole-model rate cannot be read off these
numbers without reweighting by parameter count.

## Metrics

Two are recorded per injection, because **they disagree, and the disagreement is a
finding**.

**1. mAP50-95** — COCO mean Average Precision over IoU standards of `0.5:0.05:0.95`.
- "mAP-critical" means the bit flip causes `mAP < 0.5 × golden`.
- It **systematically under-reports**: losing one object out of 50 barely moves a dataset
average.

**2. Error class** (per image) — golden vs corrupted detections

| error class | condition | detectable at runtime? |
|---|---|---|
| **DUE** | Any entry in output tensor contains NaN or Inf | **yes** — check and re-run |
| **SDC** | Clean output, but objects have lost/phantom/class flip | **no** |
| **benign** | all objects match, correct classes, boxes drifted (IoU < 0.99) | n/a |
| **masked** | detection sets identical (all matched at IoU ≥ 0.99) | n/a |


- DUE takes precedence whenever
NaN/Inf is present — the fault announced itself regardless of what happened to the
detections.
- Detection outcomes recorded alongside the error class: `object_lost` (object
vanished), `phantom` (spurious detection), `class_flip` (matched box, wrong label).
These are a **separate axis** — they occur under DUE as well as SDC. At transient
bit 30, 167 DUE rows had `object_lost`.


`sdc_critical` fires if **any** of the 16 subset images shows an SDC, making it
sensitive but an upward-biased rate. Per-image counts (`v_masked`, `v_benign`,
`v_sdc`, `v_due`) are in every CSV; use those for a rate. `analyze.py` reports 95%
Wilson intervals (±12 points at n=16).

# Results

18,432 injections: 3 campaigns × 6 bit positions × 1,024, on coco128.
All tables below come from `./summarize.py` and use the **per-image rate** —
the fraction of individual golden-vs-corrupted image comparisons that were SDC or
DUE. The alternative "any of 16 images" flag (`sdc_critical`) runs 4–12× higher and
is not used here.

## Per Bit Position

> SDC and DUE are properties of the fault *site*, not the layer

| bit | field | weight SDC | weight DUE | persist SDC | persist DUE | transient SDC | transient DUE |
|---|---|---|---|---|---|---|---|
| 31 | sign | 2.8% | **0.0%** | 4.8% | **0.0%** | 4.8% | **0.0%** |
| **30** | exp MSB | **83.0%** | 5.9% | 22.5% | **30.5%** | 22.0% | **31.2%** |
| 29 | exp | 2.8% | **0.0%** | 5.1% | **0.0%** | 5.6% | **0.0%** |
| 24 | exp | 2.4% | **0.0%** | 0.2% | **0.0%** | 0.0% | **0.0%** |
| 23 | exp | 1.2% | **0.0%** | 0.0% | **0.0%** | 0.1% | **0.0%** |
| 22 | mant MSB | 0.5% | **0.0%** | 0.0% | **0.0%** | 0.0% | **0.0%** |

**1. Weight faults are silent; activation faults are loud.** At bit 30, weight faults
are 83.0% SDC against 5.9% DUE. Activation faults invert it: 22.5%
SDC against 30.5% DUE.

- Mechanism: convolution weights are typically less than 1, so the corrupted weight stays large-but-finite and propagates as plausible garbage. A
corrupted activation is evaluated against many different images, and any image whose
value at that location lands in `|v| ∈ [1,2)` hits the `E=255` NaN encoding directly.


**2. DUE is exactly 0.0% at every bit except 30**. Only bit 30 can carry an
exponent to 255.  **Other bit flip
positions are 100% undetectable from the output.**

**3. Bit 30 dominates, but no bit is harmless.** Weight SDC falls from 83.0% at bit
30 to 0.5–2.8% elsewhere.

## Where failures happen

> Per-block SDC rate at bit 30 (per-image). Weight faults on top, persistent
activation faults below. Colors are assigned based on the **weight** rate: red >= 90%,
amber 45-90%, green < 45%.

```mermaid
flowchart TD
    IN["input 3x640x640"] --> B0

    subgraph BB["backbone"]
        direction TB
        B0["0 Conv 16ch<br/>W 46% · A 6%"] --> B1["1 Conv 32ch<br/>W 50% · A 5%"]
        B1 --> B2["2 C2f 32ch<br/>W 68% · A 16%"] --> B3["3 Conv 64ch<br/>W 90% · A 8%"]
        B3 --> B4["4 C2f 64ch · P3<br/>W 92% · A 29%"] --> B5["5 Conv 128ch<br/>W 100% · A 38%"]
        B5 --> B6["6 C2f 128ch · P4<br/>W 99% · A 27%"] --> B7["7 Conv 256ch<br/>W 100% · A 24%"]
        B7 --> B8["8 C2f 256ch<br/>W 99% · A 27%"] --> B9["9 SPPF · P5<br/>W 82% · A 13%"]
    end

    subgraph NK["neck (PAN-FPN)"]
        direction TB
        B9 --> U10["10 Upsample<br/>no params"]
        U10 --> C11["11 Concat 384ch"]
        C11 --> N12["12 C2f 128ch<br/>W 93% · A 26%"]
        N12 --> U13["13 Upsample"]
        U13 --> C14["14 Concat 192ch"]
        C14 --> N15["15 C2f 64ch<br/>W 86% · A 21%"]
        N15 --> D16["16 Conv 64ch<br/>W 91% · A 25%"]
        D16 --> C17["17 Concat 192ch"]
        C17 --> N18["18 C2f 128ch<br/>W 93% · A 17%"]
        N18 --> D19["19 Conv 128ch<br/>W 98% · A 23%"]
        D19 --> C20["20 Concat 384ch"]
        C20 --> N21["21 C2f 256ch<br/>W 94% · A 18%"]
    end

    B6 -. "P4 skip" .-> C11
    B4 -. "P3 skip" .-> C14
    N12 -. skip .-> C17
    B9 -. skip .-> C20

    subgraph HD["22 Detect"]
        direction TB
        CV3["cv3 · class branch<br/>W 98% · A 50%<br/>MOST SENSITIVE"]
        CV2["cv2 · box branch<br/>W 42% · A 0%"]
        DFL["dfl · 16 params<br/>W 30% · A 0%"]
        CV2 --> DFL
    end

    N15 -- "P3 s8" --> HD
    N18 -- "P4 s16" --> HD
    N21 -- "P5 s32" --> HD
    HD --> OUT["84 x 8400<br/>4 box + 80 cls"]

    classDef hi fill:#c0392b,stroke:#7b241c,color:#fff
    classDef mid fill:#d68910,stroke:#9c640c,color:#fff
    classDef lo fill:#1e8449,stroke:#145a32,color:#fff
    classDef io fill:#5d6d7e,stroke:#34495e,color:#fff
    class B3,B4,B5,B6,B7,B8,N12,N15,D16,N18,D19,N21,CV3 hi
    class B0,B1,B2,B9 mid
    class CV2,DFL lo
    class IN,OUT,U10,U13,C11,C14,C17,C20 io
```

Three things the picture makes visible.

**The `cv3` class branch is the only red node in the head**, at 98% weight / 50%
activation SDC, while `cv2` and `dfl` beside it are green. Same depth, same inputs,
opposite sensitivity — a corrupted class logit poisons NMS ranking globally, a
corrupted box coordinate only misplaces one detection.

**Blocks 0-2 are amber, not red** (46-68% weight SDC) despite being the furthest
upstream. Very early features are partly redundant: 16-32 channels at high spatial
resolution, and the network tolerates losing one. Sensitivity peaks at blocks 5-8,
where channel count has grown but spatial redundancy has not.

**Activation rates are uniformly far below weight rates** (green-ish everywhere by
comparison) because one activation element is used once, while one weight is reused
at every spatial position of its output.

## Per-section (bit 30, per-image rate, 95% Wilson CI)

| section | weight SDC | weight DUE | persist SDC | persist DUE | transient SDC |
|---|---|---|---|---|---|
| backbone | 87.6% [87, 88] | 9.5% | 22.7% | 32.0% | 23.1% |
| neck | 92.0% [91, 93] | 5.2% | 20.7% | 34.9% | 20.8% |
| head: class | **97.9%** [97, 98] | 0.8% | **50.2%** | 17.5% | **45.1%** |
| head: box | 42.4% [40, 44] | 1.9% | **0.2%** | 30.6% | **0.0%** |
| head: DFL | 29.7% [24, 36] | 0.0% | 0.0% | 28.9% | 0.0% |

**The classification head is the most sensitive section under every fault type** —
97.9% / 50.2% / 45.1%. A corrupted class logit poisons NMS ranking globally: one
absurd score suppresses every legitimate detection competing with it.

**The box branch is fault-type dependent.** 0.0–0.2% SDC against activation faults,
but 42.4% against weight faults. It is not a safe region; it is a region safe against
one of three fault classes. Its 30.6% persistent DUE with 0.2% SDC is the one
genuinely benign combination in the data: corruption is detected and harmless.

## Error Toelrance Per-layer


### Bit 30 Effect Per-Layer
> Flip bit 30, measure Per-image SDC

| layer | section | weight | act-persist | act-transient |
|---|---|---|---|---|
| **`model.22.cv3.0.2`** | head:cls | **100%** | **100%** | **100%** |
| **`model.22.cv3.1.2`** | head:cls | 98% | **100%** | **100%** |
| **`model.22.cv3.2.2`** | head:cls | 98% | 94% | 94% |
| `model.22.cv3.1.0.conv` | head:cls | 100% | 50% | 50% |
| `model.4.m.1.cv2.conv` | backbone | 97% | 39% | 56% |
| `model.5.conv` | backbone | 100% | 38% | 38% |

Least sensitive:

| layer | section | weight | act-persist | act-transient |
|---|---|---|---|---|
| `model.22.cv2.0.2` | head:box | 18% | 0% | 0% |
| `model.22.cv2.0.0.conv` | head:box | 23% | 0% | 0% |
| `model.22.dfl.conv` | head:DFL | 30% | 0% | 0% |
| `model.0.conv` | backbone | 46% | 6% | 0% |

**The top three are the three class-output convs** — `cv3.{0,1,2}.2`, the final 1×1
in each scale's classification branch, producing the 80 class logits. All three are
at or near 100% under every fault type at bit 30, and together hold ~19,200
parameters (0.6% of the model).

But see the cross-bit table below: this ranking holds **at bit 30 only**.

**No layer is safe.** Zero of 64 layers measure 0% SDC across all three campaigns.
An earlier version of this document listed eight "0% critical" layers; that was an
artifact of mixing metrics across campaigns (mAP for weights, SDC for activations).
The box-branch layers are 18–49% SDC under weight faults.

### Bit 30 Effect Per-Group

| group | weight SDC | weight DUE | persist SDC | persist DUE |
|---|---|---|---|---|
| blocks 0–4 | 77.6% | **19.5%** | 19.6% | 32.4% |
| blocks 5–9 | 97.0% | 0.3% | 25.6% | 31.6% |
| neck 10–21 | 92.0% | 5.2% | 20.7% | 34.9% |
| head 22 | 68.1% | 1.3% | 23.9% | 24.3% |

**Early blocks are ~60× louder than blocks 5–9** for weight faults (19.5% vs 0.3%
DUE) — the BatchNorm-fusion magnitude effect. Fusion inflates first-layer weights
(43.3% of `model.0.conv` exceeds 2.0 versus 0.01% model-wide), putting far more of
them in `[1,2)`, the only range that reaches NaN in a single flip.

Otherwise depth barely predicts loudness. Within a fault type, DUE is roughly flat —
what varies by layer is *how much* damage, not whether it announces itself.

### Compare Weight Bits 31, 30, 29

> `./summarize.py layerbits`. Weight campaign, per-image SDC:

| layer | section | bit 31 | bit 30 | bit 29 |
|---|---|---|---|---|
| **`model.22.dfl.conv`** | head:DFL | **37%** | 30% | **88%** |
| **`model.0.conv`** | backbone | 21% | 46% | 26% |
| `model.4.cv2.conv` | backbone | 18% | 87% | 9% |
| `model.4.cv1.conv` | backbone | 12% | 87% | 8% |
| `model.2.cv2.conv` | backbone | 12% | 87% | 7% |
| `model.5.conv` | backbone | 1% | **100%** | 1% |
| `model.7.conv` | backbone | 0% | **100%** | 0% |
| `model.8.m.0.cv1.conv` | backbone | 0% | **100%** | 0% |
| Overall | -- | 2.8% | 83.0% | 2.8%

**The layer ranking is bit-dependent, and it inverts.** `model.22.dfl.conv` is the
*most* sensitive layer at bits 29, 24 and 31 — 88% at bit 29 — and the *least*
sensitive at bit 30 (30%). The layers that dominate at bit 30 (`model.5.conv`,
`model.7.conv`, 100%) are at 0-1% everywhere else.

- The cause is the magnitude duality from the Bit position section. `model.22.dfl.conv`
is a **fixed projection holding the integers 0..15** (bit30=1, bit29=0)

`model.0.conv` behaves the same way via BatchNorm fusion (43.3% of its weights exceed
2.0 against 0.01% model-wide).

In conclusion,
- Small-weight layers are vulnerable at bit 30
- Large-weight layers are vulnerable at bits 29/24/31.

Note the events are *concentrated*, not thin: bit 29 has 460 SDC events over 30 of 64
layers, 224 of them in `model.22.dfl.conv` alone. That is why a 2.8% overall rate
still resolves a per-layer ranking.



## Additional Observations


### Transient outcome by flip direction (bit 30)

| direction | n | outcome |
|---|---|---|
| nan | 133 | 67 object_lost, 66 masked |
| grew | 262 | 58 lost, 41 phantom, 16 class_flip, 147 masked |
| shrank | 117 | 24 phantom, 93 masked — **never loses an object** |

`nan` arises when `|a| ∈ [1,2)`. `grew` is the only direction producing class flips.
`shrank` means bit 30 was already set, so the flip zeroes the activation —
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


