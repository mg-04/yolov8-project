# Hardware weight read path

The comparator from `src/weightSentry.py`, in Verilog, validated against that
Python and synthesised. This is the step that turns "an 8-bit compare is
cheaper than two fp32 compares" from an assertion into a measurement.

Stage layout follows `ece725/digital/gcd` — one directory per flow stage,
each with `src/` for inputs and a gitignored run directory for tool output.

Chip-level context and the tapeout plan: **[RadHard Edge ML Inference](https://claude.ai/artifact/MDFHVxUWta1jqd9MKk34Kv)**.
Environment setup: **[WeightSentry Setup](https://claude.ai/artifact/PaQ535jDTSBX49g3Jr6jzm)** or
[../GETTING_STARTED.md](../GETTING_STARTED.md).

RTL is grouped by component, not by status: a stub sits beside the implemented
RTL of the same block, so nothing moves when it lands. `rtl/stub.args` is the
only status record.

```
rtl/        top.v               nine pads, the S0-S3 pipeline registers
            stub_guard.vh       included by every stub
  check/    sentry_check.v      the check: combinational, parameterised by
                                format / bound / repair
            secded.v            FP16 encoder and SECDED reference decoder
            bound_table.v       triplicated exponent bounds with majority voting
  readpath/ fault_inject.v      address- and region-gated XOR on the read word
            weight_read_check.v three-way MAC source mux over one SRAM read
  control/  spi_cfg.v           SPI slave and configuration register file
            ctrl.v              pass sequencer
            addrgen.v           STUB - loop-nest walker, weight and bound index
            counters.v          STUB - event classification and per-bit histogram
  mac/      mac_array.v         STUB - eight FP16 multiply-accumulate lanes
            actwin.v            STUB - sliding activation window
            golden.v            STUB - store the golden tile, then compare
model/      sram_22b.v          simulation model standing in for the weight macro
verif/    src/gen_vectors.py    vectors from `weightSentry.Bounds` itself
          src/tb_sentry_check.cpp   Verilator harness, compares `flag`
          src/tb_weight_read_check.sv   integrated FP16 read-path simulation
          data/*.vec            1,920 per mode: real weights, real bounds
          rtl/                  simulator build directory, gitignored
          syn/ par/             gate-level and post-PnR runs, when a simulator
                                is available
syn/      src/synth.py          all 16 variants, tabulated
          synth_results.json    tracked: these numbers are cited
          WORK/                 gitignored
par/      src/                  place and route; empty until the flow starts
pwr/      src/                  power analysis; empty
Makefile                        `make test`, `make test-read`, `make vectors`, `make synth`
```

The control plane and the whole check path are implemented. What remains is
address generation, the arithmetic, and the counters — see `make stubs`.

Two deliberate departures from gcd. Its `verif/data` is gitignored because the
golden data regenerates from the tool; these vectors need the trained
checkpoint, so tracking them is what lets someone re-verify the equivalence
claim without the model. And `syn/synth_results.json` sits outside `syn/WORK`
because the area numbers are cited rather than scratch.

Still missing for a real flow: no `.sdc` in `syn/src`, since Yosys does no STA
and there are no constraints to write yet.

## Run

```sh
make test            # all four bound modes against the checked-in vectors
make test-read       # FP16 read path, single/double ECC faults, and bound voting
make vectors         # regenerate them from the Python (needs the checkpoint)
make synth           # 16 variants -> syn/synth_results.json
make clean
```

One mode by hand, if needed:

```sh
make run-exp1
```

`make test` uses Verilator. `make test-read` uses VCS, which `source setup.sh`
puts on PATH — the testbench needs hierarchical `force`/`release`, `#delay`s and
four-state compares, none of which Verilator 4.028 has. `make synth` uses Yosys.

## FP16 SRAM interface

Use `secded_encode` to generate the write word for the SRAM. The 22-bit word
packs FP16 data in `[15:0]`, five Hamming bits in `[20:16]`, and even overall
parity in `[21]`. On read, present the SRAM word and its weight address together
with `read_valid`. The XOR mask applies to all 22 bits, and fires only when the read address equals
`inject_addr` and the site lies in the armed region, so exactly one word per pass
is corrupted; `inject_hit` reports that it fired. Both the SECDED decoder and
`sentry_check` receive that same faulted read.

`golden_weight` is the SECDED reference for clean and single-error reads.
`ecc_error_mask` identifies the corrected bit in the packed word, including
parity bits. A detected double error clears `golden_valid`. Use `weight_valid`
and `golden_valid` together when interpreting the bitwise comparison
`weight_matches_golden`. `mac_src` selects which of the three sources reaches
the MACs — bound-checked, raw, or SECDED-corrected — and the bound check keeps
reporting `sentry_flag` whichever is selected.

Pulse `rst` before loading the table. Write clean, finite FP16 exponent bounds
on a rising edge with `bound_wr_en` asserted. One entry per output channel:
`bound_addr` is supplied separately from `weight_addr` rather than derived from
it, because weights per channel is `Cin*k*k` — 144 for a 16-channel 3x3 slice —
which is not a power of two, so no shift extracts it. That also means the
grouping is a policy of `addrgen`, not a hardware commitment: per channel, per
K=128 block or per layer are the same silicon. Reads and the vote across the
three register banks are combinational. An unprogrammed or out-of-range entry
clears `bound_valid` and blocks a defended read.
`TWO_SIDED=0` stores a 5-bit ceiling per copy, and `TWO_SIDED=1` stores a 5-bit
floor and ceiling. Each copy also stores a valid bit per entry.

## Chip top

`rtl/top.v` is real at the boundary and hollow inside: nine signal pads, the
S0-S3 pipeline registers and the whole control plane are final, while the
datapath blocks are stubs. Integrate one real block at a time and keep
`make lint-top` passing — a change that breaks it does not merge.

```sh
make lint-top    # must always pass
make test-top    # SPI config, bound write, status readback, sequencer
make stubs       # what is left to implement
```

Three file lists drive the flow. `rtl/rtl.args` is implemented RTL,
`rtl/stub.args` is what remains, and `model/model.args` holds simulation models
standing in for vendor macros. Every file deleted from `stub.args` is a block
delivered.

A stub drives `x` and, when clocked out of reset, kills the simulation rather
than returning a plausible value. `make test-top` passes `+define+STUB_WARN` to
downgrade that to a warning so the sequencer can be seen to complete; drop the
define once `stub.args` is empty.

### Pipeline

| stage | does |
|---|---|
| S0 | address generate — weight address and bound index |
| S1 | SRAM read; the bound *index* is registered here, the lookup is in S2 |
| S2 | address-gated inject, then SECDED and the bound check in parallel |
| S3 | MAC, accumulate, compare against the golden tile |

The injector compares the address of the word *in flight*, so the address is
pipelined alongside the data. Feeding it the live S0 address would corrupt a
neighbouring weight and the counters could not tell the difference.

The bound lookup, the majority vote and the compare are all combinational, so
S2 is the critical path and the one that decides whether the checker closes
timing at II=1.

`drain` is a single pulse on the cycle the pipeline goes empty, not a level
across the flush, so `golden` stores or compares one tile exactly once.
`PIPE_DEPTH` on `ctrl` counts S1, S2, S3 and the accumulator register; raise it
if a stage is added. Asserting late costs idle cycles and nothing else,
asserting early captures a partial accumulation, so err high.

`VDD_chk`/`VSS_chk` are a separate supply domain over `sentry_check` and the
bound table, so energy per check is measurable. They are pads, not RTL ports.

### SPI register map

24-bit frame, MSB first: one R/W bit, seven address bits, sixteen data bits.
Reads shift the addressed register out on MISO during the same frame. Sampled
in the core clock domain, so hold the core in reset while loading; only
`start`/`done` cross domains.

| addr | reg | fields |
|---|---|---|
| 0x00 | CTRL | `[0]` start (pulse), `[1]` clear counters |
| 0x01 | MODE | `[1:0]` mac_src — 0 sentry, 1 raw, 2 ECC; `[2]` ecc_correct_en |
| 0x02 | INJ_CTRL | `[0]` inject_enable, `[1]` inject_act_region |
| 0x03 | INJ_ADDR | `[11:0]` armed site |
| 0x04 | INJ_MASK_L | `[15:0]` mask bits 15:0 |
| 0x05 | INJ_MASK_H | `[5:0]` mask bits 21:16 |
| 0x06 | BND_ADDR | bound table write address |
| 0x07 | BND_DATA | `[4:0]` emax, `[12:8]` emin — the write pulses `bound_wr_en` |
| 0x08 | N_WEIGHTS | weights in one pass |
| 0x09 | N_OC | output channels in the slice |
| 0x0a | CNT_SEL | which counter CNT_LO/HI returns |
| 0x0b | MEM_ADDR | memory pointer; auto-increments after each word |
| 0x0c | MEM_DAT_L | `[15:0]` word bits 15:0 |
| 0x0d | MEM_DAT_H | `[5:0]` word bits 21:16 — the write pulses `mem_wr_en` |
| 0x40 | STATUS (r) | `[0]` busy, `[1]` done (sticky), `[2]` always 1, `[3]` bound_disagreement |
| 0x41 | CNT_LO (r) | selected counter, low half |
| 0x42 | CNT_HI (r) | selected counter, high half |

`mac_src` has three positions because the ECC baseline has to be *runnable*,
not merely observable: all three configurations then execute over identical
cycles, so switching one changes the result and nothing else.

`done` is sticky so the host can poll STATUS for it; it clears on the next
start.

Addresses below `0x40` are write registers, `0x40` and above are read
registers. Writes to CTRL, BND_DATA and MEM_DAT_H fire one-cycle pulses; every
other register holds until rewritten.

Loading the memory is one MEM_ADDR then two frames per word, since the pointer
auto-increments: 2,304 weight words is ~4,600 frames, about 5 ms at 100 MHz,
once per slice. The SRAM is single-ported, so load while the core is idle — the
same constraint the bound table already has. There is **no readback path yet**;
adding one is what a raw-scrub mode for multi-bit-upset statistics needs.

Golden is stored once per campaign, not per injection site: the tile depends
only on the weights and activations, so one pass with `inject_enable` clear
establishes it and every armed pass afterward compares against it.

### Area as blocks land

Generic gates, Yosys `synth -flatten`, at the top's defaults
(`NUM_BOUNDS=16`, FP16, `REPAIR=0`):

| block | cells |
|---|---|
| `weight_read_check` (whole read path) | 1,496 |
| &nbsp;&nbsp;of which `bound_table`, TMR, DEPTH=16 | 1,115 |
| `spi_cfg` | 433 |
| `ctrl` | 349 |
| `fault_inject` | 124 |
| `sentry_check` | 28 |

The read path was 5,095 cells at the old `NUM_BOUNDS=72`. Sizing the bound
table to the slice depth rather than the layer depth accounts for the 3.4x
drop, and swapping TMR for parity plus reload is worth roughly another 3x.

The bound table is still 75% of the read path, so it is where the remaining
area is. `DEPTH=8` measures 440 cells against 1,115 at `DEPTH=16`, and
`cout % 8 == 0` holds for 99.99% of yolov8n's MACs, so halving the slice depth
costs no coverage.

### Open decisions

Four of the five stubs are blocked on these rather than on anyone's time. Only
`counters` can start today, and it is also the highest-value one: seven outputs
of `weight_read_check` — the whole SECDED bit-level oracle — are computed and
discarded until it exists.

| decision | options | what it blocks |
|---|---|---|
| loop order | `ic`-outermost (444 B of state) or `oc`-outermost (984 B) | `addrgen`, `actwin` |
| `acc` width | 192 b (one bank) or 3,072 b (16 banks, needed by `ic`-outermost) | `mac_array`, `golden` |
| `K` | hardwired 3, or configurable — 1x1 layers are 19% of yolov8n's MACs | `actwin` |
| numeric format | `mac_array`'s header says both "FP16" and "integer"; roughly a 2x swing in effort | `mac_array` |
| array unroll | 8 output pixels (today, one weight per cycle) or 8 output channels (8 weights per cycle) | SRAM width, checker count |
| SRAM width | 1 word or 8 words per row | ECC granularity below |
| ECC granularity | per 16-bit weight (37.5% overhead) or per 128-bit row (7.0%) | the published SECDED baseline |

The last three are coupled and the measurements say the 8-wide option is close
to free: SAED32 composition gives 232.1 k um^2 for 2304x22 against 215.6 k for
288x176, because 22 bits fits a 8/32/46/48/128-bit macro library badly. The
bound table does not need true multiporting either — 8 aligned channels read as
one 80-bit row measures 323 cells against 416 for today's single port, where
eight independent read ports would be 2,025.

What it does change is the baseline: a designer reading 8 weights per cycle
would put one SECDED over the row, dropping ECC overhead from 37.5% to 7.0%
and the headline ratio from 82x to roughly 15x. Row ECC is cheaper and weaker —
it corrects one bit per 128 rather than one per 16 — and which way that trade
goes under real upsets is exactly what the chip is built to measure. It needs
stating before the ratio goes in a paper.

Two register addresses are worth reserving before the map freezes: a weights-
per-output-channel stride so `addrgen` needs no divider, and `PIPE_DEPTH` if
post-silicon adjustment is wanted.

## Parameters

`MODE` selects the bound, matching the Python exactly:

| MODE | bound | metadata |
|---|---|---|
| 0 | `e <= emax` | 1 B/unit |
| 1 | `e <= emax && (e >= emin \|\| x == 0)` | 2 B/unit |
| 2 | `lo <= w <= hi` | 8 B/unit |
| 3 | mode 2 `&& (e >= emin \|\| x == 0)` | 9 B/unit |

`REPAIR` selects what happens on a failure — `0` substitutes exponent zero,
`1` is Algorithm 1 as published (toggle exponent bits MSB-first, take the first
candidate in range). Synthesising both prices the search.

`EXP_W`/`MANT_W` are `8`/`23` for FP32 and `5`/`10` for FP16.

Float comparison uses the sign-magnitude to monotonic-unsigned mapping, so
modes 2 and 3 need a full-width integer comparator rather than an FPU.

## Equivalence

**7,680 vectors, 0 mismatches** — 1,920 per mode, each a real yolov8n weight
with its real per-unit bound, clean and then faulted at every exponent bit.

Vectors come from `weightSentry.Bounds` itself rather than from a
reimplementation of the rule, so the RTL is checked against the exact code that
produced every number in this project. `in_range()` is evaluated on the whole
tensor and then indexed: its bound tables are expanded to tensor shape, so
passing a single element would broadcast and judge every site against element
0's bound.

Only `flag` is compared. Detection is what the measurements rest on; the repair
path's no-candidate fallback is deliberately simplified in the RTL (it
substitutes `emax` in every mode, where Python clamps to `lo`/`hi` in the
signed modes).

## Area

Generic gates, Yosys `synth -flatten`, detection only (`REPAIR=0`):

| bound | metadata | FP32 | FP16 |
|---|---|---|---|
| exp 1-sided | 1 B/unit | **46** | **28** |
| exp 2-sided | 2 B/unit | 116 | 67 |
| signed | 8 B/unit | 421 | 209 |
| signed + floor | 9 B/unit | 491 | 248 |

**Exponent against signed: 9.2x fewer gates in FP32, 7.5x in FP16.**

Cost of Algorithm 1's candidate search, which the software result showed is
redundant — the first candidate always wins:

| | detection | + search | |
|---|---|---|---|
| FP32 exp 1-sided | 46 | 457 | **+411** |
| FP32 signed | 421 | 1013 | +592 |
| FP16 exp 1-sided | 28 | 189 | **+161** |
| FP16 signed | 209 | 537 | +328 |

The search is **85-90% of the block** and provably does nothing. Dropping it is
the largest single area saving available, and it follows from a software
finding rather than from any hardware work.

## What these numbers are not

**Generic gate counts, not LUTs and not um^2.** They order the variants
correctly and the ratios are sound; absolute area needs Vivado or a
standard-cell flow.

**Yosys' Xilinx LUT mapping is not monotonic at this scale.** Mapped to xc7 it
reports `signed` (161 LUT) as *larger* than `signed + floor` (102 LUT), which
does strictly more work, while the generic gate count orders them correctly
(421 against 491). `synth.py` records both; trust the gates. Publishable LUT
figures need Vivado.

**No timing.** Yosys does no STA, so there is no Fmax here. Whether the
comparator closes timing inside an II=1 pipeline — the question that decides
the throughput overhead — needs a real place-and-route.
