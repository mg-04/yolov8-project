# Hardware weight read path

The comparator from `src/weightSentry.py`, in Verilog, validated against that
Python and synthesised. This is the step that turns "an 8-bit compare is
cheaper than two fp32 compares" from an assertion into a measurement.

Stage layout follows `ece725/digital/gcd` — one directory per flow stage,
each with `src/` for inputs and a gitignored run directory for tool output.

```
rtl/      sentry_check.v        the check: combinational, parameterised by
                                format / bound / repair
          secded.v              FP16 encoder and SECDED reference decoder
          bound_table.v         triplicated exponent bounds with majority voting
          fault_inject.v        enabled XOR on the SRAM read word
          weight_read_check.v   connects both checkers to a shared SRAM read
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

The chip-level integration still needs the SRAM macro, the configuration and
scan interface, the MAC array, the activation LFSR, and the event counters.

Two deliberate departures from gcd. Its `verif/data` is gitignored because the
golden data regenerates from the tool; these vectors need the trained
checkpoint, so tracking them is what lets someone re-verify the equivalence
claim without the model. And `syn/synth_results.json` sits outside `syn/WORK`
because the area numbers are cited rather than scratch.

Still missing for a real flow: no `.sdc` in `syn/src` (Yosys does no STA, so
there are no constraints to write yet) and no `.args` file lists, which
Xcelium would need in `verif/src` the way gcd has `rtl.args`.

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

`make test` uses Verilator. `make test-read` uses Icarus Verilog (`iverilog`
and `vvp`). `make synth` uses Yosys.

## FP16 SRAM interface

Use `secded_encode` to generate the write word for the SRAM. The 22-bit word
packs FP16 data in `[15:0]`, five Hamming bits in `[20:16]`, and even overall
parity in `[21]`. On read, present the SRAM word and its weight address together
with `read_valid`. The enabled XOR mask applies to all 22 bits, and both the
SECDED decoder and `sentry_check` receive that same faulted read.

`golden_weight` is the SECDED reference for clean and single-error reads.
`ecc_error_mask` identifies the corrected bit in the packed word, including
parity bits. A detected double error clears `golden_valid`. Use `weight_valid`
and `golden_valid` together when interpreting the bitwise comparison
`weight_matches_golden`. `bypass_sentry` sends raw FP16 data to `weight_out`
while the checker continues reporting `sentry_flag` on configured reads.

Pulse `rst` before loading the table. Write clean, finite FP16 exponent bounds
on a rising edge with `bound_wr_en` asserted. The default address groups 128
consecutive weights per entry, with 72 entries for 9,216 weights. Reads and the
vote across the three register banks are combinational. An unprogrammed or
out-of-range entry clears `bound_valid` and blocks a defended read.
`TWO_SIDED=0` stores a 5-bit ceiling per copy, and `TWO_SIDED=1` stores a 5-bit
floor and ceiling. Each copy also stores a valid bit per entry.

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
