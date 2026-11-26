# Getting started

Setup for this project on CMU ECE Linux hosts (`ece-linux`, RHEL 8 or 9). Two
independent halves: a **Python** side that runs the model and the fault-injection
campaigns, and an **RTL/C++** side that builds and simulates the Verilog. You
can work on either without the other.

This guide is also a page: **[WeightSentry Setup](https://claude.ai/artifact/PaQ535jDTSBX49g3Jr6jzm)**. For what the project
is and where it is going, see **[RadHard Edge ML Inference](https://claude.ai/artifact/MDFHVxUWta1jqd9MKk34Kv)**.

## Quick start

```bash
git clone git@github.com:mg-04/yolov8-project.git
cd yolov8-project

./setup.sh --install        # one time: create .venv, install Python deps (~10 min)
source setup.sh             # every new shell: set PATH, PYTHONPATH, licenses
```

Then verify both halves:

```bash
python -c "import weightSentry, torch; print(torch.cuda.is_available())"

cd hw
make test        # Verilator: 1920 vectors x 4 bound modes vs the Python model
make test-read   # VCS: read path, SECDED, TMR vote, fault injection
```

Expected:

```
exp1      1920 vectors  0 mismatches  OK
exp2      1920 vectors  0 mismatches  OK
sgn       1920 vectors  0 mismatches  OK
sgnf      1920 vectors  0 mismatches  OK
PASS: read path, TMR bound vote, address-gated injection, 88 single flips, 924 double flips
```

`source setup.sh --check` reports what was found without changing anything. Use
it first when something breaks.

## Part 1 — Python

| | |
|---|---|
| interpreter | system `python3.11` (**not** `python3`, which is 3.6.8 and too old for torch) |
| venv | `.venv/` in the repo root, gitignored |
| deps | `requirements.txt`, pinned to the versions the results were produced with |
| import path | `src/` is added to `PYTHONPATH` — `weightSentry`, `fi_lib` and `shield` are imported by bare name, not as a package |

Entry points:

```
src/weightSentry.py    bound construction and the in_range check (the reference model)
src/campaign.py        weight fault-injection campaigns
src/campaign_act.py    activation fault-injection campaigns
src/fi_lib.py          injection primitives
src/shield.py          the defence under test
bench_harness.py       accuracy/mAP evaluation
```

Model checkpoints (`*.pt`, 639 MB) and datasets are gitignored. Ultralytics
downloads checkpoints on first use. For COCO val2017, run `./get_coco_val.sh`
(it fetches the 5,000 validation images only, not the 19 GB training set).

### If you need a different Python

`setup.sh` takes the first of `python3.11`, `python3.12` it finds. To override
the venv location, set `YOLOV8_VENV`. CPU-only machines should drop the two
`+cu121` pins from `requirements.txt` and install plain `torch torchvision`.

## Part 2 — RTL and C++

| tool | where it comes from | needed for |
|---|---|---|
| **Verilator** 4.028 | `/usr/bin`, already installed | `make test` |
| **VCS** T-2022.06 / Y-2026.03 | `source setup.sh` runs the ece240 setup script | `make test-read` |
| **Yosys** | a VTR build; `setup.sh` falls back to a shared AFS copy | `make synth` |
| **g++** | `/usr/bin`, already installed | Verilator compiles its C++ output |
| **SAED32 EDK** | `$SAED32_EDK`, set by `setup.sh` | gate-level sim and power, later |

Stage layout under `hw/`, following `ece725/digital/gcd` — one directory per
flow stage, each with `src/` for inputs and a gitignored run directory:

```
hw/rtl/      Verilog sources
hw/verif/    src/ testbenches, data/ vectors, rtl/ build output (gitignored)
hw/syn/      synthesis
hw/par/      place and route
hw/pwr/      power
```

Make targets:

```bash
cd hw
make test        # Verilator equivalence suite against the Python
make test-read   # VCS read-path testbench
make vectors     # regenerate verif/data/*.vec from the Python model (needs Part 1)
make synth       # Yosys gate counts -> syn/synth_results.json
make clean
```

### Why two simulators

They are not interchangeable. Verilator is a **cycle-based** Verilog-to-C++
compiler: fast, but 2-state, it ignores `#delay`s, and it has no
`force`/`release`. That is fine for `sentry_check`, which is combinational and
driven from a C++ harness over 7,680 vectors.

`tb_weight_read_check.sv` needs an **event-driven** simulator: it has clocked
config writes, `#1` settling between stimulus and check, 4-state `!==`
comparisons, and a hierarchical `force` into one TMR bank to prove the majority
vote masks a stored upset. Verilator 4.028 cannot do any of those — it would
silently drop the delays and appear to pass. Icarus Verilog could do most of it
but not the `force`, and VCS is a superset of Icarus, so Icarus is not installed.

## Gotchas

These all cost real time to find.

**`/tmp` is 4 GB.** pip unpacks wheels there, and the CUDA wheels overflow it as
a confusing `No space left on device` while `/home` has terabytes free.
`setup.sh` redirects `TMPDIR` and `PIP_CACHE_DIR` to `/home/$USER/.cache`.

**`pypi.nvidia.com` truncates large wheels.** `nvidia-nccl-cu12` stalls at
162 of 189 MB every time, and pip resolves it from there by default. `setup.sh`
passes `--index-url https://pypi.org/simple` to make PyPI primary, which serves
the identical wheel without trouble.

**Don't hardcode a VCS version.** RHEL 8 hosts get `T-2022.06` and RHEL 9 hosts
get `Y-2026.03`; `/afs/ece.cmu.edu/class/ece240/bin/setup_vcs` picks the right
one and sets `LM_LICENSE_FILE`. `setup.sh` sources it.

**AFS tokens expire.** Symptoms are permission errors on `/afs/...` paths or VCS
failing to find its license. Run `aklog && aklog ece.cmu.edu` (aliased to `ki`).

**AFS home has a 100 GB quota.** `df` reports the whole AFS cell and is
meaningless here; use `fs listquota ~`.

**Yosys is not packaged** on these hosts. `setup.sh` falls back to a copy under
another user's AFS tools directory, which works only while that path stays
readable. If `make synth` fails, build your own from
[YosysHQ/yosys](https://github.com/YosysHQ/yosys) and put it on `PATH` first.

**`yolov8-env/` is dead.** It is a leftover Python 3.6 venv with no torch, kept
only because an old shell alias points at it. Use `.venv`.

## Known verification gaps

Worth knowing before you trust a number:

1. **The equivalence vectors are FP32; the chip is FP16.** `weight_read_check.v`
   hardwires `EXP_W=5, MANT_W=10`, but `make test` runs `EXP_W=8, MANT_W=23`.
   `WS.Bounds` hardcodes the FP32 exponent field, so FP16 equivalence is blocked
   on the Python side.
2. **Only the flag is compared, never the repaired value.** `tb_sentry_check.cpp`
   checks `flag` and ignores `w_out`, so `REPAIR=1` has no equivalence proof —
   4,045 of the existing vectors would exercise it.
3. **The TMR vote is one-ninth covered.** The testbench forces `bank_b` only,
   and forces the whole word to zero rather than flipping one bit.
