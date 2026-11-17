# Equivalence check: the Verilog comparator against the Python that produced
# every number in this project. One command per bound mode, or `make test`
# for all four.
#
# Verilator 4.028 has no --build, so the generated makefile is invoked
# directly, and --exe needs an absolute path to the harness.

VERILATOR ?= verilator
RTL       := rtl/sentry_check.v
TB        := sim/tb_sentry_check.cpp
BUILD     := sim/build

MODES     := exp1 exp2 sgn sgnf
MODE_exp1 := 0
MODE_exp2 := 1
MODE_sgn  := 2
MODE_sgnf := 3

# 8/23 is FP32, 5/10 is FP16. The checked-in vectors are FP32.
EXP_W  ?= 8
MANT_W ?= 23
REPAIR ?= 0

.PHONY: test vectors synth clean
.PRECIOUS: $(BUILD)/%/sim

test: $(addprefix run-,$(MODES))

run-%: $(BUILD)/%/sim
	@printf '%-6s ' $*
	@./$< sim/vectors/$*.vec

$(BUILD)/%/sim: $(RTL) $(TB)
	@mkdir -p $(BUILD)/$*
	@$(VERILATOR) --cc $(RTL) --exe $(abspath $(TB)) --top-module sentry_check \
	  -GMODE=$(MODE_$*) -GEXP_W=$(EXP_W) -GMANT_W=$(MANT_W) -GREPAIR=$(REPAIR) \
	  --Mdir $(BUILD)/$* -o sim -Wno-fatal
	@$(MAKE) -s -C $(BUILD)/$* -f Vsentry_check.mk sim

vectors:
	./sim/gen_vectors.py

synth:
	./syn/synth.py --json syn/synth_results.json

clean:
	rm -rf $(BUILD)
