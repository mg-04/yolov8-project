# Equivalence check: the Verilog comparator against the Python that produced
# every number in this project. `make test` runs all four bound modes.
#
# Stage layout follows ece725/digital/gcd: rtl/ verif/ syn/ par/ pwr/, each
# stage with src/ for inputs and a gitignored run directory for tool output.
# verif/rtl is the RTL-netlist run dir; verif/syn and verif/par are where the
# same vectors run against the synthesised and placed netlists once a
# gate-level simulator is available.
#
# Verilator 4.028 has no --build, so the generated makefile is invoked
# directly, and --exe needs an absolute path to the harness.

VERILATOR ?= verilator
IVERILOG  ?= iverilog
VVP       ?= vvp
RTL       := rtl/sentry_check.v
TB        := verif/src/tb_sentry_check.cpp
DATA      := verif/data
BUILD     := verif/rtl
READ_RTL  := $(RTL) rtl/secded.v rtl/bound_table.v rtl/fault_inject.v rtl/weight_read_check.v
READ_TB   := verif/src/tb_weight_read_check.sv

MODES     := exp1 exp2 sgn sgnf
MODE_exp1 := 0
MODE_exp2 := 1
MODE_sgn  := 2
MODE_sgnf := 3

# 8/23 is FP32, 5/10 is FP16. The checked-in vectors are FP32.
EXP_W  ?= 8
MANT_W ?= 23
REPAIR ?= 0

.PHONY: test test-read vectors synth clean
.PRECIOUS: $(BUILD)/%/sim

test: $(addprefix run-,$(MODES))

test-read: $(BUILD)/weight_read_check.vvp
	$(VVP) $<

$(BUILD)/weight_read_check.vvp: $(READ_RTL) $(READ_TB)
	@mkdir -p $(BUILD)
	$(IVERILOG) -g2012 -s tb_weight_read_check -o $@ $(READ_RTL) $(READ_TB)

run-%: $(BUILD)/%/sim
	@printf '%-6s ' $*
	@./$< $(DATA)/$*.vec

$(BUILD)/%/sim: $(RTL) $(TB)
	@mkdir -p $(BUILD)/$*
	@$(VERILATOR) --cc $(RTL) --exe $(abspath $(TB)) --top-module sentry_check \
	  -GMODE=$(MODE_$*) -GEXP_W=$(EXP_W) -GMANT_W=$(MANT_W) -GREPAIR=$(REPAIR) \
	  --Mdir $(BUILD)/$* -o sim -Wno-fatal
	@$(MAKE) -s -C $(BUILD)/$* -f Vsentry_check.mk sim

vectors:
	./verif/src/gen_vectors.py

synth:
	./syn/src/synth.py --json syn/synth_results.json

clean:
	rm -rf $(BUILD) verif/syn verif/par syn/WORK/* par/src/WORK pwr/WORK
