# Makefile
# See https://docs.cocotb.org/en/stable/quickstart.html for more info
# Makefile (pytest-driven)

# defaults
SIM ?= icarus
FST ?= -fst
TOPLEVEL_LANG ?= verilog
GATES ?= no

SRC_DIR := $(PWD)/../src
PROJECT_SOURCES := fc_layer.sv neuron.sv

# junit output
JUNIT ?= results.xml
TEST_FILE ?= test.py

# Pytest config
export COCOTB_LOG_LEVEL=INFO
PYTEST ?= pytest
PYTEST_ARGS ?= -s -q

# -------- RTL vs Gate-level selection (exported for pytest harness) --------
ifeq ($(GATES),yes)
SIM_BUILD := sim_build/gl
COMPILE_ARGS += -DGL_TEST -DFUNCTIONAL -DUSE_POWER_PINS -DSIM -DUNIT_DELAY=\#1
VERILOG_SOURCES += \
  $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/primitives.v \
  $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v \
  $(PWD)/gate_level_netlist.v
else
SIM_BUILD := sim_build/rtl
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(PROJECT_SOURCES))
endif

COMPILE_ARGS += -I$(SRC_DIR)
VERILOG_SOURCES += $(PWD)/tb.sv
TOPLEVEL := tb

# Export so pytest / your runner can read them if you choose to
export SIM FST TOPLEVEL_LANG GATES SRC_DIR PROJECT_SOURCES SIM_BUILD VERILOG_SOURCES COMPILE_ARGS TOPLEVEL

.PHONY: all
all: pytest

.PHONY: pytest
pytest:
	$(PYTEST) $(PYTEST_ARGS) --junitxml=$(JUNIT) $(TEST_FILE) -rA $(if $(ARGS),-k "$(ARGS)",)

.PHONY: clean
clean:
	rm -rf sim_build run __pycache__ .pytest_cache $(JUNIT)