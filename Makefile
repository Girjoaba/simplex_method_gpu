# Compiler flags
NVCC        := nvcc
CXX         := g++
CXXFLAGS    := --std=c++20
LIBS        := -lcublas

# Paths
SRC_DIR     	:= src
SRC_CUDA_DIR    := $(SRC_DIR)/cuda
SRC_GLPK_DIR    := $(SRC_DIR)/glpk
INPUT_DIR   	:= input
BIN_SOLVER_DIR  := bin_solver
BIN_GLPK_DIR    := bin_glpk

# GLPK
GLPK_PROGS   := glpk_interface glpk_solver
GLPK_TARGETS := $(patsubst %, $(BIN_GLPK_DIR)/%, $(GLPK_PROGS))
GLPK_LIBS    := -lglpk

INPUT_FILE  := $(INPUT_DIR)/sample.txt

# =============================================== |
# ------------ Dynamic Source Loading ----------- |
# =============================================== |


# 1. match v*.cu
SRCS := $(wildcard $(SRC_CUDA_DIR)/v*_*.cu)

# 2. extract version numbers
VERSIONS := $(shell echo "$(SRCS)" | tr ' ' '\n' | sed -n 's|.*/v\([0-9]\+\)_.*|\1|p' | tr '\n' ' ')

# 3. generate targets from version numbers
TARGETS := $(patsubst %, $(BIN_SOLVER_DIR)/solver%.out, $(VERSIONS))

# 4. generate 'run' target names from the version numbers
RUN_TARGETS := $(patsubst %, run%, $(VERSIONS))

# =============================================== |
# ------------------ Targets -------------------- |
# =============================================== |

all: $(TARGETS) $(GLPK_TARGETS)


# build glpk "interface" and "solver"
$(BIN_GLPK_DIR)/%: $(SRC_GLPK_DIR)/%.cpp
	@mkdir -p $(BIN_GLPK_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(GLPK_LIBS)
	

# === Compile rule (Dynamic) ===
# This rule matches targets like 'bin/solver1.out', 'bin/solver2.out', etc.
# The '%' wildcard captures the version number (1, 2, ...).
#
# .SECONDEXPANSION is used to delay the 'wildcard' evaluation.
# This way, '%' is first expanded (e.g., to '1'), and *then*
# make searches for the dependency (e.g., $(wildcard src/v1_*.cu)).
.SECONDEXPANSION:
$(BIN_SOLVER_DIR)/solver%.out: $$(wildcard $(SRC_CUDA_DIR)/v%_*.cu)
	@mkdir -p $(BIN_SOLVER_DIR)
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LIBS)


# === Run rules (Dynamic) ===
# This creates phony targets like 'run1', 'run2', etc.
# 'run1' depends on 'bin/solver1.out'
# '$<' refers to the first dependency (bin/solverN.out)
$(RUN_TARGETS): run%: $(BIN_SOLVER_DIR)/solver%.out
	@./$< $(INPUT_FILE)


clean:
	rm -rf $(BIN_SOLVER_DIR) $(BIN_GLPK_DIR)

.PHONY: all clean $(RUN_TARGETS)
