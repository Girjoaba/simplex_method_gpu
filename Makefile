EIGEN_DIR ?= $(HOME)/eigen-3.4.0

# Compiler flags
NVCC        := nvcc
CXX         := g++
CXXFLAGS    := -I $(EIGEN_DIR) --std=c++17 -O3
CPP_FLAGS	  := -march=native -ffp-contract=fast
CUDA_FLAGS  := -arch=sm_80 -lcusolver -lcudart -lcublas
# LIBS        := -lcublas

# Paths
SRC_DIR           := src
SRC_BIG_M_DIR     := $(SRC_DIR)/big_m
SRC_GLPK_DIR      := $(SRC_DIR)/glpk
SRC_TWO_PHASE_DIR := $(SRC_DIR)/two_phase
INPUT_DIR         := input
BIN_SOLVER_DIR    := bin_solver
BIN_GLPK_DIR      := bin_glpk

# GLPK
GLPK_PROGS   := glpk_interface glpk_solver
GLPK_TARGETS := $(patsubst %, $(BIN_GLPK_DIR)/%, $(GLPK_PROGS))
GLPK_LIBS    := -lglpk

INPUT_FILE  := $(INPUT_DIR)/sample.txt

# =============================================== |
# ------------ Dynamic Source Loading ----------- |
# =============================================== |

# --- Big M Sources (will be prefixed with bm_) ---
# 1. Match all source files (e.g., src/big_m/v1_cpu.cu)
BIG_M_SRCS := $(wildcard $(SRC_BIG_M_DIR)/v*_*.cu)

# 2. Extract filenames without extension (e.g., v1_cpu)
BIG_M_BASENAMES_RAW := $(basename $(notdir $(BIG_M_SRCS)))

# 3. *** Apply 'bm_' prefix ***
BIG_M_BASENAMES := $(addprefix bm_, $(BIG_M_BASENAMES_RAW))

# 4. Generate output targets (e.g., bin_solver/bm_v1_cpu.out)
BIG_M_TARGETS := $(patsubst %, $(BIN_SOLVER_DIR)/%.out, $(BIG_M_BASENAMES))

# CPU solver target (Eigen-based, no CUDA) - renamed for consistency
CPU_TARGET := $(BIN_SOLVER_DIR)/bm_v1_cpu.out

# --- Two-Phase Sources (will be prefixed with tp_) ---
TWO_PHASE_SRCS := $(wildcard $(SRC_TWO_PHASE_DIR)/*.cu)

TWO_PHASE_BASENAMES_RAW := $(basename $(notdir $(TWO_PHASE_SRCS)))

# *** Apply 'tp_' prefix ***
TWO_PHASE_BASENAMES := $(addprefix tp_, $(TWO_PHASE_BASENAMES_RAW))

TWO_PHASE_TARGETS := $(patsubst %, $(BIN_SOLVER_DIR)/%.out, $(TWO_PHASE_BASENAMES))

# =============================================== |
# ------------------ Targets -------------------- |
# =============================================== |

all:  $(BIG_M_TARGETS) $(CPU_TARGET) $(TWO_PHASE_TARGETS)

# Build GLPK tools
$(BIN_GLPK_DIR)/%: $(SRC_GLPK_DIR)/%.cpp
	@mkdir -p $(BIN_GLPK_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(GLPK_LIBS)

# === Compile Rule (Pattern Match) ===
# This maps 'bin_solver/NAME.out' directly to 'src/cuda_slow/NAME.cu'
$(BIN_SOLVER_DIR)/bm_%.out: $(SRC_BIG_M_DIR)/%.cu
	@mkdir -p $(BIN_SOLVER_DIR)
	$(NVCC) $(CXXFLAGS) $(CUDA_FLAGS) $< -o $@ $(LIBS)

# === CPU Solver (Eigen-based, no CUDA) ===
$(CPU_TARGET): $(SRC_BIG_M_DIR)/v1_cpu.cpp
	@mkdir -p $(BIN_SOLVER_DIR)
	$(CXX) $(CXXFLAGS) $(CPP_FLAGS) $< -o $@

# === two-phase-method ===
$(BIN_SOLVER_DIR)/tp_%.out: $(SRC_TWO_PHASE_DIR)/%.cu
	@mkdir -p $(BIN_SOLVER_DIR)
	# Hardcoded, whatever
	$(NVCC) $< -o $@ \
        --std=c++20 \
        -ccbin /usr/bin/g++-13 \
        -I $(EIGEN_DIR) \
        -lcublas -lcusolver \
        --expt-relaxed-constexpr

# === Run Rules ===
# Allows running specific versions like: make run-v1_cpu
run-%: $(BIN_SOLVER_DIR)/%.out
	@./$< $(INPUT_FILE)

clean:
	rm -rf $(BIN_SOLVER_DIR) $(BIN_GLPK_DIR)

.PHONY: all clean
