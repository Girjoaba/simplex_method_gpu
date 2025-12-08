EIGEN_DIR ?= $(HOME)/eigen-3.4.0

# Compiler flags
NVCC        := nvcc
CXX         := g++
CXXFLAGS    := -I $(EIGEN_DIR) --std=c++17 -O3
CPP_FLAGS	:= -march=native -ffp-contract=fast
CUDA_FLAGS  := -arch=sm_80 -lcusolver -lcudart -lcublas
# LIBS        := -lcublas

# Paths
SRC_DIR         := src
SRC_CUDA_DIR    := $(SRC_DIR)/cuda_slow
SRC_GLPK_DIR    := $(SRC_DIR)/glpk
INPUT_DIR       := input
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

# 1. Match all source files (e.g., src/cuda_slow/v1_cpu.cu)
SRCS := $(wildcard $(SRC_CUDA_DIR)/v*_*.cu)

# 2. Extract filenames without extension (e.g., v1_cpu)
#    $(notdir ...) removes the directory path
#    $(basename ...) removes the .cu extension
BASENAMES := $(basename $(notdir $(SRCS)))

# 3. Generate output targets (e.g., bin_solver/v1_cpu.out)
TARGETS := $(patsubst %, $(BIN_SOLVER_DIR)/%.out, $(BASENAMES))

# =============================================== |
# ------------------ Targets -------------------- |
# =============================================== |

all: $(TARGETS)

# Build GLPK tools
$(BIN_GLPK_DIR)/%: $(SRC_GLPK_DIR)/%.cpp
	@mkdir -p $(BIN_GLPK_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(GLPK_LIBS)

# === Compile Rule (Pattern Match) ===
# This maps 'bin_solver/NAME.out' directly to 'src/cuda_slow/NAME.cu'
$(BIN_SOLVER_DIR)/%.out: $(SRC_CUDA_DIR)/%.cu
	@mkdir -p $(BIN_SOLVER_DIR)
	$(NVCC) $(CXXFLAGS) $(CUDA_FLAGS) $< -o $@ $(LIBS)

# === Run Rules ===
# Allows running specific versions like: make run-v1_cpu
run-%: $(BIN_SOLVER_DIR)/%.out
	@./$< $(INPUT_FILE)

clean:
	rm -rf $(BIN_SOLVER_DIR) $(BIN_GLPK_DIR)

.PHONY: all clean