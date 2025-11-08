# === Compiler and flags ===
NVCC        := nvcc
CXXFLAGS    := --std=c++20
CCBIN       := /usr/bin/g++-13
LIBS        := -lcublas

# === Paths ===
SRC_DIR     := src
BIN_DIR     := bin
INPUT_DIR   := input

# === Files ===
INPUT_FILE  := $(INPUT_DIR)/sample.txt

# === Dynamic File Discovery ===
# 1. Find all source files matching the pattern 'vN_....cu'
SRCS := $(wildcard $(SRC_DIR)/v*_*.cu)

# 2. Extract the base filenames (e.g., v1_foo.cu, v2_bar.cu)
BASENAMES := $(notdir $(SRCS))

# 3. Extract just the version numbers (e.g., "1 2 3")
#    Uses 'tr' to put each file on a new line,
#    'sed' to extract the number (v<num>_... -> <num>),
#    and 'tr' again to put them back on one line.
VERSIONS := $(shell echo "$(BASENAMES)" | tr ' ' '\n' | sed -n 's/v\([0-9]\+\)_.*/\1/p' | tr '\n' ' ')

# 4. Generate target executable names from version numbers
#    (e.g., 1 -> bin/solver1.out, 2 -> bin/solver2.out)
TARGETS := $(patsubst %, $(BIN_DIR)/solver%.out, $(VERSIONS))

# 5. Generate 'run' target names from the version numbers
#    (e.g., 1 -> run1, 2 -> run2)
RUN_TARGETS := $(patsubst %, run%, $(VERSIONS))

# === Default target ===
# 'all' will build all found executables
all: $(TARGETS)

# === Compile rule (Dynamic) ===
# This rule matches targets like 'bin/solver1.out', 'bin/solver2.out', etc.
# The '%' wildcard captures the version number (1, 2, ...).
#
# .SECONDEXPANSION is used to delay the 'wildcard' evaluation.
# This way, '%' is first expanded (e.g., to '1'), and *then*
# make searches for the dependency (e.g., $(wildcard src/v1_*.cu)).
.SECONDEXPANSION:
$(BIN_DIR)/solver%.out: $$(wildcard $(SRC_DIR)/v%_*.cu)
	@mkdir -p $(BIN_DIR)
	@echo "--- Compiling: $<  ->  $@ ---"
	$(NVCC) $(CXXFLAGS) $< -o $@ -ccbin $(CCBIN) $(LIBS)

# === Run rules (Dynamic) ===
# This creates phony targets like 'run1', 'run2', etc.
# 'run1' depends on 'bin/solver1.out'
# '$<' refers to the first dependency (bin/solverN.out)
$(RUN_TARGETS): run%: $(BIN_DIR)/solver%.out
	@echo "--- Running: $< $(INPUT_FILE) ---"
	@./$< $(INPUT_FILE)

# === Clean rule ===
# Removes the entire bin directory
clean:
	@echo "--- Cleaning $(BIN_DIR) ---"
	rm -rf $(BIN_DIR)

# === Phony targets ===
# Declare targets that are not files to prevent conflicts
.PHONY: all clean $(RUN_TARGETS)