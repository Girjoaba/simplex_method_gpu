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
TARGET      := $(BIN_DIR)/solver.out
SRC         := $(SRC_DIR)/solver.cu
INPUT_FILE  := $(INPUT_DIR)/sample.txt

# === Default target ===
all: $(TARGET)

# === Compile rule ===
$(TARGET): $(SRC)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) -ccbin $(CCBIN) $(LIBS)

# === Run rule ===
run: $(TARGET)
	@./$(TARGET) $(INPUT_FILE)

# === Clean rule ===
clean:
	rm -f $(TARGET)

.PHONY: all run clean