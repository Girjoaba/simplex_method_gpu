# GLOP Baseline Makefile Wrapper
# Convenience targets for common build operations

.PHONY: all configure build quick-build clean-cache clean test help

# Default target
all: build

# Initial configuration (run cmake)
configure:
	@echo "Configuring CMake..."
	@mkdir -p build
	@cd build && cmake ..

# Full build from scratch (includes configure)
build: configure
	@echo "Building glop_canonical..."
	@cd build && $(MAKE) glop_canonical -j4

# Quick rebuild (assumes already configured)
quick-build:
	@echo "Quick rebuild..."
	@cd build && $(MAKE) glop_canonical -j4

# Clean CMake cache only (preserves downloaded dependencies in _deps/)
clean-cache:
	@echo "Cleaning CMake cache (keeping dependencies)..."
	@rm -f build/CMakeCache.txt
	@rm -rf build/CMakeFiles/
	@echo "Done. Run 'make configure' to reconfigure."

# Full clean (WARNING: deletes all build artifacts including _deps/)
clean:
	@echo "WARNING: This will delete ALL build artifacts including downloaded dependencies!"
	@echo "Consider using 'make clean-cache' instead to preserve dependencies."
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf build; \
		echo "Build directory removed."; \
	else \
		echo "Cancelled."; \
	fi

# Run unit tests
test:
	@echo "Running tests..."
	@cd build && $(MAKE) glop_test
	@cd build && ./glop_test

# Run benchmark
benchmark:
	@echo "Running benchmark..."
	@cd build && $(MAKE) glop_benchmark
	@cd build && ./glop_benchmark

# Test with sample input
test-sample:
	@echo "Testing glop_canonical with sample.txt..."
	@cd build && ./glop_canonical ../input/sample.txt

# Help message
help:
	@echo "GLOP Baseline Makefile"
	@echo ""
	@echo "Common targets:"
	@echo "  make build        - Full build (configure + compile)"
	@echo "  make quick-build  - Rebuild only (assumes configured)"
	@echo "  make configure    - Run CMake configuration"
	@echo "  make clean-cache  - Clean CMake cache (keeps _deps/)"
	@echo "  make clean        - DANGEROUS: Full clean (deletes _deps/)"
	@echo "  make test         - Run unit tests"
	@echo "  make benchmark    - Run performance benchmark"
	@echo "  make test-sample  - Test with sample.txt"
	@echo ""
	@echo "Build time estimates:"
	@echo "  First build:       10-15 minutes (downloads dependencies)"
	@echo "  After clean-cache: 30-60 seconds (reuses _deps/)"
	@echo "  Quick rebuild:     5-10 seconds (incremental)"
	@echo ""
	@echo "Recommended workflow:"
	@echo "  1. First time:     make build"
	@echo "  2. After changes:  make quick-build"
	@echo "  3. If issues:      make clean-cache && make build"
