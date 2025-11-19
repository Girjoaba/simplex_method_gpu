#!/usr/bin/env bash
# Simple correctness test for GLOP and Gurobi solvers

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root (parent of benchmarks directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Solvers to test
GLOP_BINARY="$PROJECT_ROOT/glop_baseline/build/glop_canonical"
GUROBI_BINARY="$PROJECT_ROOT/src/gurobi/gurobi_canonical"

# Test problems and expected optima
PROBLEMS=(
    "sample:9.0"
    "sample3:4620.8333333"
    "afiro:464.75314285714285"
    "adlittle:-225494.96316238024"
    "e226:18.751929066370547"
)

# Tolerance for comparison (relative)
RTOL=1e-4

echo "======================================================================"
echo "Correctness Test: GLOP vs Gurobi"
echo "======================================================================"
echo ""

# Function to compare floats with relative tolerance
compare_floats() {
    local computed=$1
    local expected=$2
    local rtol=$3

    # Use Python for precise float comparison
    python3 -c "
import sys
computed = float('$computed')
expected = float('$expected')
rtol = float('$rtol')

if expected == 0:
    diff = abs(computed - expected)
    if diff < rtol:
        sys.exit(0)
    else:
        sys.exit(1)
else:
    rel_diff = abs((computed - expected) / expected)
    if rel_diff < rtol:
        sys.exit(0)
    else:
        sys.exit(1)
"
}

# Function to test a solver on a problem
test_solver() {
    local solver_name=$1
    local solver_binary=$2
    local problem_name=$3
    local expected_optimum=$4

    local input_file="$PROJECT_ROOT/test/input/${problem_name}.canonical"

    # Run solver and capture output (both stdout and stderr)
    local output
    local stderr
    local exit_code
    stderr=$(mktemp)
    output=$("$solver_binary" "$input_file" 2>"$stderr") || exit_code=$?

    if [ -n "$exit_code" ]; then
        echo -e "${RED}✗ $solver_name/$problem_name: Solver failed${NC}"
        if [ -s "$stderr" ]; then
            echo -e "${YELLOW}Error output:${NC}"
            cat "$stderr" | head -10
        fi
        rm -f "$stderr"
        return 1
    fi
    rm -f "$stderr"

    # Extract optimum value
    local optimum=$(echo "$output" | grep "Optimum found:" | awk '{print $3}')

    if [ -z "$optimum" ]; then
        echo -e "${RED}✗ $solver_name/$problem_name: Could not parse optimum${NC}"
        return 1
    fi

    # Validate optimum
    if compare_floats "$optimum" "$expected_optimum" "$RTOL"; then
        echo -e "${GREEN}✓ $solver_name/$problem_name: $optimum${NC}"
        return 0
    else
        echo -e "${RED}✗ $solver_name/$problem_name: got $optimum, expected $expected_optimum${NC}"
        return 1
    fi
}

# Test all solvers on all problems
total_tests=0
passed_tests=0

for entry in "${PROBLEMS[@]}"; do
    problem=$(echo "$entry" | cut -d':' -f1)
    expected=$(echo "$entry" | cut -d':' -f2)

    # Test GLOP
    ((total_tests++))
    if test_solver "GLOP" "$GLOP_BINARY" "$problem" "$expected"; then
        ((passed_tests++))
    fi

    # Test Gurobi
    ((total_tests++))
    if test_solver "Gurobi" "$GUROBI_BINARY" "$problem" "$expected"; then
        ((passed_tests++))
    fi
done

echo ""
echo "======================================================================"
echo "Results: $passed_tests/$total_tests tests passed"
echo "======================================================================"

if [ $passed_tests -eq $total_tests ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
