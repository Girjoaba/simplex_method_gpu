#!/bin/bash
# Run tp_v1 on all 69 problems with nohup for persistence
# Run from: ~/simplex_method_gpu
#
# Usage:
#   ./scripts/netlib/run_problem_test.sh         # foreground
#   nohup ./scripts/netlib/run_problem_test.sh & # background (persistent)
#
# Check progress: tail -f test/netlib/problem_test.log
# Check if running: ps aux | grep run_problem_test

set -euo pipefail

cd "$(dirname "$0")/../.."  # Go to repo root

source .venv/bin/activate

LOG_DIR="test/netlib"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/problem_test.log"

echo "=== PROBLEM TEST START: $(date) ===" | tee "$LOG_FILE"
echo "Solver: tp_v1_baseline" | tee -a "$LOG_FILE"
echo "Problems: 69 (from problem_summary.csv)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Ensure we're using the full problem set
cp scripts/netlib/problem_summary.csv.bak scripts/netlib/problem_summary.csv

# Run the solver on all problems
scripts/netlib/solve_and_compare.sh bin_solver/tp_v1_baseline.out 200000 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== PROBLEM TEST COMPLETE: $(date) ===" | tee -a "$LOG_FILE"
