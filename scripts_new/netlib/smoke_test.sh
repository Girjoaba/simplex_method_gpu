#!/bin/bash
# Smoke test script for benchmarking setup
# Run from: ~/simplex_method_gpu

set -euo pipefail

# Activate Python venv for gurobipy
source .venv/bin/activate

LOG_DIR="test/netlib/smoke_test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== SMOKE TEST START: $TIMESTAMP ===" | tee "$LOG_DIR/summary_$TIMESTAMP.log"

# --- Phase 1: Setup ---
echo "[Phase 1] Setting up problem set..."
cd scripts_new/netlib
cp problem_summary.csv problem_summary.csv.bak
cp problem_summary_backup.csv problem_summary.csv
cd ../..

# --- Phase 2: Prepare all 69 problems ---
echo "[Phase 2] Preprocessing all 69 problems..."
bash scripts_new/netlib/prepare_problems.sh 2>&1 | tee "$LOG_DIR/prepare_$TIMESTAMP.log"

# --- Phase 3: Build solvers ---
echo "[Phase 3] Building solvers..."
make clean && make 2>&1 | tee "$LOG_DIR/build_$TIMESTAMP.log"

# --- Phase 4: Interface test (all solvers, 1 problem) ---
echo "[Phase 4] Interface test: All solvers on AFIRO..."

# Create single-problem test file
echo 'Name,Rows,Cols,Nonzeros,Bytes,BR,Optimal Value
AFIRO,28,32,88,794,,-4.6475314286E+02' > scripts_new/netlib/problem_summary.csv

for solver in tp_v1_baseline tp_v2_accurate_xB tp_v3_fast_xB bm_v8_full_gpu; do
    echo "  Testing $solver..."
    scripts_new/netlib/solve_and_compare.sh "bin_solver/${solver}.out" 200000 2>&1 \
        | tee "$LOG_DIR/interface_${solver}_$TIMESTAMP.log"
done

# --- Phase 5: Problem test (1 solver, all 69 problems) ---
echo "[Phase 5] Problem test: tp_v3 on all 69 problems..."
cp scripts_new/netlib/problem_summary_backup.csv scripts_new/netlib/problem_summary.csv

scripts_new/netlib/solve_and_compare.sh bin_solver/tp_v3_fast_xB.out 200000 2>&1 \
    | tee "$LOG_DIR/problems_tp_v3_$TIMESTAMP.log"

# --- Cleanup ---
echo "[Cleanup] Restoring original problem_summary.csv..."
cp scripts_new/netlib/problem_summary.csv.bak scripts_new/netlib/problem_summary.csv

echo "=== SMOKE TEST COMPLETE ===" | tee -a "$LOG_DIR/summary_$TIMESTAMP.log"
echo "Logs saved to: $LOG_DIR/"
