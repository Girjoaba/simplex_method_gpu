#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------|
# | Script Description                         |
# ---------------------------------------------|
# | 1. For every canonical file                |
# | 1.1.      find the groundtruth file        |
# | 1.2.      feed canonical into solver bin   |
# | 1.3.      compare with the gt result       |
# ---------------------------------------------|

make

INPUT_DIR="./test/input"
GROUNDTRUTH_DIR="./test/groundtruth"
EXPERIMENT_DIR="./test/experiment"

SOLVER_BIN="${1:-./bin_solver/solver1.out}"   # optional arg

# =====================================================
# Filter flag:
#   - Leave empty ("") to run on ALL canonical files
#   - Set to a problem root name to run only that one,
#     e.g.: TARGET_PROBLEM="adlittle"
#     This corresponds to:
#       ./test/input/adlittle.canonical
#       ./test/groundtruth/adlittle.mps.txt
# =====================================================
TARGET_PROBLEM="" # afiro

# sanity
if [ ! -x "$SOLVER_BIN" ]; then
    echo "error: solver '$SOLVER_BIN' not found or not executable" >&2
    exit 1
fi

# clean experiment dir
rm -rf "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR"

echo "[info] writing results to $EXPERIMENT_DIR"
if [ -n "$TARGET_PROBLEM" ]; then
    echo "[info] filtering to problem: $TARGET_PROBLEM"
else
    echo "[info] running on all problems"
fi

attempted=0          # has matching gt file
glpk_errors=0        # repurposed: missing gt / mapping issues
solver_errors=0
success=0            # solver ran OK
compared=0
correct=0
wrong=0

for canonical_path in "$INPUT_DIR"/*.canonical; do
    # if no files, skip pattern literal
    if [ ! -f "$canonical_path" ]; then
        continue
    fi

    canonical_base=$(basename "$canonical_path")      # e.g. adlittle.canonical
    problem_root=${canonical_base%.canonical}         # e.g. adlittle

    # Apply filter if TARGET_PROBLEM is set
    if [ -n "$TARGET_PROBLEM" ] && [ "$problem_root" != "$TARGET_PROBLEM" ]; then
        continue
    fi

    problem_name="${problem_root}.mps"                # e.g. adlittle.mps
    gt="$GROUNDTRUTH_DIR/$problem_name.txt"           # e.g. ./test/groundtruth/adlittle.mps.txt

    if [ ! -f "$gt" ]; then
        echo "[warn] ⭕ groundtruth file not found for $canonical_base -> expected $gt, skipping"
        glpk_errors=$((glpk_errors + 1))
        continue
    fi

    attempted=$((attempted + 1))
    # echo "[run] $canonical_path"

    # 1) run solver on canonical file
    out_file="$EXPERIMENT_DIR/$problem_name.txt"
    if ! "$SOLVER_BIN" "$canonical_path" > "$out_file"; then
        echo "[warn] ⭕ solver failed for $canonical_path, skipping"
        rm -f "$out_file"
        solver_errors=$((solver_errors + 1))
        continue
    fi
    success=$((success + 1))

    # 2) compare with ground truth
    # ground truth has just the numeric optimal value
    gt_val=$(tr -d ' \t\r\n' < "$gt")

    # extract "Optimum found: NUM" from experiment
    exp_line=$(grep '^Optimum found:' "$out_file" || true)

    compared=$((compared + 1))

    if [ -z "$exp_line" ]; then
        echo "[compare] ❌ $problem_name: experiment missing 'Optimum found:' -> WRONG"
        wrong=$((wrong + 1))
        continue
    fi

    # extract the number (3rd field)
    exp_val=$(printf '%s\n' "$exp_line" | awk '{print $3}')

    # if either is empty -> wrong
    if [ -z "$exp_val" ] || [ -z "$gt_val" ]; then
        echo "[compare] ❌ $problem_name: cannot parse numbers -> WRONG"
        wrong=$((wrong + 1))
        continue
    fi

    # compare with a small tolerance using awk (handles scientific notation)
    # tolerance = 1e-4 * max(1, |gt|)
    if awk -v a="$exp_val" -v b="$gt_val" 'BEGIN {
        da = (a - b); if (da < 0) da = -da;
        ab = b; if (ab < 0) ab = -ab;
        tol = 1e-4;
        if (ab > 1) tol = tol * ab;
        exit !(da <= tol);
    }'; then
        echo "[compare] $problem_name: OK ✅ (exp=$exp_val, gt=$gt_val)"
        correct=$((correct + 1))
    else
        echo "[compare] ❌ $problem_name: MISMATCH (exp=$exp_val, gt=$gt_val)"
        wrong=$((wrong + 1))
    fi
done

echo "===================================="
echo "Attempted (had matching gt)      : $attempted"
# echo "GT / mapping errors ⭕           : $glpk_errors"
echo "Solver errors ⭕                 : $solver_errors"
# echo "Successfully solved              : $success"
echo "Compared                         : $compared"
echo "Correct ✅                       : $correct"
echo "Wrong ❌                         : $wrong"
echo "Experiment dir                   : $EXPERIMENT_DIR"
