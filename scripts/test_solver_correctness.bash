#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------|
# | Script Description                         |
# ---------------------------------------------|
# | 1. For every groundtruth                   |
# | 1.1.      find the actual problem          |
# | 1.2.      feed it into the glpk interface  |
# | 1.3.      feed into our solver             |
# | 1.3.      compare with the gt result       |
# ---------------------------------------------|

make

PROBLEMS_DIR="./problems"
GROUNDTRUTH_DIR="./test/groundtruth"
EXPERIMENT_DIR="./test/experiment"

GLPK_INTERFACE="./bin_glpk/glpk_interface"
SOLVER_BIN="${1:-./bin_solver/solver1.out}"   # optional arg

# sanity
if [ ! -x "$GLPK_INTERFACE" ]; then
    echo "error: $GLPK_INTERFACE not found or not executable" >&2
    exit 1
fi
if [ ! -x "$SOLVER_BIN" ]; then
    echo "error: solver '$SOLVER_BIN' not found or not executable" >&2
    exit 1
fi

# clean experiment dir
rm -rf "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR"

echo "[info] writing results to $EXPERIMENT_DIR"

attempted=0          # has matching problem file
glpk_errors=0
solver_errors=0
success=0            # solver ran OK
compared=0
correct=0
wrong=0

for gt in "$GROUNDTRUTH_DIR"/*.txt; do
    # if no files, skip pattern literal
    if [ ! -f "$gt" ]; then
        continue
    fi

    base_gt=$(basename "$gt")          # e.g. timtab1.mps.txt
    problem_name=${base_gt%.txt}       # e.g. timtab1.mps
    problem_path="$PROBLEMS_DIR/$problem_name"

    if [ ! -f "$problem_path" ]; then
        echo "[warn] problem file not found for $gt -> expected $problem_path, skipping"
        continue
    fi

    attempted=$((attempted + 1))
    echo "[run] $problem_path"

    tmp_lp=$(mktemp)

    # 1) MPS -> LP
    if ! "$GLPK_INTERFACE" "$problem_path" > "$tmp_lp"; then
        echo "[warn] glpk_interface failed for $problem_path, skipping"
        rm -f "$tmp_lp"
        glpk_errors=$((glpk_errors + 1))
        continue
    fi

    # 2) run solver on lp
    out_file="$EXPERIMENT_DIR/$problem_name.out"
    if ! "$SOLVER_BIN" "$tmp_lp" > "$out_file"; then
        echo "[warn] solver failed for $problem_path, skipping"
        rm -f "$tmp_lp" "$out_file"
        solver_errors=$((solver_errors + 1))
        continue
    fi
    rm -f "$tmp_lp"
    success=$((success + 1))

    # 3) compare with ground truth
    # ground truth should have just the numeric optimal value (maybe in 1.23e+04)
    gt_val=$(tr -d ' \t\r\n' < "$gt")

    # extract "Optimum found: NUM" from experiment
    exp_line=$(grep '^Optimum found:' "$out_file" || true)

    compared=$((compared + 1))

    if [ -z "$exp_line" ]; then
        echo "[compare] $problem_name: experiment missing 'Optimum found:' -> WRONG"
        wrong=$((wrong + 1))
        continue
    fi

    # extract the number (3rd field)
    exp_val=$(printf '%s\n' "$exp_line" | awk '{print $3}')

    # if either is empty -> wrong
    if [ -z "$exp_val" ] || [ -z "$gt_val" ]; then
        echo "[compare] $problem_name: cannot parse numbers -> WRONG"
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
        echo "[compare] $problem_name: OK (exp=$exp_val, gt=$gt_val)"
        correct=$((correct + 1))
    else
        echo "[compare] $problem_name: MISMATCH (exp=$exp_val, gt=$gt_val)"
        wrong=$((wrong + 1))
    fi
done

echo "===================================="
echo "Attempted (had matching problem) : $attempted"
echo "GLPK interface errors            : $glpk_errors"
echo "Solver errors                    : $solver_errors"
echo "Successfully solved              : $success"
echo "Compared                         : $compared"
echo "Correct                          : $correct"
echo "Wrong                            : $wrong"
echo "Experiment dir                   : $EXPERIMENT_DIR"
