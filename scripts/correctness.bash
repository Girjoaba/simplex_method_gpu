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

# We are all grown up, run make yourself
# make

if [ $# -lt 1 ]; then
    echo "Error incorrect number of arguments."
    echo "Usage: $0 <path_to_solver_binary> [small|medium|big]"
    echo "Example: $0 ./bin_solver/v1_cpu.out small"
    exit 1
fi


INPUT_DIR="./test/input"
GROUNDTRUTH_DIR="./test/groundtruth"
EXPERIMENT_DIR="./test/experiment"

SOLVER_BIN="$1"
SIZE_CLASS="$2"
if [ "$SIZE_CLASS" == "small" ]; then
    MAX_SIZE=300  # Run problems up to 600 KB
    SIZE_FILTER_INFO="Running SMALL problems (max 600 KB)"
elif [ "$SIZE_CLASS" == "medium" ]; then
    MAX_SIZE=500 # Run problems up to 10 MB
    SIZE_FILTER_INFO="Running MEDIUM problems (max 100000 KB)"
elif [ "$SIZE_CLASS" == "big" ]; then
    # A large number to effectively disable the MAX_SIZE filter for 'big' problems
    MAX_SIZE=5000 
    SIZE_FILTER_INFO="Running BIG problems (max 500000 KB, effectively all)"
else
    echo "error: invalid size class '$SIZE_CLASS'. Must be one of: small, medium, big, or omitted." >&2
    exit 1
fi

# =====================================================
# Filter flag:
#   - Leave empty ("") to run on ALL canonical files
#   - Set to a problem root name to run only that one,
#     e.g.: TARGET_PROBLEM="adlittle"
#     This corresponds to:
#       ./test/input/adlittle.canonical
#       ./test/groundtruth/adlittle.mps.txt
# =====================================================

# bore3d if you want to test numerical stability
TARGET_PROBLEM=""

# sanity
if [ ! -x "$SOLVER_BIN" ]; then
    echo "error: solver '$SOLVER_BIN' not found or not executable" >&2
    exit 1
fi

# clean experiment dir
rm -rf "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR"

echo "[info] writing results to $EXPERIMENT_DIR"

attempted=0          # has matching gt file
glpk_errors=0        # repurposed: missing gt / mapping issues
solver_errors=0
success=0            # solver ran OK
compared=0
correct=0
wrong=0

if [[ $(basename "$SOLVER_BIN") == bm* ]]; then
    echo "[info] Solver binary starts with 'bm'. Using *.canonical files."
    readarray -t problems_array < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.canonical")
else
    echo "[info] Solver binary starts with 'tp'. Using *.twophase files."
    readarray -t problems_array < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.twophase")
fi

# if [ -n "$TARGET_PROBLEM" ]; then
#     echo "[info] filtering to problem: $TARGET_PROBLEM"
# else
#     echo "[info] running on all problems: ${problems_array[@]}"
# fi

for canonical_path in "${problems_array[@]}"; do
    base_name="${canonical_path%.*}"
    mps_path="${base_name}.mps"
    problem_size=$(du -k "$mps_path" 2>/dev/null | awk '{print $1}' || echo 0)
    # problem_size=$(du -k "$canonical_path" 2>/dev/null | awk '{print $1}' || echo 0)
    if [ "$problem_size" -gt "$MAX_SIZE" ]; then
      continue
    fi
    # if no files, skip pattern literal
    if [ ! -f "$canonical_path" ]; then
        continue
    fi

    canonical_base=$(basename "$canonical_path")      # e.g. adlittle.canonical

    if [[ $(basename "$SOLVER_BIN") == bm* ]]; then
        problem_root=${canonical_base%.canonical} # e.g. adlittle
    else
        problem_root=${canonical_base%.twophase} # e.g. adlittle
    fi

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

    # 1) run solver on canonical file
    temp_stdout=$(mktemp --suffix=.stdout.tmp)
    temp_stderr=$(mktemp --suffix=.stderr.tmp)
    temp_time=$(mktemp --suffix=.time.tmp)
    { time "$SOLVER_BIN" < "$canonical_path" > "$temp_stdout" 2>> "$temp_stderr" ; } 2>> "$temp_time"

    solver_exit_code=$?

    stdout=$(cat "$temp_stdout")
    stderr=$(cat "$temp_stderr")
    if [ "$solver_exit_code" -ne 0 ] ; then
        echo "[warn] ⭕ solver failed for $canonical_path, skipping"
        echo "stdout:"
        echo "$stdout"
        echo "stderr:"
        echo "$stderr"
        rm -rf "$temp_stdout" "$temp_stderr" "$temp_time"
        solver_errors=$((solver_errors + 1))

        continue
    fi
    success=$((success + 1))

    # 2) compare with ground truth
    # ground truth has just the numeric optimal value
    gt_val=$(tr -d ' \t\r\n' < "$gt")

    # extract "Optimum found: NUM" from experiment
    exp_line=$(grep '^Optimum found:' "$stdout" || true)

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
    # Handle NaN returns
    if ! echo "$exp_val" | grep -E '^[+\-]?[0-9.]*([Ee][+\-][0-9]+)?$' >/dev/null || \ ! echo "$gt_val" | grep -E '^[+\-]?[0-9.]*([Ee][+\-][0-9]+)?$' >/dev/null; then
      echo "[compare] ❌ $problem_name: value is NOT a valid number (e.g., NaN or invalid format) -> WRONG"
      echo "         (Got: $exp_val, Expected: $gt_val)"
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
        real_time=$(grep '^real' "$temp_stderr" | awk '{print $2}' | tr -d '[:space:]')
        echo "[compare] $problem_name: OK ✅ (got=$exp_val, expected=$gt_val) in "$real_time
        correct=$((correct + 1))
    else
        echo "[compare] ❌ $problem_name: MISMATCH (got=$exp_val, expected=$gt_val)"
        wrong=$((wrong + 1))
    fi
    rm -rf "$temp_stdout" "$temp_stderr" "$temp_time"
done

echo "===================================="
echo "Attempted (had matching gt)      : $attempted"
echo "GT / mapping errors ⭕           : $glpk_errors"
# echo "Solver errors ⭕                 : $solver_errors"
echo "Compared                         : $compared"
echo "Correct ✅                       : $correct"
echo "Experiment dir                   : $EXPERIMENT_DIR"
echo "===================================="
if [ "$correct" -eq "$attempted" ]; then
    echo "All tests passed!"
else
    echo "Wrong ❌                         : $wrong"
    echo "Wrong tests! </3"
fi
