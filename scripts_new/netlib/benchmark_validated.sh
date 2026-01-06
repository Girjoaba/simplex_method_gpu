#!/bin/bash
# benchmark_validated.sh - Benchmark solvers with validation
#
# For each solver, for each problem:
#   1. Run validation (1x) - if correct, save as run_id=0
#   2. If correct: run N more times (run_id=1..N), all recorded to TSV
#   3. If failed: log to failures.txt, continue to next
#
# Total runs per problem = 1 (validation) + extra_runs
#
# Usage: ./benchmark_validated.sh [max_nonzeros] [extra_runs] [timeout_sec] [force]
# Example: ./benchmark_validated.sh 10000000 4 3600        # skip existing (default)
# Example: ./benchmark_validated.sh 10000000 4 3600 force  # re-run all

# Don't use set -e so we can handle errors gracefully
set -uo pipefail

MAX_NONZEROS="${1:-10000000}"
EXTRA_RUNS="${2:-4}"
TIMEOUT_SEC="${3:-3600}"
FORCE_RERUN="${4:-}"  # Set to "force" to re-run existing benchmarks

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Solvers to benchmark
SOLVERS=(
    "tp_v1_baseline"
    "tp_v3_fast_xB"
    "bm_v5_thrust_max_elem"
    "bm_v8_full_gpu"
    "bm_v10_sherman_morris_opt"
    "bm_v11_sparse"
    "bm_v12_fused"
    "bm_v13_graph"
)

PROBLEM_SUMMARY="scripts_new/netlib/problem_summary.csv"
PREPROCESSED_DIR="test/netlib/preprocessed"
RESULTS_DIR="benchmarks/measurements_validated"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
FAILURES_FILE="${RESULTS_DIR}/failures_${TIMESTAMP}.txt"
SUMMARY_FILE="${RESULTS_DIR}/summary_${TIMESTAMP}.txt"

# Counters for summary
declare -A solver_correct
declare -A solver_failed

# ------------------------------------------------------------------------------
# Validation function
# Returns: "correct|time|optimum|iterations" or an error description
# The data after | is only present if validation passed
# ------------------------------------------------------------------------------
validate_single_run() {
    local solver_bin="$1"
    local problem_file="$2"
    local expected_val="$3"

    local tmp_stdout=$(mktemp)
    local tmp_stderr=$(mktemp)

    local exit_code=0
    local start_time=$(date +%s.%N)
    timeout "$TIMEOUT_SEC" "$solver_bin" < "$problem_file" > "$tmp_stdout" 2> "$tmp_stderr" || exit_code=$?
    local end_time=$(date +%s.%N)
    local real_time=$(awk "BEGIN {printf \"%.6f\", $end_time - $start_time}")

    # Check for timeout (exit code 124)
    if [ "$exit_code" -eq 124 ]; then
        rm -f "$tmp_stdout" "$tmp_stderr"
        echo "timeout_${TIMEOUT_SEC}s"
        return
    fi

    # Check for crash/error
    if [ "$exit_code" -ne 0 ]; then
        rm -f "$tmp_stdout" "$tmp_stderr"
        echo "error_exit_code_$exit_code"
        return
    fi

    # Check for unbounded (check both stdout and stderr)
    if grep -qi "unbounded" "$tmp_stdout" "$tmp_stderr" 2>/dev/null; then
        rm -f "$tmp_stdout" "$tmp_stderr"
        echo "unbounded"
        return
    fi

    # Extract optimum
    local optimum_line=$(grep '^Optimum found:' "$tmp_stdout" || true)
    if [ -z "$optimum_line" ]; then
        rm -f "$tmp_stdout" "$tmp_stderr"
        echo "no_optimum_found"
        return
    fi

    # Parse value (note sign handling from existing script)
    local got_val=$(echo "${optimum_line##* }" | sed 's/^-//; t; s/^/-/')

    # Parse iterations if available
    local iterations=$(grep -m1 '^Iteration' "$tmp_stdout" | awk '{print $2}' | tr -d ':' || true)

    # Check for NaN
    if echo "$got_val" | grep -qi "nan"; then
        rm -f "$tmp_stdout" "$tmp_stderr"
        echo "nan"
        return
    fi

    # Validate with tolerance (relative tolerance 1e-4)
    if awk -v a="$got_val" -v b="$expected_val" 'BEGIN {
        if (tolower(a) ~ /nan/) { exit 1 }
        diff = (a > b) ? a - b : b - a
        abs_b = (b > 0) ? b : -b
        exit !(diff <= 1e-4 * ((abs_b > 1) ? abs_b : 1))
    }'; then
        rm -f "$tmp_stdout" "$tmp_stderr"
        # Return result with data: correct|time|optimum|iterations
        echo "correct|${real_time}|${got_val}|${iterations}"
    else
        rm -f "$tmp_stdout" "$tmp_stderr"
        echo "wrong_value:got=${got_val}_expected=${expected_val}"
    fi
}

# ------------------------------------------------------------------------------
# Benchmark function - runs N times and records to TSV
# First writes validation run data (run_id=0), then runs additional benchmarks
# ------------------------------------------------------------------------------
run_benchmark() {
    local solver_bin="$1"
    local problem_file="$2"
    local output_tsv="$3"
    local num_extra_runs="$4"
    local solver_name="$5"
    local problem_name="$6"
    local m="$7"
    local n="$8"
    local validation_time="$9"
    local validation_optimum="${10}"
    local validation_iterations="${11}"

    # Write header
    echo -e "solver\tproblem\tm\tn\toptimum\titerations\ttime_sec\trun_id" > "$output_tsv"

    # Write validation run as run_id=0
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n" \
        "$solver_name" "$problem_name" "$m" "$n" \
        "$validation_optimum" "$validation_iterations" "$validation_time" 0 >> "$output_tsv"

    # Run additional benchmarks (run_id=1 through num_extra_runs)
    for ((run_id=1; run_id<=num_extra_runs; run_id++)); do
        local tmp_stdout=$(mktemp)

        # Time the execution (no timeout here since validation passed)
        local start_time=$(date +%s.%N)
        "$solver_bin" < "$problem_file" > "$tmp_stdout" 2>/dev/null
        local end_time=$(date +%s.%N)
        local real_time=$(awk "BEGIN {printf \"%.6f\", $end_time - $start_time}")

        # Parse optimum
        local optimum_line=$(grep '^Optimum found:' "$tmp_stdout" || true)
        local optimum=$(echo "${optimum_line##* }" | sed 's/^-//; t; s/^/-/')

        # Parse iterations if available
        local iterations=$(grep -m1 '^Iteration' "$tmp_stdout" | awk '{print $2}' | tr -d ':' || true)

        # Write row
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\n" \
            "$solver_name" "$problem_name" "$m" "$n" \
            "$optimum" "$iterations" "$real_time" "$run_id" >> "$output_tsv"

        rm -f "$tmp_stdout"
    done
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

echo "========================================"
echo "Benchmark with Validation"
echo "========================================"
echo "Max nonzeros:  $MAX_NONZEROS"
echo "Extra runs:    $EXTRA_RUNS"
echo "Timeout:       ${TIMEOUT_SEC}s"
echo "Results dir:   $RESULTS_DIR"
echo "========================================"
echo ""

# Check problem summary exists
if [ ! -f "$PROBLEM_SUMMARY" ]; then
    echo "ERROR: Problem summary not found: $PROBLEM_SUMMARY" >&2
    exit 1
fi

# Initialize output directory
mkdir -p "$RESULTS_DIR"

# Initialize failures file
cat > "$FAILURES_FILE" << EOF
# Benchmark Failures Report
# Generated: $(date)
# Max nonzeros: $MAX_NONZEROS
# Timeout: ${TIMEOUT_SEC}s
# Format: solver: <name>, problem: <problem>, issue: <description>

EOF

# Count total problems
total_problems=$(tail -n +2 "$PROBLEM_SUMMARY" | awk -F',' -v max="$MAX_NONZEROS" '$4 < max' | wc -l)

for solver_name in "${SOLVERS[@]}"; do
    solver_bin="bin_solver/${solver_name}.out"

    # Check if solver exists
    if [ ! -x "$solver_bin" ]; then
        echo "WARNING: Solver not found or not executable: $solver_bin" >&2
        echo "solver: ${solver_name}, problem: ALL, issue: solver_binary_not_found" >> "$FAILURES_FILE"
        continue
    fi

    echo ""
    echo "=== Solver: $solver_name ==="

    # Create solver results directory
    solver_results_dir="${RESULTS_DIR}/${solver_name}"
    mkdir -p "$solver_results_dir"

    # Initialize counters
    solver_correct[$solver_name]=0
    solver_failed[$solver_name]=0

    problem_idx=0

    while IFS=',' read -r PROBLEM rows cols nonzeros bytes br gt_val; do
        problem_idx=$((problem_idx + 1))
        problem="${PROBLEM,,}"  # lowercase
        problem_file="${PREPROCESSED_DIR}/${problem}.preprocessed"
        output_tsv="${solver_results_dir}/${problem}.tsv"

        # Skip if already benchmarked (unless force flag is set)
        if [ -f "$output_tsv" ] && [ "$FORCE_RERUN" != "force" ]; then
            echo "  [$problem_idx/$total_problems] $problem... SKIP (already benchmarked)"
            # Count as correct since it was previously successful
            solver_correct[$solver_name]=$((solver_correct[$solver_name] + 1))
            continue
        fi

        # Check if preprocessed file exists
        if [ ! -f "$problem_file" ]; then
            echo "  [$problem_idx/$total_problems] $problem... SKIP (preprocessed file not found)"
            echo "solver: ${solver_name}, problem: ${problem}, issue: preprocessed_file_not_generated" >> "$FAILURES_FILE"
            solver_failed[$solver_name]=$((solver_failed[$solver_name] + 1))
            continue
        fi

        # Step 1: Validation run
        echo -n "  [$problem_idx/$total_problems] $problem... "

        result=$(validate_single_run "$solver_bin" "$problem_file" "$gt_val")

        # Parse result - format is "correct|time|optimum|iterations" or error string
        result_status="${result%%|*}"

        if [ "$result_status" == "correct" ]; then
            # Parse validation data: correct|time|optimum|iterations
            IFS='|' read -r _ val_time val_optimum val_iterations <<< "$result"

            echo "OK, benchmarking ${EXTRA_RUNS}x more..."

            # Step 2: Run benchmark (includes validation run as run_id=0)
            run_benchmark "$solver_bin" "$problem_file" \
                "$output_tsv" \
                "$EXTRA_RUNS" "$solver_name" "$problem" "$rows" "$cols" \
                "$val_time" "$val_optimum" "$val_iterations"

            solver_correct[$solver_name]=$((solver_correct[$solver_name] + 1))
        else
            echo "FAILED ($result)"

            # Log failure
            echo "solver: ${solver_name}, problem: ${problem}, issue: ${result}" >> "$FAILURES_FILE"

            solver_failed[$solver_name]=$((solver_failed[$solver_name] + 1))
        fi

    done < <(tail -n +2 "$PROBLEM_SUMMARY" | awk -F',' -v max="$MAX_NONZEROS" '$4 < max')

    echo "  Summary: ${solver_correct[$solver_name]} correct, ${solver_failed[$solver_name]} failed"
done

# ------------------------------------------------------------------------------
# Generate summary
# ------------------------------------------------------------------------------

echo ""
echo "========================================"
echo "FINAL SUMMARY"
echo "========================================"

cat > "$SUMMARY_FILE" << EOF
# Benchmark Summary
# Generated: $(date)
# Max nonzeros: $MAX_NONZEROS
# Extra runs per correct problem: $EXTRA_RUNS
# Timeout: ${TIMEOUT_SEC}s

## Results by Solver

EOF

total_correct=0
total_failed=0

for solver_name in "${SOLVERS[@]}"; do
    c=${solver_correct[$solver_name]:-0}
    f=${solver_failed[$solver_name]:-0}
    total_correct=$((total_correct + c))
    total_failed=$((total_failed + f))

    echo "$solver_name: $c correct, $f failed"
    echo "$solver_name: $c correct, $f failed" >> "$SUMMARY_FILE"
done

echo ""
echo "Total: $total_correct correct, $total_failed failed"
echo "" >> "$SUMMARY_FILE"
echo "Total: $total_correct correct, $total_failed failed" >> "$SUMMARY_FILE"

echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Failures logged to: $FAILURES_FILE"
echo "Summary: $SUMMARY_FILE"
echo "========================================"
