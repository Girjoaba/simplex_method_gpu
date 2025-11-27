#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_DIR="./test/input"       # Where your .canonical files are
GROUNDTRUTH_DIR="./test/groundtruth" # Where the GT files are
BIN_DIR="./bin_solver"         # Where make puts the binaries
TEMP_DIR=$(mktemp -d)          # Temporary dir for caching GT outputs

# Cleanup temp files on exit
trap "rm -rf $TEMP_DIR" EXIT

# ==============================================================================
# PRE-FLIGHT CHECKS
# ==============================================================================

# 1. Compile everything to ensure binaries are fresh
echo "[build] Running make to ensure binaries are up to date..."
make -j$(nproc) > /dev/null

# 2. Find the GT solver
#    Logic: Look for the binary starting with 'v1_' (e.g., v1_cpu.out or v1_naive.out)
#    We assume v1 is always the ground truth.
GT_BIN=$(find "$BIN_DIR" -name "v[0-9]*.out" | sort -V | head -n 1)

if [ -z "$GT_BIN" ]; then
    # Fallback for legacy naming if v1_*.out not found
    if [ -f "$BIN_DIR/solver1.out" ]; then
        GT_BIN="$BIN_DIR/solver1.out"
    else
        echo "Error: Ground truth solver (v1_*.out) not found in $BIN_DIR."
        exit 1
    fi
fi

GT_SOLVER_NAME=$(basename "$GT_BIN")

# 3. Find all candidate solvers
#    Logic: Find all files matching v*.out (or solver*.out for legacy) and sort version-aware
SOLVERS=($(find "$BIN_DIR" -name "v*.out" -o -name "solver*.out" | sort -V))

if [ ${#SOLVERS[@]} -eq 0 ]; then
    echo "Error: No solvers found in $BIN_DIR"
    exit 1
fi

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

get_optimum() {
    local file="$1"
    grep '^Optimum found:' "$file" | tail -n 1 | awk '{print $3}' || true
}

check_tolerance() {
    local val1="$1"
    local val2="$2"
    
    # 1. Handle exact string matches
    if [ "$val1" == "$val2" ]; then return 0; fi

    # 2. Check for inf/nan mismatch explicitly
    if [[ "$val1" == *"inf"* || "$val2" == *"inf"* || "$val1" == *"nan"* || "$val2" == *"nan"* ]]; then
        return 1
    fi

    # 3. Floating point check with awk
    awk -v a="$val1" -v b="$val2" 'BEGIN {
        diff = a - b; if (diff < 0) diff = -diff;
        absb = b; if (absb < 0) absb = -absb;
        tol = 1e-5;
        if (absb > 1) tol = tol * absb;
        if (diff <= tol) exit 0; else exit 1;
    }'
}

# ==============================================================================
# EXECUTION
# ==============================================================================

echo "============================================================"
echo " CROSS VALIDATION SUITE"
echo " Ground Truth Solver: $GT_SOLVER_NAME"
echo " Test Set:            Files in $GROUNDTRUTH_DIR"
echo " Solvers Found:       ${#SOLVERS[@]}"
echo "============================================================"

# --- PHASE 1: GENERATE GROUND TRUTH ---
echo ""
echo "[Phase 1] Running Ground Truth ($GT_SOLVER_NAME)..."

gt_start=$(date +%s.%N)
gt_attempted=0
gt_crashes=0
missing_inputs=0

# Iterate over GT files to define the test set
for gt_path in "$GROUNDTRUTH_DIR"/*.txt; do
    [ -e "$gt_path" ] || continue
    
    gt_filename=$(basename "$gt_path")          # e.g., adlittle.mps.txt
    problem_root="${gt_filename%.mps.txt}"      # e.g., adlittle
    canonical_path="$INPUT_DIR/$problem_root.canonical"
    
    # Skip if corresponding input doesn't exist
    if [ ! -f "$canonical_path" ]; then
        missing_inputs=$((missing_inputs + 1))
        continue
    fi
    
    gt_cache="$TEMP_DIR/$problem_root.gt"
    
    # Run Solver 1 (Capture ALL output to silence "Problem unbounded")
    if ! "$GT_BIN" < "$canonical_path" > "$gt_cache" 2>&1; then
        echo "  [GT] 💥 Crashed on $problem_root"
        rm -f "$gt_cache"
        gt_crashes=$((gt_crashes + 1))
        continue
    fi
    
    val=$(get_optimum "$gt_cache")
    if [ -z "$val" ]; then
        echo "  [GT] ❌ No output on $problem_root"
        rm -f "$gt_cache"
        gt_crashes=$((gt_crashes + 1))
        continue
    fi
    
    gt_attempted=$((gt_attempted + 1))
done

gt_end=$(date +%s.%N)
gt_duration=$(echo "$gt_end $gt_start" | awk '{print $1 - $2}')

echo "  -> Processed $gt_attempted files in ${gt_duration}s"
if [ "$gt_crashes" -gt 0 ]; then
    echo "  -> WARNING: GT solver failed on $gt_crashes files. These will be skipped."
fi


# --- PHASE 2: CROSS VALIDATE OTHERS ---

# Adjust formatting width based on longest solver name
max_name_len=20
for solver in "${SOLVERS[@]}"; do
    len=${#solver}
    if [ $len -gt $max_name_len ]; then max_name_len=$len; fi
done
# Ensure a reasonable cap
if [ $max_name_len -gt 40 ]; then max_name_len=40; fi

printf "\n%-${max_name_len}s | %-10s | %-10s | %-10s | %-15s\n" "Solver" "Status" "Correct" "Errors" "Time (Total)"
printf "%s\n" "$(printf '%*s' $((max_name_len + 56)) '' | tr ' ' '-')"
printf "%-${max_name_len}s | %-10s | %-10s | %-10s | %-15s\n" "$GT_SOLVER_NAME (GT)" "REF" "-" "-" "${gt_duration}s"

for solver in "${SOLVERS[@]}"; do
    solver_name=$(basename "$solver")
    
    if [ "$solver_name" == "$GT_SOLVER_NAME" ]; then continue; fi
    
    correct=0
    errors=0
    crashes=0
    
    start_time=$(date +%s.%N)
    
    # Iterate over the same GT set
    for gt_path in "$GROUNDTRUTH_DIR"/*.txt; do
        [ -e "$gt_path" ] || continue

        gt_filename=$(basename "$gt_path")
        problem_root="${gt_filename%.mps.txt}"
        canonical_path="$INPUT_DIR/$problem_root.canonical"
        gt_cache="$TEMP_DIR/$problem_root.gt"
        
        # Skip if input missing or GT failed earlier
        if [ ! -f "$canonical_path" ] || [ ! -f "$gt_cache" ]; then
            continue
        fi
        
        # Run Candidate
        output_tmp="$TEMP_DIR/$solver_name.out"
        if ! "$solver" < "$canonical_path" > "$output_tmp" 2>&1; then
            crashes=$((crashes + 1))
            continue
        fi
        
        val_candidate=$(get_optimum "$output_tmp")
        val_gt=$(get_optimum "$gt_cache")
        
        if [ -z "$val_candidate" ]; then
            crashes=$((crashes + 1))
        else
            if check_tolerance "$val_candidate" "$val_gt"; then
                correct=$((correct + 1))
            else
                errors=$((errors + 1))
            fi
        fi
    done
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time $start_time" | awk '{print $1 - $2}')
    
    if [ "$errors" -eq 0 ] && [ "$crashes" -eq 0 ]; then
        status="PASS ✅"
    elif [ "$crashes" -gt 0 ]; then
        status="CRASH 💥"
    else
        status="FAIL ❌"
    fi
    
    printf "%-${max_name_len}s | %-10s | %-10s | %-10s | %-15s\n" "$solver_name" "$status" "$correct/$gt_attempted" "$errors" "${duration}s"
    
done

echo "-----------------------------------------------------------------------------"
echo "Done."