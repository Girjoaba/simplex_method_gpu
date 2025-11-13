#!/usr/bin/env bash
set -euo pipefail

PYTHON="python"
INTERFACE="./src/gurobi/interface_gurobi.py"
SOLVER="./src/gurobi/solver_gurobi.py"
OUT_DIR="./test/groundtruth"

mkdir -p "$OUT_DIR"

attempted=0
errors=0

# All .mps files from ./test/input
PROBLEM_FILES=(./test/input/*.mps)

# If no .mps files exist, bail out nicely
if [ ${#PROBLEM_FILES[@]} -eq 0 ]; then
    echo "No .mps files found in ./test/input"
    exit 1
fi

for in_file in "${PROBLEM_FILES[@]}"; do
    attempted=$((attempted + 1))

    if [ ! -f "$in_file" ]; then
        echo "[skip] input file not found: $in_file"
        errors=$((errors + 1))
        continue
    fi

    # Output ground truth file (foo.mps -> foo.mps.txt)
    name="$(basename "$in_file")"
    out_file="$OUT_DIR/$name.txt"

    # Canonical file path in the SAME directory as the input:
    # ./test/input/foo.mps -> ./test/input/foo.canonical
    canon_file="${in_file%.*}.canonical"

    echo "[run] $in_file -> $canon_file -> $out_file"

    tmp_out="$(mktemp)"       # solver stdout
    tmp_err="$(mktemp)"       # solver stderr

    # 1) Interface: original MPS -> canonical LP file (saved and kept)
    if ! "$PYTHON" "$INTERFACE" "$in_file" "$canon_file" > /dev/null 2>&1; then
        echo "[skip] interface failed on $in_file, ignoring."
        rm -f "$tmp_out" "$tmp_err" "$out_file"
        errors=$((errors + 1))
        continue
    fi

    # 2) Solver: canonical LP -> optimal objective
    if ! "$PYTHON" "$SOLVER" "$canon_file" >"$tmp_out" 2>"$tmp_err"; then
        echo "[skip] solver failed on $in_file, ignoring."
        rm -f "$tmp_out" "$tmp_err" "$out_file"
        errors=$((errors + 1))
        continue
    fi

    # Extract the objective value from the solver output
    if grep -q '^Optimal objective:' "$tmp_out"; then
        grep '^Optimal objective:' "$tmp_out" | awk '{print $NF}' > "$out_file"
    else
        echo "[skip] no 'Optimal objective:' in output for $in_file"
        rm -f "$out_file"
        errors=$((errors + 1))
    fi

    rm -f "$tmp_out" "$tmp_err"
    # NOTE: we do NOT remove "$canon_file" on purpose
done

generated=$((attempted - errors))

echo "===================================="
echo "Attempted to generate : $attempted"
echo "Successfully generated: $generated"
echo "Errors (skipped)      : $errors"
echo "Output directory      : $OUT_DIR"
