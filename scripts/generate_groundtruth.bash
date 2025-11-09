#!/usr/bin/env bash
set -euo pipefail

echo "[build] running make..."
make bin_glpk/glpk_solver

PROBLEMS_DIR="./problems"
GLPK_BIN="./bin_glpk/glpk_solver"
OUT_DIR="./test/groundtruth"

mkdir -p "$OUT_DIR"

attempted=0
errors=0

# first 100 smallest files
mapfile -t files < <(
  ls -lSr "$PROBLEMS_DIR" \
    | tail -n +2 \
    | head -n 100 \
    | awk '{for (i=9; i<=NF; i++) {printf "%s%s", $i, (i==NF?ORS:OFS)}}'
)

for name in "${files[@]}"; do
    attempted=$((attempted + 1))

    in_file="$PROBLEMS_DIR/$name"
    out_file="$OUT_DIR/$name.txt"

    echo "[run] $in_file -> $out_file"

    tmp_out="$(mktemp)"

    if ! "$GLPK_BIN" "$in_file" >"$tmp_out" 2>"$tmp_out.err"; then
        echo "[skip] solver failed on $in_file, ignoring."
        rm -f "$tmp_out" "$tmp_out.err" "$out_file"
        errors=$((errors + 1))
        continue
    fi

    # grab the line, then print only the last field (the number)
    if grep -q '^Optimal objective:' "$tmp_out"; then
        grep '^Optimal objective:' "$tmp_out" | awk '{print $NF}' > "$out_file"
    else
        echo "[skip] no 'Optimal objective:' in output for $in_file"
        rm -f "$out_file"
        errors=$((errors + 1))
    fi

    rm -f "$tmp_out" "$tmp_out.err"
done

generated=$((attempted - errors))

echo "===================================="
echo "Attempted to generate : $attempted"
echo "Successfully generated: $generated"
echo "Errors (skipped)      : $errors"
echo "Output directory      : $OUT_DIR"
