#!/bin/bash

set -euo pipefail

target_problem=""

if [ $# -lt 1 ]; then
	echo "Usage: $0 <path_to_solver_binary> <max_number_of_nonzeros>"
	echo "152K non-zeros will cover all the problems"
	exit 1
fi

solver_bin="$1"
problem_summary="scripts/netlib/problem_summary.csv"

if [ ! -f "$problem_summary" ]; then
	echo "The problem summary table does not exist: $problem_summary" >&2
	exit 1
fi

if [ ! -x "$solver_bin" ]; then
	echo "The binary does not exist or is not executable: $solver_bin" >&2
	exit 1
fi

correct=0
wrong=0
errors=0
missing=0
total_time=0

while IFS=',' read -r UPPER_CASE_PROBLEM _ _ nonzeros _ _ gt_val; do

	problem="${UPPER_CASE_PROBLEM,,}"
	preprocessed_file="test/netlib/preprocessed/${problem}.preprocessed"

	if [[ -n "${target_problem}" ]]; then
		[[ "$problem" == "$target_problem" ]] || continue
	fi

	if [ ! -f "$preprocessed_file" ]; then
		printf "[missing] %-25s %-4s %s %-60s\n" \
			"$problem" "MISS" "🗂" "(${preprocessed_file} not found)" >&2
		missing=$((missing + 1))
		continue
	fi

	tmp_stdout=$(mktemp --suffix=.stdout.tmp)
	tmp_stderr=$(mktemp --suffix=.stderr.tmp)
	tmp_time=$(mktemp --suffix=.time.tmp)

	exit_code=0
	{ time -p "$solver_bin" < "$preprocessed_file" > "$tmp_stdout" 2>> "$tmp_stderr" ; } 2>> "$tmp_time" || exit_code=$?

	real_time=$(grep "real" "$tmp_time" | awk '{print $2}')

	if [ "$exit_code" -ne 0 ] ; then
		printf "[error]   %-25s %-4s %s %-60s Time: %ss\n" \
			"$problem" "ERR" "❌" "(solver exited with an error)" "$real_time" >&2
		cat "$tmp_stderr" >&2
		rm -rf "$tmp_stdout" "$tmp_stderr" "$tmp_time"
		errors=$((errors + 1))
		continue
	fi

	optimum_line=$(grep '^Optimum found:' "$tmp_stdout" || true)

	if [ -z "$optimum_line" ]; then
		printf "[compare] %-25s %-4s %s %-60s Time: %ss\n" \
			"$problem" "WARN" "⚠️" "(experiment missing 'Optimum found:')" "$real_time"
		cat "$tmp_stdout"
		rm -rf "$tmp_stdout" "$tmp_stderr" "$tmp_time"
		wrong=$((wrong + 1))
		continue
	fi

	exp_val=$(echo "${optimum_line##* }" | sed 's/^-//; t; s/^/-/')

	if awk -v a="$exp_val" -v b="$gt_val" 'BEGIN {
	if (tolower(a) ~ /nan/) { exit 1 }
	diff = (a > b) ? a - b : b - a
		abs_b = (b > 0) ? b : -b
		exit !(diff <= 1e-4 * ((abs_b > 1) ? abs_b : 1))
	}'; then
		status_msg="OK"
		emoji="✅"
		correct=$((correct + 1))
		total_time=$(awk "BEGIN {print $total_time + $real_time}")
	else
		status_msg="WARN"
		emoji="⚠️"
		wrong=$((wrong + 1))
	fi

	detail_msg="(got=$exp_val, expected=$gt_val)"
	printf "[compare] %-25s %-4s %s %-60s Time: %ss\n" \
		"$problem" "$status_msg" "$emoji" "$detail_msg" "$real_time"

	rm -rf "$tmp_stdout" "$tmp_stderr" "$tmp_time"

done < <(tail -n +2 "$problem_summary" | awk -F',' -v max="$2" '$4 < max')

echo "===================================="
echo "Correct ✅                : $correct"
echo "Wrong   ⚠️                : $wrong"
echo "Errors  ❌                : $errors"
echo "Missing 🗂                : $missing"
echo "Time    ⏰                : ${total_time}s"
echo "===================================="
