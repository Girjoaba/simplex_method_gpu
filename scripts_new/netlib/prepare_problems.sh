#!/bin/bash

set -euo pipefail

scripts_dir="scripts_new/netlib"
problem_summary="${scripts_dir}/problem_summary.csv"

if [ ! -f "$problem_summary" ]; then
	echo "The problem summary table does not exist"
	exit 1
fi

readarray -t problems < <(tail -n +2 "$problem_summary" | cut -d',' -f1)

netlib_url="https://www.netlib.org/lp/data"
expander_bin="bin/expand_problem.out"

if [ ! -f "$expander_bin" ]; then
  gcc "${scripts_dir}/expand_problem.c" -o "$expander_bin" -Wno-format-overflow
fi

test_dir="test/netlib"
compressed_dir="${test_dir}/compressed"
mps_dir="${test_dir}/mps"
preprocessed_dir="${test_dir}/preprocessed"

mkdir -p "$compressed_dir" "$mps_dir" "$preprocessed_dir"

for UPPER_CASE_PROBLEM in "${problems[@]}"; do

	problem="${UPPER_CASE_PROBLEM,,}"

  echo "Processing problem: ${problem}"

	compressed_file="${compressed_dir}/${problem}"
	mps_file="${mps_dir}/${problem}.mps"
	preprocessed_file="${preprocessed_dir}/${problem}.preprocessed"

	if [ ! -f "$compressed_file" ]; then
		wget -P "$compressed_dir" "${netlib_url}/${problem}"
	fi

	if [ ! -f "$mps_file" ]; then
  	"$expander_bin" < "$compressed_file" > "$mps_file"
	fi

	if [ ! -f "${preprocessed_file}" ]; then
  	./src/gurobi/interface.py "$mps_file" "$preprocessed_file"
	fi
	
done