#!/bin/bash

files=("$1")

if [ -z "${files[0]}" ]; then
  files=("adlittle" "afiro" "e226" "israel" "lotfi" "scagr7" "sc105" "sc205" "sc50a" "sc50b" "share2b")
fi
BASE_URL="https://www.netlib.org/lp/data"
NETLIB_DIR="input/netlib"

if [ ! -f ${NETLIB_DIR}/emps.out ]; then
  gcc -o ${NETLIB_DIR}/emps.out ${NETLIB_DIR}/emps.c
fi

for file in "${files[@]}"; do
  echo "Processing file: ${file}"

	if [ ! -f "${NETLIB_DIR}/compressed/${file}" ]; then
  	wget -P "${NETLIB_DIR}/compressed" "${BASE_URL}/${file}"
	fi

	if [ ! -f "${NETLIB_DIR}/mps/${file}.mps" ]; then
  	./"${NETLIB_DIR}/emps.out" < "${NETLIB_DIR}/compressed/${file}" \
		> "${NETLIB_DIR}/mps/${file}.mps"
	fi

	if [ ! -f "input/${file}.txt" ]; then
  	./input/convert.py "${file}"
	fi
	
done