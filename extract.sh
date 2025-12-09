#!/bin/bash

files=("$1")

if [ -z "${files[0]}" ]; then
	files=(
		"adlittle" 
		"afiro"
		"agg"
		"bandm"
		"beaconfd"
		"blend"
		"boeing1"
		"boeing2"
		"bore3d"
		"brandy"
		"capri"
		"e226" 
		"etamacro"
		"finnis"
		"gfrd-pnc"
		"grow7"
		"israel" 
		"kb2"
		"lotfi"
		"recipe" 
		"sc105" 
		"sc205"
		"sc50a" 
		"sc50b"
		"scagr25"
		"scagr7" 
		"scfxm1"
		"scorpion"
		"scrs8"
		"scsd1"
		"sctap1"
		"share1b"
		"share2b"
		"stair"
		"standata"
		"standmps"
		"stocfor1"
		"vtp.base"
	)
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