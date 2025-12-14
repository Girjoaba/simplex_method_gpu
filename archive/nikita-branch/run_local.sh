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

for file in "${files[@]}"; do
  echo "Running solver on input file: ${file}"
  ./bin/solver.out < input/problems/${file}.txt
done