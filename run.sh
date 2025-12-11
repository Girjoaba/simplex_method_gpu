#!/bin/bash

#SBATCH --time=02:00
#SBATCH --account=dphpc
#SBATCH --output=output.txt

if [ $# -eq 0 ]; then
	source ./input/problem_list.sh

	for f in "${files[@]}"; do
		echo "  ${f^^}"
		
		./bin/solver.out < input/problems/${f}.txt
	done
else
	srun -t 1 -A dphpc ./bin/solver.out < input/problems/${1}.txt
fi