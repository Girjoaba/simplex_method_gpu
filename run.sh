#!/bin/bash

srun -t 1 -A dphpc ./bin/solver.out < input/${1}.txt