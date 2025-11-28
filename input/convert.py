#!/usr/bin/env python3

# input: the name of a problem passed as an argument
#   - the program exits if the problem is not continuous or the var bounds are non-default
# output: the problem in canonical form with artificial vars saved to a txt file

import argparse
import csv
import gurobipy as gp
import numpy as np
import sys
from gurobipy import GRB

LESS = 0
GREATER = 1
EQUAL = 2

parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()
file = args.file

model = gp.read('input/netlib/mps/' + file + '.mps')
if (set(model.getAttr("VType")) != { 'C' }):
    sys.exit('Model is not continuous')

vars = model.getVars()
num_non_default_vars = sum(
    not np.isclose(v.LB, 0.0) or not np.isinf(v.UB)
    for v in vars
)
if num_non_default_vars != 0:
    sys.exit(f'{num_non_default_vars} non-default variables detected')

optimum = 0
with open("input/stats.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        if file == row[0].lower():
            optimum = float(row[6])
if optimum == 0:
    sys.exit('Failed to find the optimum in stats.csv')

m, n = model.NumConstrs, model.NumVars
constrs = model.getConstrs()
A = np.array(model.getA().todense())
b = np.array([c.RHS for c in constrs], dtype=A.dtype)
c = np.array([v.Obj for v in vars], dtype=A.dtype)
senses = np.array(
    [
        EQUAL if c.Sense == GRB.EQUAL
        else GREATER if c.Sense == GRB.GREATER_EQUAL
        else LESS
        for c in constrs
    ],
    dtype=int
)

# multiply a constraint by -1 if the RHS is negative
flip_mask = b < 0
A[flip_mask] *= -1
b[flip_mask] *= -1
non_equal_mask = senses != EQUAL
senses[flip_mask & non_equal_mask] = 1 - senses[flip_mask & non_equal_mask]

# append surplus columns to A
surplus_ixs = np.where(senses == GREATER)[0]
num_surplus = surplus_ixs.size
surplus = np.zeros((m, num_surplus), dtype=A.dtype)
surplus[surplus_ixs, np.arange(num_surplus)] = -1
A = np.hstack((A, surplus))

# make slack rows the first rows of A so that
# in phase II we can easily discard columns
slack_ixs = np.where(senses == LESS)[0]
num_slack = slack_ixs.size
rest_ixs = np.setdiff1d(np.arange(m), slack_ixs, assume_unique=True)
order = np.concatenate((slack_ixs, rest_ixs))
A = A[order]
b = b[order]

# set zero costs in c for surplus and slack vars
# A = np.hstack((A, np.identity(m, dtype=A.dtype)))
c = np.concatenate((c, np.zeros(num_surplus + num_slack, dtype=c.dtype)))

# assert A.shape == (m, n + num_surplus + m)
assert A.shape == (m, n + num_surplus)
assert b.shape == (m,)
assert c.shape == (n + num_surplus + num_slack,)

print(f"{num_surplus} surpluses, {num_slack} slacks, {m - num_slack} artificials")
if model.ModelSense == gp.GRB.MINIMIZE:
    print("Converting minimization to maximization problem.")
    c = -c

with open('input/' + file + '.txt', "w") as f:
    f.write(f"{m} {n} {num_surplus} {num_slack}\n")
    f.write(f"{optimum:.10e}\n")
    for i in range(m):
        f.write(" ".join(map(str, A[i, :])) + '\n')
    f.write(" ".join(map(str, b)) + '\n')
    f.write(" ".join(map(str, c)) + '\n')

# for i, constr in enumerate(constrs):
#     print(f"{i}: {model.getRow(constr)} {constr.Sense} {constr.RHS}")