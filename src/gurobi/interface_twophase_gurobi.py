#!/usr/bin/env python3

# IN:  the name of a problem passed as an argument
# OUT: the problem in canonical form with artificial vars

import argparse, csv, sys
import gurobipy as gp
import numpy as np

# =================== Step 1 ===================

# read the file
# check that the problem is continuous
# get its optimum

parser = argparse.ArgumentParser()
parser.add_argument('file')
file = parser.parse_args().file

model = gp.read('input/netlib/mps/' + file + '.mps')

if (set(model.getAttr("VType")) != {'C'}):
    sys.exit('Model is not continuous')

# =================== Step 2 ===================

# convert the problem to numpy arrays
# flip the cost if it a minimisation
# flip negative constraints

m, n = model.NumConstrs, model.NumVars
constraints = model.getConstrs()
vars = model.getVars()

A = np.array(model.getA().todense())
b = np.array([c.RHS for c in constraints])
c = np.array([v.Obj for v in vars])
senses = np.array([c.Sense for c in constraints])

if model.ModelSense == gp.GRB.MINIMIZE:
    # print("Converting minimisation to maximisation problem.")
    c *= -1

    # This might cause issues!
    # optimum *= -1

# flip a constraint if the RHS is negative
flip = b < 0
A[flip] *= -1
b[flip] *= -1
ge = (senses == '>')
le = (senses == '<')
senses[flip & ge] = '<'
senses[flip & le] = '>'

# =================== Step 3 ===================

# modify the problem appropriately so that x >= 0 for all variables
# this step assumes that the order of vars matches the order of columns
# learn the theory before attempting to understand this step
# for now, we do not care about restoring the original variables

INF = 1e18
# offset = 0.0
fixed_indices = []
free_indices = []

# for i, var in enumerate(vars):
#     if var.LB == var.UB:
#         b -= A[:, i] * var.LB
#         offset += var.Obj * var.LB
#         fixed_indices.append(i)
#         continue
#     if var.LB < -INF and INF < var.UB:
#         free_indices.append(i)
#         continue
#     if var.UB < INF:
#         if var.LB == 0.0 or var.LB < -INF:
#             # y = UB - x  ->  x = UB - y
#             b -= A[:, i] * var.UB
#             A[:, i] *= -1
#             offset += var.Obj * var.UB
#             c[i] *= -1
#         else:
#             # y = x - LB  ->  x = y + LB
#             # x < UP  ->  y + LB < UP  ->  y < UB - LB
#             # y' = (UB - LB) - y  ->  y = (UP - LB) - y'
#             b -= A[:, i] * var.UB
#             offset += var.Obj * (var.UB - 2 * var.LB)
#             A[:, i] *= -1
#             c[i] *= -1
#         continue
#     # y = x - LB  ->  x = y + LB
#     b -= A[:, i] * var.LB
#     offset += var.Obj * var.LB

# A = np.hstack([A, -A[:, free_indices]])
# c = np.hstack([c, -c[free_indices]])

# A = np.delete(A, fixed_indices, axis=1)
# c = np.delete(c, fixed_indices)

# =================== Step 4 ===================

# make slack rows the first rows of A
# so that in phase II they can be easily discarded
# append surplus columns to A and c
# append the identity and finalise costs

# make slack rows the first rows of A
is_slack = (senses == '<')
A = np.vstack((A[is_slack], A[~is_slack]))
b = np.concatenate((b[is_slack], b[~is_slack]))
senses = np.concatenate((senses[is_slack], senses[~is_slack]))
n_slack = is_slack.sum()

# append surplus columns to A and c
is_surplus = (senses == '>')
n_surplus = is_surplus.sum()
surplus = np.zeros((m, n_surplus), dtype=A.dtype)
surplus[is_surplus, np.arange(n_surplus)] = -1
A = np.hstack((A, surplus))
c = np.concatenate((c, np.zeros(n_surplus, dtype=c.dtype)))

# append the identity and finalise costs
A = np.hstack((A, np.identity(m, dtype=A.dtype)))
c = np.concatenate((c, np.zeros(n_slack, dtype=c.dtype)))

n_fixed = len(fixed_indices)
n_free = len(free_indices)
n += n_free - n_fixed

assert A.shape == (m, n + n_surplus + m)
assert b.shape == (m,)
assert c.shape == (n + n_surplus + n_slack,)

# =================== Step 5 ===================

# print(f"{n_surplus} surpluses, {n_slack} slacks, {m - n_slack} artificials")

with open('test/input/' + file + '.twophase', "w") as f:
    f.write(f"{m} {n} {n_surplus} {n_slack}\n")
    # f.write(f"{repr(float(optimum))} {repr(offset)}\n")
    for i in range(m):
        f.write(" ".join(repr(float(x)) for x in A[i, :]) + '\n')
    f.write(" ".join(repr(float(x)) for x in b) + '\n')
    f.write(" ".join(repr(float(x)) for x in c) + '\n')
