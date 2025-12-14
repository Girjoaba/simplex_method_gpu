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
# IMPORTANT
# gurobi's presolve discards the old objective constant term
offset = model.ObjCon
if model.ModelSense == gp.GRB.MINIMIZE:
    offset *= -1.0
model = model.presolve()

if (set(model.getAttr("VType")) != { 'C' }):
    sys.exit('Model is not continuous')

with open("input/stats.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    row = next((r for r in reader if r[0].lower() == file), None)

if row is None:
    sys.exit('Failed to find the optimum in stats.csv')

optimum = float(row[6])

# =================== Step 2 ===================

# convert the problem to numpy arrays
# flip the cost if it is a minimisation

m, n = model.NumConstrs, model.NumVars
constraints = model.getConstrs()
vars = model.getVars()

A = np.array(model.getA().todense())
b = np.array([c.RHS for c in constraints])
c = np.array([v.Obj for v in vars])
senses = np.array([c.Sense for c in constraints])

offset += model.ObjCon
print(f"Initial offset: {offset}")
print(f"Objective scale: {model.params.ObjScale}")


if model.ModelSense == gp.GRB.MINIMIZE:
    print("Converting minimisation to maximisation problem.")
    offset *= -1.0
    c *= -1.0
    optimum *= -1.0

# =================== Step 3 ===================

# modify the problem appropriately so that x >= 0 for all variables
# for now, we do not care about restoring the original variables

INF = 1e20
fixed_indices = []
free_indices = []

new_rows_A = []
new_rows_b = []
new_rows_senses = []

for i, var in enumerate(vars):
    if var.LB == var.UB:
        b -= A[:, i] * var.LB
        offset += float(c[i] * var.LB)
        fixed_indices.append(i)
        continue
    if var.LB < -INF and INF < var.UB:
        free_indices.append(i)
        continue
    if var.UB < INF:
        if var.LB < -INF:
            b -= A[:, i] * var.UB
            A[:, i] *= -1.0
            offset += float(c[i] * var.UB)
            c[i] *= -1
        else:
            b -= A[:, i] * var.LB
            offset += float(c[i] * var.LB)
            
            new_row = np.zeros(n, dtype=A.dtype)
            new_row[i] = 1.0
            new_rows_A.append(new_row)
            new_rows_b.append(var.UB - var.LB)
            new_rows_senses.append('<')
        continue
    b -= A[:, i] * var.LB
    offset += float(c[i] * var.LB)

A = np.hstack([A, -A[:, free_indices]])
c = np.hstack([c, -c[free_indices]])

if new_rows_A:
    new_A = np.array(new_rows_A)

    if len(free_indices) > 0:
        zeros_padding = np.zeros((len(new_rows_A), len(free_indices)))
        new_A = np.hstack([new_A, zeros_padding])

    A = np.vstack([A, new_A])
    b = np.concatenate((b, np.array(new_rows_b)))
    senses = np.concatenate((senses, np.array(new_rows_senses)))
    
    m = len(b)

A = np.delete(A, fixed_indices, axis=1)
c = np.delete(c, fixed_indices)

n += len(free_indices) - len(fixed_indices)

assert n == A.shape[1]

# =================== Step 4 ===================

# flip negative constraints

flip = b < 0
A[flip] *= -1.0
b[flip] *= -1.0
ge = (senses == '>')
le = (senses == '<')
senses[flip & ge] = '<'
senses[flip & le] = '>'

# =================== Step 5 ===================

# make slack rows the first rows of A so that in phase II they can be easily discarded
# append surplus columns to A and c
# append the identity and finalise costs

is_slack = (senses == '<')
A = np.vstack((A[is_slack], A[~is_slack]))
b = np.concatenate((b[is_slack], b[~is_slack]))
senses = np.concatenate((senses[is_slack], senses[~is_slack]))
n_slack = is_slack.sum()

is_surplus = (senses == '>')
n_surplus = is_surplus.sum()
surplus = np.zeros((m, n_surplus), dtype=A.dtype)
surplus[is_surplus, np.arange(n_surplus)] = -1.0
A = np.hstack((A, surplus))
c = np.concatenate((c, np.zeros(n_surplus, dtype=c.dtype)))

A = np.hstack((A, np.identity(m, dtype=A.dtype)))
c = np.concatenate((c, np.zeros(n_slack, dtype=c.dtype)))

assert A.shape == (m, n + n_surplus + m)
assert b.shape == (m,)
assert c.shape == (n + n_surplus + n_slack,)

# =================== Step 6 ===================

print(f"{n_surplus} surpluses, {n_slack} slacks, {m - n_slack} artificials")

with open('input/problems/' + file + '.txt', "w") as f:
    f.write(f"{m} {n} {n_surplus} {n_slack}\n")
    f.write(f"{repr(float(optimum))} {repr(offset)}\n")
    for i in range(m):
        f.write(" ".join(repr(float(x)) for x in A[i, :]) + '\n')
    f.write(" ".join(repr(float(x)) for x in b) + '\n')
    f.write(" ".join(repr(float(x)) for x in c) + '\n')
