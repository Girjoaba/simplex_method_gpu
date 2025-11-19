#!/usr/bin/env python3

import numpy as np
import scipy as sp
import gurobipy as gp
import argparse
import math

args = argparse.ArgumentParser(description="Convert MPS file to canonical form.")
args.add_argument(
    "mps_file_path",
    type=str,
    help="Path to the MPS file to be converted.",
)
args.add_argument(
    "output_file",
    type=str,
    help="Path to save the converted canonical form.",
)


def convert_mps_to_canonical(mps_file_path, output_file):
    """Convert MPS file to canonical form."""
    problem = gp.read(mps_file_path)

    for var in problem.getVars():
        if var.VType != gp.GRB.CONTINUOUS:
            raise ValueError("The MPS file contains non-continuous variables.")
        if var.LB != 0:
            # add a slack variable to convert to zero lower bound
            problem.addConstr(var >= var.LB, name=f"lb_{var.VarName}")
        if var.UB != gp.GRB.INFINITY and var.UB != math.inf:
            problem.addConstr(var <= var.UB, name=f"ub_{var.VarName}")
    problem.update()

    A = problem.getA().todense()
    constraints = problem.getConstrs()

    slack_vals = [0] * problem.NumConstrs
    columns_to_add = 0
    for i in range(problem.NumConstrs):
        constr = constraints[i]
        if constr.Sense == gp.GRB.LESS_EQUAL:
            slack_vals[i] = 1
            columns_to_add += 1
        elif constr.Sense == gp.GRB.GREATER_EQUAL:
            slack_vals[i] = -1
            columns_to_add += 1

    b = np.array([constraints[i].RHS for i in range(problem.NumConstrs)])

    if columns_to_add > 0:
        slack_matrix = np.zeros((problem.NumConstrs, columns_to_add))
        slack_col_index = 0
        for i in range(problem.NumConstrs):
            if slack_vals[i] == 0:
                continue
            slack_matrix[i, slack_col_index] = slack_vals[i]
            slack_col_index += 1

        A = np.hstack((A, slack_matrix))

    c = np.array([var.Obj for var in problem.getVars()])
    c = np.concatenate((c, np.zeros(columns_to_add)))

    if problem.ModelSense == gp.GRB.MINIMIZE:
        print("Converting minimization to maximization problem.")
        c = -c

    assert A.shape == (problem.NumConstrs, problem.NumVars + columns_to_add)
    assert b.shape == (problem.NumConstrs,)
    assert c.shape == (problem.NumVars + columns_to_add,)

    with open(output_file, "w") as f:
        f.write(f"{problem.NumConstrs} {problem.NumVars + columns_to_add}\n")
        for i in range(problem.NumConstrs):
            row = " ".join(map(str, A[i, :].flatten().tolist()[0]))
            f.write(f"{row}\n")
        f.write(" ".join(map(str, b.flatten().tolist())) + "\n")
        f.write(" ".join(map(str, c.flatten().tolist())) + "\n")


if __name__ == "__main__":
    parsed_args = args.parse_args()
    convert_mps_to_canonical(
        parsed_args.mps_file_path,
        parsed_args.output_file,
    )
