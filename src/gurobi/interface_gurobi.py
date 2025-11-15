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

    m = problem.NumConstrs
    b = np.array([constraints[i].RHS for i in range(problem.NumConstrs)])

    # For each constraint type, determine what variables to add:
    # <= : slack variable (coefficient +1, used in initial basis)
    # >= : surplus variable (coefficient -1, NOT in basis) + artificial (coefficient +1, in basis)
    # =  : artificial variable (coefficient +1, used in initial basis)

    surplus_cols = []  # List of (row_index, coefficient) for >= constraints

    for i in range(m):
        constr = constraints[i]
        if constr.Sense == gp.GRB.GREATER_EQUAL:
            # Add surplus variable (coefficient -1, not in basis)
            surplus_cols.append(i)

    # Create surplus matrix for >= constraints (if any)
    num_surplus = len(surplus_cols)
    if num_surplus > 0:
        surplus_matrix = np.zeros((m, num_surplus))
        for idx, row in enumerate(surplus_cols):
            surplus_matrix[row, idx] = -1.0
        A = np.hstack((A, surplus_matrix))

    # Create identity matrix for initial basis (last m columns)
    # All rows get coefficient +1 (either slack for <=, or artificial for >= and =)
    identity_matrix = np.eye(m)
    A = np.hstack((A, identity_matrix))

    # Extend objective with zeros for surplus variables and identity matrix
    # Note: Artificials and slacks both get zero cost
    # The solver will naturally drive them out if better solutions exist
    c = np.array([var.Obj for var in problem.getVars()])
    c = np.concatenate((c, np.zeros(num_surplus + m)))

    # Count for reporting
    num_slacks = sum(1 for c in constraints if c.Sense == gp.GRB.LESS_EQUAL)
    num_artificials = sum(1 for c in constraints if c.Sense == gp.GRB.EQUAL) + num_surplus
    print(f"Added {num_slacks} slacks, {num_surplus} surplus, {num_artificials} artificials")

    if problem.ModelSense == gp.GRB.MINIMIZE:
        print("Converting minimization to maximization problem.")
        c = -c

    n_total = problem.NumVars + num_surplus + m
    assert A.shape == (m, n_total)
    assert b.shape == (m,)
    assert c.shape == (n_total,)

    with open(output_file, "w") as f:
        f.write(f"{m} {n_total}\n")
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
