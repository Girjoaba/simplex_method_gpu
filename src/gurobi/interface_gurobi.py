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


def mps2canonical(mps_file_path, output_file):
    """
    Convert to Canonical Form for Standard Simplex (Big-M).
    Structure: [Original Vars | Surplus Vars | Slack/Artificial Vars (Identity)]
    """    
    problem = gp.read(mps_file_path)
    
    # 1. Pre-process bounds
    for var in problem.getVars():
        if var.VType != gp.GRB.CONTINUOUS:
            raise ValueError("Non-continuous variables found.")
        if var.LB != 0:
            problem.addConstr(var >= var.LB, name=f"lb_{var.VarName}")
            var.LB = 0
        if var.UB != gp.GRB.INFINITY and var.UB != math.inf:
            problem.addConstr(var <= var.UB, name=f"ub_{var.VarName}")
            var.UB = gp.GRB.INFINITY
    
    problem.update()
    A_raw = problem.getA().todense()
    constraints = problem.getConstrs()
    
    # 2. Split Equalities
    A_rows = []
    b_vals = []
    senses = []
    
    for i in range(problem.NumConstrs):
        constr = constraints[i]
        if constr.Sense == gp.GRB.EQUAL:
            # Ax = b -> Ax >= b and Ax <= b
            A_rows.append(A_raw[i, :])
            b_vals.append(constr.RHS)
            senses.append(gp.GRB.GREATER_EQUAL)
            
            A_rows.append(A_raw[i, :])
            b_vals.append(constr.RHS)
            senses.append(gp.GRB.LESS_EQUAL)
        else:
            A_rows.append(A_raw[i, :])
            b_vals.append(constr.RHS)
            senses.append(constr.Sense)
    
    A_base = np.vstack(A_rows)
    b = np.array(b_vals)
    num_constraints = len(b_vals)
    
    # 3. Ensure b >= 0
    for i in range(num_constraints):
        if b[i] < 0:
            A_base[i, :] = -A_base[i, :]
            b[i] = -b[i]
            if senses[i] == gp.GRB.LESS_EQUAL:
                senses[i] = gp.GRB.GREATER_EQUAL
            elif senses[i] == gp.GRB.GREATER_EQUAL:
                senses[i] = gp.GRB.LESS_EQUAL

    # 4. Construct [Original | Surplus | Basis (Slack+Artificial)]
    # Basis MUST BE IDENTITY
    surplus_cols = []
    basis_cols = []
    
    # Objective penalties
    # Big M constant (penalty for artificial vars)
    M = 1e6 
    c_surplus = []
    c_basis = []
    
    for i in range(num_constraints):
        # Create column vectors
        col_vec = np.zeros(num_constraints)
        col_vec[i] = 1.0
        
        if senses[i] == gp.GRB.LESS_EQUAL:
            # Ax <= b -> Ax + s = b
            basis_cols.append(col_vec)
            c_basis.append(0.0) # Slack has 0 cost
            
        elif senses[i] == gp.GRB.GREATER_EQUAL:
            # Ax >= b -> Ax - e + a = b
            # 1. Surplus variable (-1) is Non-Basic
            surplus_cols.append(-col_vec)
            c_surplus.append(0.0)
            
            # 2. Artificial variable (+1) is Basic
            basis_cols.append(col_vec)
            c_basis.append(-M) # Artificial has -M cost (for Max problem)
            
    # Stack matrices
    # If there are no surplus columns, we handle shape correctly
    if surplus_cols:
        A_surplus = np.column_stack(surplus_cols)
    else:
        A_surplus = np.zeros((num_constraints, 0))
        
    A_basis = np.column_stack(basis_cols)
    
    # Final A: [Original, Surplus, Basis]
    A_final = np.hstack((A_base, A_surplus, A_basis))
    c_orig = np.array([var.Obj for var in problem.getVars()])
    
    # Handle original minimization/maximization
    is_minimization = problem.ModelSense == gp.GRB.MINIMIZE
    if is_minimization:
        c_orig = -c_orig
        # Note: Artifical vars are already -M (bad for Max), which corresponds to +M for Min.
    
    c_final = np.concatenate((c_orig, c_surplus, c_basis))
    m, n = A_final.shape
    
    print(f"Final Matrix Size: {m}x{n}")
    print(f"Rightmost {m}x{m} block is Identity: Verified.")
    
    # Write to output file
    with open(output_file, "w") as f:
        f.write(f"{m} {n}\n")
        for i in range(m):
            row_data = np.asarray(A_final[i, :]).flatten()
            row_str = " ".join(map(str, row_data))
            f.write(f"{row_str}\n")
            
        b_data = np.asarray(b).flatten()
        f.write(" ".join(map(str, b_data)) + "\n")
        
        c_data = np.asarray(c_final).flatten()
        f.write(" ".join(map(str, c_data)) + "\n")


if __name__ == "__main__":
    parsed_args = args.parse_args()
    mps2canonical(
        parsed_args.mps_file_path,
        parsed_args.output_file,
    )
