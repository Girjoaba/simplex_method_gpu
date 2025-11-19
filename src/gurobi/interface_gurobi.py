#!/usr/bin/env python3
import numpy as np
import gurobipy as gp
import argparse
import math

args = argparse.ArgumentParser(description="Convert MPS file to canonical form.")
args.add_argument("mps_file_path", type=str, help="Path to the MPS file to be converted.")
args.add_argument("output_file", type=str, help="Path to save the converted canonical form.")

def convert_mps_to_canonical(mps_file_path, output_file):
    """Convert MPS file to canonical form with all variables >= 0."""
    problem = gp.read(mps_file_path)
    
    # Check all variables are continuous
    for var in problem.getVars():
        if var.VType != gp.GRB.CONTINUOUS:
            raise ValueError("The MPS file contains non-continuous variables.")
    
    # Transform variables to have LB = 0
    original_vars = problem.getVars().copy()
    var_shifts = {}  # Store the shift amount for each variable
    
    for var in original_vars:
        if var.LB != 0:
            var_shifts[var.VarName] = var.LB
            # Update variable bounds: x_new = x_old - LB
            var.LB = 0
            if var.UB != gp.GRB.INFINITY:
                var.UB = var.UB - var_shifts[var.VarName]
        
        # Handle finite upper bounds by adding constraint
        if var.UB != gp.GRB.INFINITY and var.UB != math.inf:
            problem.addConstr(var <= var.UB, name=f"ub_{var.VarName}")
            var.UB = gp.GRB.INFINITY
    
    # Adjust RHS of existing constraints due to variable shifts
    problem.update()
    if var_shifts:
        for constr in problem.getConstrs():
            shift_amount = 0
            for var in original_vars:
                if var.VarName in var_shifts:
                    coef = problem.getCoeff(constr, var)
                    shift_amount += coef * var_shifts[var.VarName]
            if shift_amount != 0:
                constr.RHS = constr.RHS - shift_amount
    
    problem.update()
    
    # Now convert to standard form with slacks/surplus/artificials
    A = problem.getA().todense()
    constraints = problem.getConstrs()
    m = problem.NumConstrs
    b = np.array([constraints[i].RHS for i in range(m)])
    
    # Track surplus variables (for >= constraints)
    surplus_cols = []
    for i in range(m):
        constr = constraints[i]
        if constr.Sense == gp.GRB.GREATER_EQUAL:
            surplus_cols.append(i)
    
    # Add surplus variables (coefficient -1)
    num_surplus = len(surplus_cols)
    if num_surplus > 0:
        surplus_matrix = np.zeros((m, num_surplus))
        for idx, row in enumerate(surplus_cols):
            surplus_matrix[row, idx] = -1.0
        A = np.hstack((A, surplus_matrix))
    
    # Add identity matrix for initial basis (slacks and artificials)
    identity_matrix = np.eye(m)
    A = np.hstack((A, identity_matrix))
    
    # Extend objective with zeros for surplus and identity columns
    c = np.array([var.Obj for var in problem.getVars()])
    
    # Adjust objective for variable shifts
    obj_shift = sum(c[i] * var_shifts.get(original_vars[i].VarName, 0) 
                    for i in range(len(original_vars)))
    
    c = np.concatenate((c, np.zeros(num_surplus + m)))
    
    # Report what was added
    num_slacks = sum(1 for c in constraints if c.Sense == gp.GRB.LESS_EQUAL)
    num_artificials = sum(1 for c in constraints if c.Sense == gp.GRB.EQUAL) + num_surplus
    print(f"Added {num_slacks} slacks, {num_surplus} surplus, {num_artificials} artificials")
    if obj_shift != 0:
        print(f"Objective shift due to variable transformation: {obj_shift}")
    
    # Convert minimization to maximization
    if problem.ModelSense == gp.GRB.MINIMIZE:
        print("Converting minimization to maximization problem.")
        c = -c
        obj_shift = -obj_shift
    
    n_total = problem.NumVars + num_surplus + m
    assert A.shape == (m, n_total)
    assert b.shape == (m,)
    assert c.shape == (n_total,)
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(f"{m} {n_total}\n")
        for i in range(m):
            row = " ".join(map(str, A[i, :].flatten().tolist()[0]))
            f.write(f"{row}\n")
        f.write(" ".join(map(str, b.flatten().tolist())) + "\n")
        f.write(" ".join(map(str, c.flatten().tolist())) + "\n")
    
    print(f"Converted problem saved to {output_file}")
    if obj_shift != 0:
        print(f"Note: Add {obj_shift} to the optimal objective value from the solver")

if __name__ == "__main__":
    parsed_args = args.parse_args()
    convert_mps_to_canonical(parsed_args.mps_file_path, parsed_args.output_file)