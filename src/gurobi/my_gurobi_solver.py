from interface_gurobi import mps2canonical

import gurobipy as gp
import numpy as np
import math

def print_slack_matrix(slack_matrix, filename="slack_matrix.txt"):
    """Print the slack matrix to a file for verification."""
    with open(filename, "w") as f:
        f.write(f"Slack matrix shape: {slack_matrix.shape}\n")
        f.write(f"Should be an {slack_matrix.shape[0]}×{slack_matrix.shape[1]} identity matrix\n\n")
        for i in range(slack_matrix.shape[0]):
            row = " ".join(map(str, slack_matrix[i, :]))
            f.write(f"{row}\n")
    print(f"Slack matrix saved to {filename}")

def solve_with_gurobi(mps_file_path):
    """Solve MPS file directly with Gurobi to get ground truth."""
    print("="*60)
    print("Solving with Gurobi (Ground Truth):")
    print("="*60)
    
    model = gp.read(mps_file_path)
    model.optimize()
    
    if model.status == gp.GRB.OPTIMAL:
        obj_const = model.ObjCon
        true_objective = model.objVal - obj_const
        
        print(f"Gurobi reported objective: {model.objVal}")
        print(f"Objective constant: {obj_const}")
        print(f"True objective value: {true_objective}")
        
        return true_objective
    else:
        print(f"Optimization status: {model.status}")
        return None

def convert_and_solve_canonical(mps_file_path, output_file):
    """
    Convert to Canonical Form for Standard Simplex (Big-M).
    Structure: [Original Vars | Surplus Vars | Slack/Artificial Vars (Identity)]
    """
    A_final, b, c_final, is_minimization = mps2canonical(mps_file_path, output_file)
    m, n = A_final.shape

    # --- Verification Step ---
    canonical_model = gp.Model("canonical")
    canonical_model.setParam('OutputFlag', 0)
    x = canonical_model.addVars(n, lb=0, ub=gp.GRB.INFINITY, name="x")
    
    for i in range(m):
        canonical_model.addConstr(
            gp.quicksum(A_final[i, j] * x[j] for j in range(n)) == b[i]
        )
        
    canonical_model.setObjective( 
        gp.quicksum(c_final[j] * x[j] for j in range(n)),
        gp.GRB.MAXIMIZE
    )
    
    canonical_model.optimize()
    
    if canonical_model.status == gp.GRB.OPTIMAL:
        obj_val = canonical_model.objVal
        # If Big-M worked, artificial vars should be 0.
        # For display, we convert back if original was min
        if is_minimization:
            obj_val = -obj_val
            
        print(f"Canonical form objective (Validation): {obj_val}")
        return obj_val
    else:
        print(f"Status: {canonical_model.status}")
        return None

if __name__ == "__main__":
    mps_file = "test/input/e226.mps"
    output_file = "canonical_output.txt"
    
    ground_truth = solve_with_gurobi(mps_file)
    canonical_result = convert_and_solve_canonical(mps_file, output_file)
    
    # Compare results
    print(f"\n{'='*60}")
    print(f"Ground truth objective:   {ground_truth}")
    print(f"Canonical form objective: {canonical_result}")
    if ground_truth is not None and canonical_result is not None:
        diff = abs(ground_truth - canonical_result)
        print(f"Difference: {diff}")
        print(f"{'='*60}")
        if diff < 1e-6:
            print("Results match, sweet :D")
        else:
            print("Results differ, go back to work :(")