from interface_gurobi import mps2canonical
from solver_gurobi import solve_canonical

import gurobipy as gp
import numpy as np
import math
import glob

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
    A_final, b, c_final, is_minimization, obj_offset = mps2canonical(mps_file_path, output_file)
    status, obj_val, x_vals, iters = solve_canonical(A_final, b, c_final)
    
    if status == gp.GRB.OPTIMAL:
        corrected_max_obj = obj_val + obj_offset
        
        if is_minimization:
            final_display_obj = -corrected_max_obj
        else:
            final_display_obj = corrected_max_obj
            
        print(f"Canonical form objective (Validation): {final_display_obj}")
        return final_display_obj
    else:
        print(f"Validation Failed. Status: {status}")
        return None

if __name__ == "__main__":
    mps_files = glob.glob("test/input/*.mps")
    output_file = "canonical_output.txt"
    
    for mps_file in mps_files:
        ground_truth = solve_with_gurobi(mps_file)
        canonical_result = convert_and_solve_canonical(mps_file, output_file)
        
        failed = False
        error_msg = ""

        if (ground_truth is None) != (canonical_result is None):
            failed = True
            error_msg = f"Solver Status Mismatch (GT: {ground_truth}, Canon: {canonical_result})"
            
        elif ground_truth is not None and canonical_result is not None:
            if not math.isclose(ground_truth, canonical_result, rel_tol=1e-5, abs_tol=1e-5):
                failed = True
                diff = abs(ground_truth - canonical_result)
                error_msg = f"Objective Mismatch (Diff: {diff:.6f} | GT: {ground_truth:.4f} vs Canon: {canonical_result:.4f})"

        if failed:
            print(f"FAIL [{mps_file}]: {error_msg}")
        else:
            print("Congratz, interface works! :D")
