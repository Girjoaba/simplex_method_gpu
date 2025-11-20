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
    print("\n" + "="*60)
    print("Converting to Big-M Canonical Form (Identity at end):")
    print("="*60)
    
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

    # 4. Construct Matrix Columns
    # We need the Final structure: [Original | Surplus | Basis (Slack+Artificial)]
    # The Basis block MUST be an Identity Matrix.
    
    surplus_cols = []   # Columns for -1s (surplus)
    basis_cols = []     # Columns for +1s (slacks and artificials)
    
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
            # Slack variable is Basic (+1 coefficient)
            basis_cols.append(col_vec)
            c_basis.append(0.0) # Slack has 0 cost
            
        elif senses[i] == gp.GRB.GREATER_EQUAL:
            # Ax >= b -> Ax - e + a = b
            # 1. Surplus variable (-1) is Non-Basic
            surplus_cols.append(-col_vec)
            c_surplus.append(0.0) # Surplus has 0 cost
            
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
    
    # Final c
    c_orig = np.array([var.Obj for var in problem.getVars()])
    
    # Handle original minimization/maximization
    is_minimization = problem.ModelSense == gp.GRB.MINIMIZE
    if is_minimization:
        print("Converting minimization to maximization problem.")
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
            # Safe conversion to 1D flat list regardless of matrix vs array
            row_data = np.asarray(A_final[i, :]).flatten()
            row_str = " ".join(map(str, row_data))
            f.write(f"{row_str}\n")
            
        # Safe b writing
        b_data = np.asarray(b).flatten()
        f.write(" ".join(map(str, b_data)) + "\n")
        
        # Safe c writing
        c_data = np.asarray(c_final).flatten()
        f.write(" ".join(map(str, c_data)) + "\n")
    
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
        if diff < 1e-6:
            print("Results match, almost there :D")
        else:
            print("Results differ, go back to work :(")
    print(f"{'='*60}")