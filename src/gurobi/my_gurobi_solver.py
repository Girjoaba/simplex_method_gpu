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
    """Convert MPS file to canonical form and solve with Gurobi."""
    print("\n" + "="*60)
    print("Converting to canonical form and solving:")
    print("="*60)
    
    problem = gp.read(mps_file_path)
    
    for var in problem.getVars():
        if var.VType != gp.GRB.CONTINUOUS:
            raise ValueError("The MPS file contains non-continuous variables.")
        if var.LB != 0:
            problem.addConstr(var >= var.LB, name=f"lb_{var.VarName}")
            var.LB = 0
        if var.UB != gp.GRB.INFINITY and var.UB != math.inf:
            problem.addConstr(var <= var.UB, name=f"ub_{var.VarName}")
            var.UB = gp.GRB.INFINITY
    
    problem.update()
    A = problem.getA().todense()
    constraints = problem.getConstrs()
    
    # STEP 0: Split equality constraints into >= and <=
    A_rows = []
    b_vals = []
    senses = []
    
    for i in range(problem.NumConstrs):
        constr = constraints[i]
        if constr.Sense == gp.GRB.EQUAL:
            # Split Ax = b into Ax >= b and Ax <= b
            A_rows.append(A[i, :])
            b_vals.append(constr.RHS)
            senses.append(gp.GRB.GREATER_EQUAL)
            
            A_rows.append(A[i, :])
            b_vals.append(constr.RHS)
            senses.append(gp.GRB.LESS_EQUAL)
        else:
            A_rows.append(A[i, :])
            b_vals.append(constr.RHS)
            senses.append(constr.Sense)
    
    A = np.vstack(A_rows)
    b = np.array(b_vals)
    num_constraints = len(b_vals)
    
    
    # STEP 1: Ensure all b >= 0 (flip constraints if needed, and flip their sense)
    for i in range(num_constraints):
        if b[i] < 0:
            A[i, :] = -A[i, :]
            b[i] = -b[i]
            # Flip the sense: <= becomes >=, >= becomes <=
            if senses[i] == gp.GRB.LESS_EQUAL:
                senses[i] = gp.GRB.GREATER_EQUAL
            elif senses[i] == gp.GRB.GREATER_EQUAL:
                senses[i] = gp.GRB.LESS_EQUAL
    # STEP 2: Convert all >= to <= by multiplying by -1
    slack_vals = [0] * num_constraints
    columns_to_add = 0
    
    for i in range(num_constraints):
        if senses[i] == gp.GRB.GREATER_EQUAL:
            A[i, :] = -A[i, :]
            b[i] = -b[i]
            slack_vals[i] = 1
            columns_to_add += 1
        elif senses[i] == gp.GRB.LESS_EQUAL:
            slack_vals[i] = 1
            columns_to_add += 1
    
    # STEP 3: Add slack variables (all with +1 coefficient since everything is now <=)
    if columns_to_add > 0:
        slack_matrix = np.zeros((num_constraints, columns_to_add))
        slack_col_index = 0
        for i in range(num_constraints):
            if slack_vals[i] == 0:
                continue
            slack_matrix[i, slack_col_index] = slack_vals[i]
            slack_col_index += 1
            
        # After creating slack matrix, verify:
        if columns_to_add != num_constraints:
            raise ValueError(f"Expected {num_constraints} slack variables, got {columns_to_add}")
        print(f"Added {columns_to_add} slack variables forming identity matrix")

        print_slack_matrix(slack_matrix, "slack_matrix_debug.txt")  # Add this line

        A = np.hstack((A, slack_matrix))
    
    c = np.array([var.Obj for var in problem.getVars()])
    c = np.concatenate((c, np.zeros(columns_to_add)))
    
    is_minimization = problem.ModelSense == gp.GRB.MINIMIZE
    if is_minimization:
        print("Converting minimization to maximization problem.")
        c = -c
    
    m = A.shape[0]  # constraints
    n = A.shape[1]  # variables
    
    # Write to output file
    with open(output_file, "w") as f:
        f.write(f"{m} {n}\n")
        for i in range(m):
            row = " ".join(map(str, A[i, :].flatten().tolist()[0]))
            f.write(f"{row}\n")
        f.write(" ".join(map(str, b.flatten().tolist())) + "\n")
        f.write(" ".join(map(str, c.flatten().tolist())) + "\n")
    
    # create a new gurobi model and solve this one
    canonical_model = gp.Model("canonical")
    x = canonical_model.addVars(n, lb=0, ub=gp.GRB.INFINITY, name="x")
    # Add equality constraints: Ax = b
    for i in range(m):
        canonical_model.addConstr(
            gp.quicksum(A[i, j] * x[j] for j in range(n)) == b[i],
            name=f"c{i}"
        )
    # This is MAXIMIZATION (can also minimize)
    canonical_model.setObjective( 
        gp.quicksum(c[j] * x[j] for j in range(n)),
        gp.GRB.MAXIMIZE
    )
    
    canonical_model.optimize()
    if canonical_model.status == gp.GRB.OPTIMAL:
        canonical_objective = canonical_model.objVal
        # Convert back to original problem sense
        if is_minimization:
            canonical_objective = -canonical_objective
        
        print(f"Canonical form objective: {canonical_objective}")
        return canonical_objective
    else:
        print(f"Canonical optimization status: {canonical_model.status}")
        return None

if __name__ == "__main__":
    mps_file = "test/input/adlittle.mps"
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