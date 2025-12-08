import gurobipy as gp
import numpy as np
import argparse
import sys

args = argparse.ArgumentParser(description="Path to the canonical problem file.")
args.add_argument(
    "canonical_file_path",
    type=str,
    help="Path to the MPS file to be converted.",
)

def solve_canonical(A, b, c):
    """
    Shared core logic to solve Max c^T x s.t. Ax = b, x >= 0
    Returns (status, obj_val, x_vector, iter_count)
    """
    m, n = A.shape
    
    # Create Model
    model = gp.Model("canonical_core")
    model.Params.OutputFlag = 0
    
    # We use MVar for efficient dense matrix handling
    x = model.addMVar(n, lb=0.0, name="x")

    # Constraints: A x = b
    model.addConstr(A @ x == b, name="eq_constrs")
    model.setObjective(c @ x, gp.GRB.MAXIMIZE)
    model.optimize()
    
    # Extract results if optimal
    if model.Status == gp.GRB.OPTIMAL:
        return model.Status, model.ObjVal, x.X, model.IterCount
    else:
        return model.Status, None, None, 0


def solve_canonical_file(path):
    # Read file
    with open(path) as f:
        # First line: m n
        m, n = map(int, f.readline().split())

        # Next m lines: A rows
        A_rows = []
        for _ in range(m):
            A_rows.append(list(map(float, f.readline().split())))
        A = np.array(A_rows, dtype=float)
        b = np.array(list(map(float, f.readline().split())), dtype=float)
        c = np.array(list(map(float, f.readline().split())), dtype=float)

        status, obj_val, x_vals, iters = solve_canonical(A, b, c)

        if status == gp.GRB.OPTIMAL:
            # Output format required by benchmark infrastructure
            print(f"Optimum found: {obj_val:.16f}")
            for i in range(len(x_vals)):
                print(f"x[{i}] = {x_vals[i]:.16e}")
            print(f"Iterations: {int(iters)}")
        else:
            print(f"Solver failed with status: {status}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    parsed_args = args.parse_args()
    solve_canonical_file(
        parsed_args.canonical_file_path,
    )