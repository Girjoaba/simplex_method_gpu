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

    model = gp.Model("canonical_lp")
    model.Params.OutputFlag = 0  # silence solver if you like

    # Variables: here I assume nonnegativity (lb=0). If your canonical
    # form allows negative variables, use lb=-gp.GRB.INFINITY instead.
    x = model.addMVar(n, lb=0.0, name="x")

    # Constraints: A x = b
    for i in range(m):
        model.addConstr(A[i, :] @ x == b[i], name=f"row_{i}")

    # Objective: maximize c^T x
    model.setObjective(c @ x, gp.GRB.MAXIMIZE)

    model.optimize()

    if model.Status == gp.GRB.OPTIMAL:
        # Output format required by benchmark infrastructure
        print(f"Optimum found: {model.ObjVal:.16f}")
        for i in range(len(x.X)):
            print(f"x[{i}] = {x.X[i]:.16e}")
        print(f"Iterations: {int(model.IterCount)}")
    else:
        # Handle non-optimal status
        print(f"Solver failed with status: {model.Status}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parsed_args = args.parse_args()
    solve_canonical_file(
        parsed_args.canonical_file_path,
    )