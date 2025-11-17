#!/usr/bin/env python3
"""
Solve an MPS file with Gurobi and print the optimal objective value.
"""

import gurobipy as gp
import sys

def solve_mps(mps_path):
    """Solve MPS file and return optimal objective."""
    model = gp.read(mps_path)
    model.optimize()
    
    if model.Status == gp.GRB.OPTIMAL:
        print(f"Optimal objective: {model.ObjVal}")
        return model.ObjVal
    else:
        print(f"Not optimal. Status: {model.Status}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solve_mps.py <file.mps>")
        sys.exit(1)
    
    solve_mps(sys.argv[1])