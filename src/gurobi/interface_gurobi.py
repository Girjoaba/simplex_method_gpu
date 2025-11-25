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
    
    problem = gp.read(mps_file_path)
    is_minimization = problem.ModelSense == gp.GRB.MINIMIZE
    
    # extract data
    c_raw = np.array([var.Obj for var in problem.getVars()])
    if is_minimization:
        c_raw = -c_raw # Convert to Max
        
    A_raw = problem.getA().todense()
    constraints = problem.getConstrs()
    b_raw = np.array([c.RHS for c in constraints])
    senses = np.array([c.Sense for c in constraints]) # numpy array for masking
    
    # handle LB
    # x_new = x_old - LB
    # A * x_old = b  =>  A * (x_new + LB) = b  => A * x_new = b - A * LB
    lbs = np.array([var.LB for var in problem.getVars()])
    if np.any(lbs != 0):
        adjustment = np.dot(A_raw, lbs)
        b_raw = b_raw - np.asarray(adjustment).flatten()

    # ensure b >= 0
    neg_b_mask = b_raw < 0
    if np.any(neg_b_mask):
        A_raw[neg_b_mask] = -A_raw[neg_b_mask]
        b_raw[neg_b_mask] = -b_raw[neg_b_mask]
        
        # flip senses using numpy masks
        # '<' becomes '>', '>' becomes '<', '=' stays '='
        le_mask = (senses == gp.GRB.LESS_EQUAL) & neg_b_mask
        ge_mask = (senses == gp.GRB.GREATER_EQUAL) & neg_b_mask
        
        senses[le_mask] = gp.GRB.GREATER_EQUAL
        senses[ge_mask] = gp.GRB.LESS_EQUAL

    # construct canonical matrix
    m, n_orig = A_raw.shape
    
    # Determine dynamic BIG M
    raw_max = np.max(np.abs(c_raw)) if len(c_raw) > 0 else 1.0
    M = max(raw_max * 1000.0, 1e5)

    # Identify row types
    rows_le = (senses == gp.GRB.LESS_EQUAL)
    rows_ge = (senses == gp.GRB.GREATER_EQUAL)
    rows_eq = (senses == gp.GRB.EQUAL)

    # -- Surplus Block --
    # only rows with >= need a surplus variable (-1)
    num_surplus = np.sum(rows_ge)
    if num_surplus > 0:
        A_surplus = np.zeros((m, num_surplus))
        subset_indices = np.where(rows_ge)[0]
        for idx, row_idx in enumerate(subset_indices):
            A_surplus[row_idx, idx] = -1.0
        
        c_surplus = np.zeros(num_surplus)
    else:
        A_surplus = np.zeros((m, 0))
        c_surplus = np.array([])

    # -- Basis Block (Identity) --
    # the far right block MUST be Identity.
    A_basis = np.eye(m)
    
    # artificials (>= and =) cost -M.
    c_basis = np.zeros(m)
    c_basis[rows_ge | rows_eq] = -M

    # -- Assemble Final Matrices --
    A_final = np.hstack((A_raw, A_surplus, A_basis))
    c_final = np.concatenate((c_raw, c_surplus, c_basis))
    
    m_final, n_final = A_final.shape
    
    print(f"Final Matrix Size: {m_final}x{n_final}")
    
    # write output:
    # n m
    # A (m x n)
    # b (m)
    # c (n)
    with open(output_file, "w") as f:
        f.write(f"{m_final} {n_final}\n")
        np.savetxt(f, A_final, fmt='%.6g', delimiter=' ')
        np.savetxt(f, b_raw.reshape(1, -1), fmt='%.6g', delimiter=' ')
        np.savetxt(f, c_final.reshape(1, -1), fmt='%.6g', delimiter=' ')

    return A_final, b_raw, c_final, is_minimization

    
if __name__ == "__main__":
    parsed_args = args.parse_args()
    mps2canonical(
        parsed_args.mps_file_path,
        parsed_args.output_file,
    )
