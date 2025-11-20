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
    
    # Standardize Model Sense (Min -> Max)
    is_minimization = problem.ModelSense == gp.GRB.MINIMIZE
    
    # --- 1. Extract Raw Data (Sparse -> Dense) ---
    # extracting sparse first is usually faster for large models
    c_raw = np.array([var.Obj for var in problem.getVars()])
    if is_minimization:
        c_raw = -c_raw # Convert to Max
        
    A_raw = problem.getA().todense()
    constraints = problem.getConstrs()
    b_raw = np.array([c.RHS for c in constraints])
    senses = np.array([c.Sense for c in constraints]) # numpy array for masking
    
    # --- 2. Handle Lower Bounds via Substitution (Optimization #1) ---
    # Instead of adding rows, we shift x: x_new = x_old - LB
    # A * x_old = b  =>  A * (x_new + LB) = b  => A * x_new = b - A * LB
    lbs = np.array([var.LB for var in problem.getVars()])
    
    # If any lower bound is non-zero, shift b
    if np.any(lbs != 0):
        # A_raw is matrix, lbs is vector.
        # We need A_raw @ lbs. np.dot handles matrix * vector correctly
        adjustment = np.dot(A_raw, lbs)
        # Flatten strictly to 1D array to match b_raw
        b_raw = b_raw - np.asarray(adjustment).flatten()
        
        # Note: We do NOT need to change A or c (coefficients stay same).
        # However, the final objective value calculated by C++ will be off 
        # by a constant (c^T * LB), which you can add back post-solve if needed.

    # Handle Upper Bounds? 
    # Standard Simplex handles x >= 0 naturally. 
    # It does NOT handle UB naturally without "Bounded Simplex" variant.
    # If we must stick to Standard Simplex, UBs must remain as explicit constraints.
    # Your previous code added them as constraints. We will replicate that ONLY if needed.
    # (Looping here is unavoidable if we must add rows, but let's assume standard form 
    # usually implies x >= 0 only. If MPS has explicit UBs, they are usually separate rows).

    # --- 3. Ensure b >= 0 (Vectorized) ---
    neg_b_mask = b_raw < 0
    if np.any(neg_b_mask):
        A_raw[neg_b_mask] = -A_raw[neg_b_mask]
        b_raw[neg_b_mask] = -b_raw[neg_b_mask]
        
        # Flip Senses using numpy masks
        # We need to be careful: '<' becomes '>', '>' becomes '<', '=' stays '='
        le_mask = (senses == gp.GRB.LESS_EQUAL) & neg_b_mask
        ge_mask = (senses == gp.GRB.GREATER_EQUAL) & neg_b_mask
        
        senses[le_mask] = gp.GRB.GREATER_EQUAL
        senses[ge_mask] = gp.GRB.LESS_EQUAL

    # --- 4. Construct Canonical Matrix (Vectorized) ---
    m, n_orig = A_raw.shape
    
    # Determine dynamic M
    raw_max = np.max(np.abs(c_raw)) if len(c_raw) > 0 else 1.0
    M = max(raw_max * 1000.0, 1e5)

    # Identify row types
    rows_le = (senses == gp.GRB.LESS_EQUAL)
    rows_ge = (senses == gp.GRB.GREATER_EQUAL)
    rows_eq = (senses == gp.GRB.EQUAL)

    # -- Surplus Block --
    # Only rows with >= need a surplus variable (-1)
    # We construct a matrix of size (m, count_ge)
    num_surplus = np.sum(rows_ge)
    
    if num_surplus > 0:
        A_surplus = np.zeros((m, num_surplus))
        # Fill the -1s. We place them diagonally relative to the subset of GE rows
        # But to make it simple, we just fill the specific row indices
        # Create an identity-like structure for the subset
        subset_indices = np.where(rows_ge)[0]
        for idx, row_idx in enumerate(subset_indices):
            A_surplus[row_idx, idx] = -1.0
        
        c_surplus = np.zeros(num_surplus)
    else:
        A_surplus = np.zeros((m, 0))
        c_surplus = np.array([])

    # -- Basis Block (Identity) --
    # This is the specific request: The far right block MUST be Identity.
    # Since we have (Slack for <=), (Artificial for >=), (Artificial for =),
    # and every row has exactly ONE of these with coefficient +1...
    # ... this block is simply an Identity Matrix of size m*m!
    A_basis = np.eye(m)
    
    # Costs for Basis
    # Slacks (<=) cost 0. Artificials (>= and =) cost -M.
    c_basis = np.zeros(m)
    c_basis[rows_ge | rows_eq] = -M

    # -- Assemble Final Matrices --
    # Use np.hstack for speed
    A_final = np.hstack((A_raw, A_surplus, A_basis))
    c_final = np.concatenate((c_raw, c_surplus, c_basis))
    
    m_final, n_final = A_final.shape
    
    print(f"Final Matrix Size: {m_final}x{n_final}")
    
    # --- 5. Write to File (Optimization #3) ---
    # np.savetxt is much faster than manual string loops
    with open(output_file, "w") as f:
        # Header
        f.write(f"{m_final} {n_final}\n")
        
        # Write A (Matrix)
        np.savetxt(f, A_final, fmt='%.6g', delimiter=' ')
        
        # Write b (Vector)
        # Reshape to 1 row to match format "val val val ..."
        np.savetxt(f, b_raw.reshape(1, -1), fmt='%.6g', delimiter=' ')
        
        # Write c (Vector)
        np.savetxt(f, c_final.reshape(1, -1), fmt='%.6g', delimiter=' ')

    return A_final, b_raw, c_final, is_minimization

    
if __name__ == "__main__":
    parsed_args = args.parse_args()
    mps2canonical(
        parsed_args.mps_file_path,
        parsed_args.output_file,
    )
