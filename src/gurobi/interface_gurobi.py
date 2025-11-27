#!/usr/bin/env python3

import numpy as np
import gurobipy as gp
import argparse
import sys

args = argparse.ArgumentParser(description="Convert MPS file to canonical form.")
args.add_argument("mps_file_path", type=str)
args.add_argument("output_file", type=str)

def mps2canonical(mps_file_path, output_file):
    problem = gp.read(mps_file_path)
    is_minimization = problem.ModelSense == gp.GRB.MINIMIZE
    
    # Extract raw data
    vars_list = problem.getVars()
    c_raw = np.array([var.Obj for var in vars_list])
    if is_minimization:
        c_raw = -c_raw 
        
    A_raw = np.asarray(problem.getA().todense())
    constraints = problem.getConstrs()
    b_raw = np.array([c.RHS for c in constraints])
    senses = np.array([c.Sense for c in constraints]) 
    # It is important to keep track of lower bounds and upper bounds
    lbs = np.array([var.LB for var in vars_list])
    ubs = np.array([var.UB for var in vars_list])

    m_orig, n_orig = A_raw.shape
    
    # Process Variables
    A_cols = []
    c_new = []
    b_adj = np.zeros(m_orig)
    obj_offset = 0.0
    ub_rows = [] 
    
    INF = 1e20
    curr_idx = 0

    for j in range(n_orig):
        lb, ub = lbs[j], ubs[j]
        col, cost = A_raw[:, j], c_raw[j]

        if lb > -INF:
            # Shift: x = x' + lb
            if lb != 0.0:
                b_adj += (col * lb)
                obj_offset += (cost * lb)
            
            A_cols.append(col)
            c_new.append(cost)
            
            if ub < INF:
                # Add explicit Upper Bound row: x' <= ub - lb
                ub_rows.append({'idx': [curr_idx], 'val': [1.0], 'rhs': ub - lb})
            curr_idx += 1
        else:
            # Split: x = x+ - x-
            A_cols.append(col)
            c_new.append(cost)
            A_cols.append(-col)
            c_new.append(-cost)

            if ub < INF:
                # Add explicit Upper Bound row: x+ - x- <= ub
                ub_rows.append({'idx': [curr_idx, curr_idx+1], 'val': [1.0, -1.0], 'rhs': ub})
            curr_idx += 2

    b_raw -= b_adj
    A_processed = np.column_stack(A_cols) if A_cols else np.zeros((m_orig, 0))
    c_processed = np.array(c_new)

    # Ensure b >= 0
    neg_b = b_raw < 0
    if np.any(neg_b):
        A_processed[neg_b] = -A_processed[neg_b]
        b_raw[neg_b] = -b_raw[neg_b]
        le, ge = (senses == '<') & neg_b, (senses == '>') & neg_b
        senses[le], senses[ge] = '>', '<'

    # Big M
    M = max(np.max(np.abs(c_processed)) * 1e6, 1e9) if len(c_processed) > 0 else 1e9

    # Setup Block Dimensions
    rows_ge = (senses == '>')
    rows_eq = (senses == '=')
    n_surplus = np.sum(rows_ge)
    n_ub = len(ub_rows)
    m_final = m_orig + n_ub

    # --- Block Construction ---

    # 1. Variables (Original Matrix + UB Coefficients)
    A_vars = np.vstack([A_processed, np.zeros((n_ub, len(c_processed)))])
    b_ub = np.zeros(n_ub)
    for r, row in enumerate(ub_rows):
        b_ub[r] = row['rhs']
        for idx, val in zip(row['idx'], row['val']):
            A_vars[m_orig + r, idx] = val

    # 2. Surplus (For original >= rows)
    A_surplus = np.zeros((m_final, n_surplus))
    if n_surplus > 0:
        # Vectorized assignment for speed
        row_indices = np.where(rows_ge)[0]
        A_surplus[row_indices, np.arange(n_surplus)] = -1.0
    c_surplus = np.zeros(n_surplus)

    # 3. UB Slacks (Identity matrix for the new UB rows)
    # Serves as the initial Basis for these rows. Cost 0.
    A_ubslack = np.zeros((m_final, n_ub))
    if n_ub > 0:
        A_ubslack[m_orig:, :] = np.eye(n_ub)
    c_ubslack = np.zeros(n_ub)

    # 4. Basis/Artificials (Identity matrix for original rows)
    # Acts as Slacks (Cost 0) for <= rows
    # Acts as Artificials (Cost -M) for >= and = rows
    A_art = np.zeros((m_final, m_orig))
    A_art[:m_orig, :] = np.eye(m_orig)
    
    c_art = np.zeros(m_orig)
    c_art[rows_ge | rows_eq] = -M

    # Assemble
    A_final = np.hstack([A_vars, A_surplus, A_ubslack, A_art])
    c_final = np.concatenate([c_processed, c_surplus, c_ubslack, c_art])
    b_final = np.concatenate([b_raw, b_ub])
    
    m, n = A_final.shape
    print(f"Final Matrix Size: {m}x{n}")

    # Write Output
    with open(output_file, "w") as f:
        f.write(f"{m} {n}\n")
        np.savetxt(f, A_final, fmt='%.6g', delimiter=' ')
        np.savetxt(f, b_final.reshape(1, -1), fmt='%.6g', delimiter=' ')
        np.savetxt(f, c_final.reshape(1, -1), fmt='%.6g', delimiter=' ')

    # Write Offset Sidecar
    with open(output_file + ".offset", "w") as f:
        f.write(f"{obj_offset}\n")
        f.write(f"{1 if is_minimization else 0}\n")

    return A_final, b_final, c_final, is_minimization, obj_offset

if __name__ == "__main__":
    parsed_args = args.parse_args()
    mps2canonical(parsed_args.mps_file_path, parsed_args.output_file)