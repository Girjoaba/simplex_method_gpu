#!/usr/bin/env python3
import numpy as np
import random
import argparse

args = argparse.ArgumentParser(
    description="Generate a random LP problem in canonical form."
)
args.add_argument(
    "num_vars",
    type=int,
    help="Number of variables.",
)
args.add_argument(
    "num_constraints",
    type=int,
    help="Number of constraints.",
)
args.add_argument(
    "output_file",
    type=str,
    help="Path to save the generated problem.",
)
args.add_argument(
    "--density",
    type=float,
    default=0.8,
    help="Density of the constraint matrix (between 0 and 1). Default is 0.8.",
)
args.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility. Default is None.",
)


def generate_random(num_vars, num_constraints, output_file, density=0.8, seed=None):
    """Generate a random linear programming problem in canonical form.

    Args:
        num_vars (int): Number of variables.
        num_constraints (int): Number of constraints.
        density (float): Density of the constraint matrix (between 0 and 1).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        A (np.ndarray): Coefficient matrix of shape (num_constraints, num_vars).
        b (np.ndarray): Right-hand side vector of shape (num_constraints,).
        c (np.ndarray): Coefficient vector of the objective function of shape (num_vars,).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    A = np.zeros((num_constraints, num_vars))
    for i in range(num_constraints):
        for j in range(num_vars):
            if random.random() < density:
                A[i, j] = random.uniform(-10, 10)

    # add slack variables to convert inequalities to equalities
    slack_vars = np.eye(num_constraints)
    A = np.hstack((A, slack_vars))
    c = np.random.uniform(0, 10, size=num_vars)
    num_vars += (
        num_constraints  # update number of variables after adding slack variables
    )

    b = np.random.uniform(10, 100, size=num_constraints)
    c = np.concatenate((c, np.zeros(num_constraints)))

    with open(output_file, "w") as f:
        f.write(f"{num_constraints} {num_vars}\n")
        for i in range(num_constraints):
            f.write(" ".join(map(str, A[i])) + "\n")
        f.write(" ".join(map(str, b)) + "\n")
        f.write(" ".join(map(str, c)) + "\n")


if __name__ == "__main__":
    parsed_args = args.parse_args()
    generate_random(
        parsed_args.num_vars,
        parsed_args.num_constraints,
        parsed_args.output_file,
        parsed_args.density,
        parsed_args.seed,
    )
