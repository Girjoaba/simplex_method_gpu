# Revised Simplex Algorithm in CUDA

## Introduction

This repository contains the project for the DPHPC course at ETH. We have
implemented the Revised Simplex algorithm in CUDA, employing both the Big-M and
Two-Phase methods. Multiple implementations are provided, each building on the
previous one and introducing additionals optimizations. The Big-M
implementations focus on maximizing performance, while the Two-Phase
implementations successfully transition to sparse linear algebra.

## Requirements

- CUDA Toolkit
- [Eigen](https://libeigen.gitlab.io)
- cuDSS (required only for the Two-Phase `v3_sparse.cu`)
- Python:
    - numpy
    - gurobipy

## Compilation

To specify your GPU’s compute capability and set the library paths, compile the
solvers as follows:

```bash
make   ARCH=sm_XX   EIGEN_DIR=path-to-eigen   CUDSS_DIR=path-to-cudss
```

## Quick Start

### Key Files

- `scripts/problem_summary.csv` - [summary table](https://www.netlib.org/lp/data/readme) of Netlib LP problems with known optimum
- `scripts/prepare_problems.sh` - downloads compressed MPS files from Netlib, then expands and preprocesses them
- `src/gurobi/interface.py` - saves an MPS file as a plain-text file in canonical form
- `scripts/solve_and_compare.sh` - runs a solver on the Netlib problems and compares its output to the groundtruth

### Example Run

To test the solver `bin_solver/bm_v8_full_gpu.out` on all Netlib problems with at most 10,000 non-zero entries in the constraint matrix, run:

```bash
./scripts/prepare_problems.sh
./scripts/solve_and_compare.sh   bin_solver/bm_v8_full_gpu.out   10000
```

To test the solver on a specific problem, run:

```bash
./scripts/prepare_problems.sh
./bin_solver/bm_v8_full_gpu.out < test/netlib/preproccessed/afiro.preproccessed
```

