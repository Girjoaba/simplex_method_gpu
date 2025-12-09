"""
Benchmark configuration for Simplex solver testing.

Defines:
- PROBLEMS: Test problems with metadata (dimensions, file paths, expected optima)
- SOLVERS: Available solvers (binaries, descriptions)
- SUITES: Predefined benchmark suites (which problems, which solvers, how many reps)
"""

import os

# Get absolute path to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# PROBLEM DEFINITIONS
# -----------------------------------------------------------------------------
# Problems are ordered by size (m × n) for plotting

PROBLEMS = {
    'sample': {
        'm': 2,
        'n': 4,
        'size': 8,  # m × n for plotting
        'file': os.path.join(PROJECT_ROOT, 'test/input/sample.canonical'),
        'expected_optimum': 9.0,
        'description': 'Tiny test problem'
    },
    'sample3': {
        'm': 5,
        'n': 15,
        'size': 75,
        'file': os.path.join(PROJECT_ROOT, 'test/input/sample3.canonical'),
        'expected_optimum': 4620.8333333,
        'description': 'Small test problem'
    },
    'afiro': {
        'm': 27,
        'n': 51,
        'size': 1377,
        'file': os.path.join(PROJECT_ROOT, 'test/input/afiro.canonical'),
        'expected_optimum': 464.75314285714285,
        'description': 'NETLIB afiro problem'
    },
    'adlittle': {
        'm': 56,
        'n': 138,
        'size': 7728,
        'file': os.path.join(PROJECT_ROOT, 'test/input/adlittle.canonical'),
        'expected_optimum': -225494.96316238024,
        'description': 'NETLIB adlittle problem'
    },
    'e226': {
        'm': 223,
        'n': 472,
        'size': 105256,
        'file': os.path.join(PROJECT_ROOT, 'test/input/e226.canonical'),
        'expected_optimum': 18.751929066370547,
        'description': 'NETLIB e226 problem'
    },
    'd6cube': {
        'm': 416,
        'n': 6184,
        'size': 2572544,
        'file': os.path.join(PROJECT_ROOT, 'test/input/d6cube.canonical'),
        'expected_optimum': -315.49166666667,
        'description': 'NETLIB d6cube problem (large)'
    }
}

# -----------------------------------------------------------------------------
# SOLVER DEFINITIONS
# -----------------------------------------------------------------------------

SOLVERS = {
    'glop': {
        'binary': os.path.join(PROJECT_ROOT, 'glop_baseline/build/glop_canonical'),
        'description': 'Google GLOP (serial, revised simplex)',
        'type': 'serial'
    },
    'gurobi': {
        'binary': os.path.join(PROJECT_ROOT, 'src/gurobi/gurobi_canonical'),
        'description': 'Gurobi 11.0.3 (Python API, dual simplex)',
        'type': 'serial'
    },
    # cuda_slow/ solvers
    'cuda_slow_v1_cpu': {
        'binary': os.path.join(PROJECT_ROOT, 'bin_solver/v1_cpu.out'),
        'description': 'CPU Eigen baseline (double)',
        'type': 'serial'
    },
    'cuda_slow_v1_lu': {
        'binary': os.path.join(PROJECT_ROOT, 'bin_solver/v1_lu_cuda.out'),
        'description': 'cuSolver LU (double)',
        'type': 'parallel'
    },
    'cuda_slow_v2': {
        'binary': os.path.join(PROJECT_ROOT, 'bin_solver/v2_A2device.out'),
        'description': 'Matrix A on device (double)',
        'type': 'parallel'
    },
    'cuda_slow_v3': {
        'binary': os.path.join(PROJECT_ROOT, 'bin_solver/v3_cublas_mvm.out'),
        'description': 'cuBLAS matrix-vector (double)',
        'type': 'parallel'
    },
    'cuda_slow_v5': {
        'binary': os.path.join(PROJECT_ROOT, 'bin_solver/v5_thrust_max_elem.out'),
        'description': 'Thrust max_element (double)',
        'type': 'parallel'
    },
    'cuda_slow_v6': {
        'binary': os.path.join(PROJECT_ROOT, 'bin_solver/v6_update_curr_pos.out'),
        'description': 'Optimized position updates (double)',
        'type': 'parallel'
    },
    'cuda_slow_v7': {
        'binary': os.path.join(PROJECT_ROOT, 'bin_solver/v7_thrust_ratio.out'),
        'description': 'Thrust transform_reduce (double)',
        'type': 'parallel'
    },
    'cuda_slow_v8': {
        'binary': os.path.join(PROJECT_ROOT, 'bin_solver/v8_full_gpu.out'),
        'description': 'Full GPU parallelization (double)',
        'type': 'parallel'
    }
    # Future solvers:
    # 'cuda_v1': {
    #     'binary': os.path.join(PROJECT_ROOT, 'cuda_simplex/build/cuda_canonical'),
    #     'description': 'CUDA Simplex (parallel)',
    #     'type': 'parallel'
    # }
}

# -----------------------------------------------------------------------------
# BENCHMARK SUITES
# -----------------------------------------------------------------------------

SUITES = {
    'baseline': {
        'description': 'Baseline performance on 5 standard problems',
        'problems': ['sample', 'sample3', 'afiro', 'adlittle', 'e226'],  # Ordered by size
        'solvers': ['glop'],
        'repetitions': 1000,  # Good balance of statistical confidence and runtime
        'warmup_iterations': 100,  # Suite-level warmup to eliminate cold start
        'warmup_problem': 'sample'  # Use smallest problem for warmup
    },
    'quick_test': {
        'description': 'Quick test with small problem (10 reps)',
        'problems': ['sample'],
        'solvers': ['glop'],
        'repetitions': 10,
        'warmup_iterations': 10,  # Scaled-down warmup for quick testing
        'warmup_problem': 'sample'
    },
    'all_problems': {
        'description': 'All 6 problems with balanced runtime (100 reps)',
        'problems': ['sample', 'sample3', 'afiro', 'adlittle', 'e226', 'd6cube'],
        'solvers': ['glop'],
        'repetitions': 100,
        'warmup_iterations': 100,
        'warmup_problem': 'afiro'  # Median-sized problem (27×51, size 1377)
    },
    'comparison': {
        'description': 'Compare GLOP vs Gurobi on all 6 problems',
        'problems': ['sample', 'sample3', 'afiro', 'adlittle', 'e226', 'd6cube'],
        'solvers': ['glop', 'gurobi'],
        'repetitions': 100,
        'warmup_iterations': 100,
        'warmup_problem': 'afiro'  # Median-sized problem (27×51, size 1377)
    },
    'cuda_slow_progression': {
        'description': 'Compare cuda_slow optimization progression',
        'problems': ['sample', 'sample3', 'afiro', 'adlittle', 'e226'],
        'solvers': ['cuda_slow_v1_cpu', 'cuda_slow_v1_lu', 'cuda_slow_v2',
                    'cuda_slow_v3', 'cuda_slow_v5', 'cuda_slow_v6',
                    'cuda_slow_v7', 'cuda_slow_v8', 'glop'],
        'repetitions': 100,
        'warmup_iterations': 50,
        'warmup_problem': 'afiro'
    }
}

# -----------------------------------------------------------------------------
# PLOTTING CONFIGURATION
# -----------------------------------------------------------------------------

PLOT_STYLE = {
    'figure_size': (12, 6),
    'title_fontsize': 18,
    'label_fontsize': 14,
    'tick_fontsize': 12,
    'legend_fontsize': 12,
    'dpi': 300
}

# Colors for different solver types
SOLVER_COLORS = {
    'glop': '#2E86AB',      # Blue
    'gurobi': '#A23B72',    # Purple
    'cuda_v1': '#F18F01',   # Orange
    'cuda_v2': '#C73E1D',   # Red
    # cuda_slow solvers
    'cuda_slow_v1_cpu': '#E63946',   # Red
    'cuda_slow_v1_lu': '#F4A261',    # Orange
    'cuda_slow_v2': '#E9C46A',       # Yellow
    'cuda_slow_v3': '#2A9D8F',       # Teal
    'cuda_slow_v5': '#264653',       # Dark blue
    'cuda_slow_v6': '#8338EC',       # Purple
    'cuda_slow_v7': '#FF006E',       # Pink
    'cuda_slow_v8': '#3A86FF'        # Blue
}
