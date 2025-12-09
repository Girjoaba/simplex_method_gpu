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
    "agg": {
        "m": 698,
        "n": 488,
        "size": 340624,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/agg.canonical",
        "expected_optimum": 35991696.93936366,
        "description": "whatever",
    },
    "bandm": {
        "m": 777,
        "n": 305,
        "size": 236985,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/bandm.canonical",
        "expected_optimum": 158.62801845009145,
        "description": "whatever",
    },
    "beaconfd": {
        "m": 435,
        "n": 173,
        "size": 75255,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/beaconfd.canonical",
        "expected_optimum": -33592.48580719999,
        "description": "whatever",
    },
    "blend": {
        "m": 157,
        "n": 74,
        "size": 11618,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/blend.canonical",
        "expected_optimum": 30.81214984582821,
        "description": "whatever",
    },
    "boeing1": {
        "m": 1317,
        "n": 596,
        "size": 784932,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/boeing1.canonical",
        "expected_optimum": 358.9428485847717,
        "description": "whatever",
    },
    "boeing2": {
        "m": 542,
        "n": 239,
        "size": 129538,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/boeing2.canonical",
        "expected_optimum": 324.84362801520274,
        "description": "whatever",
    },
    "bore3d": {
        "m": 560,
        "n": 245,
        "size": 137200,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/bore3d.canonical",
        "expected_optimum": -730.0391635745854,
        "description": "whatever",
    },
    "brandy": {
        "m": 469,
        "n": 220,
        "size": 103180,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/brandy.canonical",
        "expected_optimum": -1518.5098964881292,
        "description": "whatever",
    },
    "capri": {
        "m": 843,
        "n": 418,
        "size": 352374,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/capri.canonical",
        "expected_optimum": -2690.035553719555,
        "description": "whatever",
    },
    "etamacro": {
        "m": 1384,
        "n": 617,
        "size": 853928,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/etamacro.canonical",
        "expected_optimum": -444.66160497715276,
        "description": "whatever",
    },
    "finnis": {
        "m": 1334,
        "n": 578,
        "size": 771052,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/finnis.canonical",
        "expected_optimum": -72756.5420106706,
        "description": "whatever",
    },
    "gfrd-pnc": {
        "m": 1966,
        "n": 874,
        "size": 1718284,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/gfrd-pnc.canonical",
        "expected_optimum": -6902237.379548814,
        "description": "whatever",
    },
    "grow7": {
        "m": 721,
        "n": 420,
        "size": 302820,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/grow7.canonical",
        "expected_optimum": 47787811.814710684,
        "description": "whatever",
    },
    "israel": {
        "m": 324,
        "n": 174,
        "size": 56376,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/israel.canonical",
        "expected_optimum": 896644.8218630458,
        "description": "whatever",
    },
    "kb2": {
        "m": 108,
        "n": 52,
        "size": 5616,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/kb2.canonical",
        "expected_optimum": 1749.900198893992,
        "description": "whatever",
    },
    "lotfi": {
        "m": 477,
        "n": 153,
        "size": 72981,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/lotfi.canonical",
        "expected_optimum": 25.264708000000006,
        "description": "whatever",
    },
    "recipe": {
        "m": 384,
        "n": 186,
        "size": 71424,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/recipe.canonical",
        "expected_optimum": 266.29199999999963,
        "description": "whatever",
    },
    "sc105": {
        "m": 208,
        "n": 105,
        "size": 21840,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/sc105.canonical",
        "expected_optimum": 52.202061211707246,
        "description": "whatever",
    },
    "sc205": {
        "m": 408,
        "n": 205,
        "size": 83640,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/sc205.canonical",
        "expected_optimum": 52.202061211707225,
        "description": "whatever",
    },
    "sc50a": {
        "m": 98,
        "n": 50,
        "size": 4900,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/sc50a.canonical",
        "expected_optimum": 64.57507705856449,
        "description": "whatever",
    },
    "sc50b": {
        "m": 98,
        "n": 50,
        "size": 4900,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/sc50b.canonical",
        "expected_optimum": 70.0,
        "description": "whatever",
    },
    "scagr25": {
        "m": 996,
        "n": 471,
        "size": 469116,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/scagr25.canonical",
        "expected_optimum": 14753433.060768526,
        "description": "whatever",
    },
    "scagr7": {
        "m": 276,
        "n": 129,
        "size": 35604,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/scagr7.canonical",
        "expected_optimum": 2331389.824330984,
        "description": "whatever",
    },
    "scfxm1": {
        "m": 787,
        "n": 330,
        "size": 259710,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/scfxm1.canonical",
        "expected_optimum": -18416.759028348944,
        "description": "whatever",
    },
    "scorpion": {
        "m": 806,
        "n": 388,
        "size": 312728,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/scorpion.canonical",
        "expected_optimum": -1878.1248227381068,
        "description": "whatever",
    },
    "scrs8": {
        "m": 1678,
        "n": 490,
        "size": 822220,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/scrs8.canonical",
        "expected_optimum": -904.2967085282481,
        "description": "whatever",
    },
    "scsd1": {
        "m": 837,
        "n": 77,
        "size": 64449,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/scsd1.canonical",
        "expected_optimum": -8.666673885776593,
        "description": "whatever",
    },
    "sctap1": {
        "m": 960,
        "n": 300,
        "size": 288000,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/sctap1.canonical",
        "expected_optimum": -1412.25,
        "description": "whatever",
    },
    "share1b": {
        "m": 342,
        "n": 117,
        "size": 40014,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/share1b.canonical",
        "expected_optimum": 76589.31993230517,
        "description": "whatever",
    },
    "share2b": {
        "m": 175,
        "n": 96,
        "size": 16800,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/share2b.canonical",
        "expected_optimum": 415.7322407414198,
        "description": "whatever",
    },
    "stair": {
        "m": 1009,
        "n": 444,
        "size": 447996,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/stair.canonical",
        "expected_optimum": 251.26695801062297,
        "description": "whatever",
    },
    "standata": {
        "m": 1565,
        "n": 479,
        "size": 749635,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/standata.canonical",
        "expected_optimum": -1257.6995,
        "description": "whatever",
    },
    "standgub": {
        "m": 1676,
        "n": 481,
        "size": 806156,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/standgub.canonical",
        "expected_optimum": -1257.6995,
        "description": "whatever",
    },
    "standmps": {
        "m": 1673,
        "n": 587,
        "size": 982051,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/standmps.canonical",
        "expected_optimum": -1406.0174999999997,
        "description": "whatever",
    },
    "stocfor1": {
        "m": 234,
        "n": 117,
        "size": 27378,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/stocfor1.canonical",
        "expected_optimum": 41131.9762194364,
        "description": "whatever",
    },
    "vtp.base": {
        "m": 511,
        "n": 281,
        "size": 143591,
        "file": "/home/ubuntu/simplex_method_gpu/test/input/vtp.base.canonical",
        "expected_optimum": -129831.45648043863,
        "description": "whatever",
    },
}

# -----------------------------------------------------------------------------
# SOLVER DEFINITIONS
# -----------------------------------------------------------------------------

SOLVERS = {
    "glop": {
        "binary": os.path.join(PROJECT_ROOT, "glop_baseline/build/glop_canonical"),
        "description": "Google GLOP (serial, revised simplex)",
        "type": "serial",
    },
    "gurobi": {
        "binary": os.path.join(PROJECT_ROOT, "src/gurobi/gurobi_canonical"),
        "description": "Gurobi 11.0.3 (Python API, dual simplex)",
        "type": "serial",
    },
    # cuda_slow/ solvers
    "cuda_slow_v1_cpu": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/v1_cpu.out"),
        "description": "CPU Eigen baseline (double)",
        "type": "serial",
    },
    "cuda_slow_v1_lu": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/v1_lu_cuda.out"),
        "description": "cuSolver LU (double)",
        "type": "parallel",
    },
    "cuda_slow_v2": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/v2_A2device.out"),
        "description": "Matrix A on device (double)",
        "type": "parallel",
    },
    "cuda_slow_v3": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/v3_cublas_mvm.out"),
        "description": "cuBLAS matrix-vector (double)",
        "type": "parallel",
    },
    "cuda_slow_v5": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/v5_thrust_max_elem.out"),
        "description": "Thrust max_element (double)",
        "type": "parallel",
    },
    "cuda_slow_v6": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/v6_update_curr_pos.out"),
        "description": "Optimized position updates (double)",
        "type": "parallel",
    },
    "cuda_slow_v7": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/v7_thrust_ratio.out"),
        "description": "Thrust transform_reduce (double)",
        "type": "parallel",
    },
    "cuda_slow_v8": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/v8_full_gpu.out"),
        "description": "Full GPU parallelization (double)",
        "type": "parallel",
    },
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
    "baseline": {
        "description": "Baseline performance on 5 standard problems",
        "problems": [
            "sample",
            "sample3",
            "afiro",
            "adlittle",
            "e226",
        ],  # Ordered by size
        "solvers": ["glop"],
        "repetitions": 1000,  # Good balance of statistical confidence and runtime
        "warmup_iterations": 100,  # Suite-level warmup to eliminate cold start
        "warmup_problem": "sample",  # Use smallest problem for warmup
    },
    "quick_test": {
        "description": "Quick test with small problem (10 reps)",
        "problems": ["sample"],
        "solvers": ["glop"],
        "repetitions": 10,
        "warmup_iterations": 10,  # Scaled-down warmup for quick testing
        "warmup_problem": "sample",
    },
    "all_problems": {
        "description": "All 6 problems with balanced runtime (100 reps)",
        "problems": ["sample", "sample3", "afiro", "adlittle", "e226", "d6cube"],
        "solvers": ["glop"],
        "repetitions": 100,
        "warmup_iterations": 100,
        "warmup_problem": "afiro",  # Median-sized problem (27×51, size 1377)
    },
    "comparison": {
        "description": "Compare GLOP vs Gurobi on all 6 problems",
        "problems": ["sample", "sample3", "afiro", "adlittle", "e226", "d6cube"],
        "solvers": ["glop", "gurobi"],
        "repetitions": 100,
        "warmup_iterations": 100,
        "warmup_problem": "afiro",  # Median-sized problem (27×51, size 1377)
    },
    "cuda_slow_progression": {
        "description": "Compare cuda_slow optimization progression",
        "problems": ["sample", "sample3", "afiro", "adlittle", "e226"],
        "solvers": [
            "cuda_slow_v1_cpu",
            "cuda_slow_v1_lu",
            "cuda_slow_v2",
            "cuda_slow_v3",
            "cuda_slow_v5",
            "cuda_slow_v6",
            "cuda_slow_v7",
            "cuda_slow_v8",
            "glop",
        ],
        "repetitions": 100,
        "warmup_iterations": 50,
        "warmup_problem": "afiro",
    },
}

# -----------------------------------------------------------------------------
# PLOTTING CONFIGURATION
# -----------------------------------------------------------------------------

PLOT_STYLE = {
    "figure_size": (12, 6),
    "title_fontsize": 18,
    "label_fontsize": 14,
    "tick_fontsize": 12,
    "legend_fontsize": 12,
    "dpi": 300,
}

# Colors for different solver types
SOLVER_COLORS = {
    "glop": "#2E86AB",  # Blue
    "gurobi": "#A23B72",  # Purple
    "cuda_v1": "#F18F01",  # Orange
    "cuda_v2": "#C73E1D",  # Red
    # cuda_slow solvers
    "cuda_slow_v1_cpu": "#E63946",  # Red
    "cuda_slow_v1_lu": "#F4A261",  # Orange
    "cuda_slow_v2": "#E9C46A",  # Yellow
    "cuda_slow_v3": "#2A9D8F",  # Teal
    "cuda_slow_v5": "#264653",  # Dark blue
    "cuda_slow_v6": "#8338EC",  # Purple
    "cuda_slow_v7": "#FF006E",  # Pink
    "cuda_slow_v8": "#3A86FF",  # Blue
}
