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
    "netlib_vtp.base": {
        "m": 511,
        "n": 281,
        "size": 143591,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_vtp.base",
        "expected_optimum": -129831.45648043863,
        "description": "whatever",
    },
    "netlib_stocfor2": {
        "m": 4314,
        "n": 2157,
        "size": 9305298,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_stocfor2",
        "expected_optimum": 39024.40853788212,
        "description": "whatever",
    },
    "netlib_stocfor1": {
        "m": 234,
        "n": 117,
        "size": 27378,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_stocfor1",
        "expected_optimum": 41131.9762194364,
        "description": "whatever",
    },
    "netlib_standmps": {
        "m": 1673,
        "n": 587,
        "size": 982051,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_standmps",
        "expected_optimum": -1406.0174999999997,
        "description": "whatever",
    },
    "netlib_standgub": {
        "m": 1676,
        "n": 481,
        "size": 806156,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_standgub",
        "expected_optimum": -1257.6995,
        "description": "whatever",
    },
    "netlib_standata": {
        "m": 1565,
        "n": 479,
        "size": 749635,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_standata",
        "expected_optimum": -1257.6995,
        "description": "whatever",
    },
    "netlib_stair": {
        "m": 1009,
        "n": 444,
        "size": 447996,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_stair",
        "expected_optimum": 251.26695801062297,
        "description": "whatever",
    },
    "netlib_sierra": {
        "m": 5365,
        "n": 3263,
        "size": 17505995,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_sierra",
        "expected_optimum": -15394362.183631929,
        "description": "whatever",
    },
    "netlib_ship12s": {
        "m": 3919,
        "n": 1151,
        "size": 4510769,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_ship12s",
        "expected_optimum": -1489236.1344061329,
        "description": "whatever",
    },
    "netlib_ship08s": {
        "m": 3173,
        "n": 778,
        "size": 2468594,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_ship08s",
        "expected_optimum": -1920098.2105346182,
        "description": "whatever",
    },
    "netlib_ship04s": {
        "m": 1868,
        "n": 402,
        "size": 750936,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_ship04s",
        "expected_optimum": -1798714.7004453912,
        "description": "whatever",
    },
    "netlib_ship04l": {
        "m": 2528,
        "n": 402,
        "size": 1016256,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_ship04l",
        "expected_optimum": -1793324.537970356,
        "description": "whatever",
    },
    "netlib_share2b": {
        "m": 175,
        "n": 96,
        "size": 16800,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_share2b",
        "expected_optimum": 415.7322407414198,
        "description": "whatever",
    },
    "netlib_share1b": {
        "m": 342,
        "n": 117,
        "size": 40014,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_share1b",
        "expected_optimum": 76589.31993230517,
        "description": "whatever",
    },
    "netlib_sctap1": {
        "m": 960,
        "n": 300,
        "size": 288000,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_sctap1",
        "expected_optimum": -1412.25,
        "description": "whatever",
    },
    "netlib_scsd8": {
        "m": 3147,
        "n": 397,
        "size": 1249359,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scsd8",
        "expected_optimum": -904.9997320645351,
        "description": "whatever",
    },
    "netlib_scsd6": {
        "m": 1497,
        "n": 147,
        "size": 220059,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scsd6",
        "expected_optimum": -50.500023366914874,
        "description": "whatever",
    },
    "netlib_scsd1": {
        "m": 837,
        "n": 77,
        "size": 64449,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scsd1",
        "expected_optimum": -8.666673885776593,
        "description": "whatever",
    },
    "netlib_scrs8": {
        "m": 1678,
        "n": 490,
        "size": 822220,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scrs8",
        "expected_optimum": -904.2967085282481,
        "description": "whatever",
    },
    "netlib_scorpion": {
        "m": 806,
        "n": 388,
        "size": 312728,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scorpion",
        "expected_optimum": -1878.1248227381068,
        "description": "whatever",
    },
    "netlib_scfxm3": {
        "m": 2361,
        "n": 990,
        "size": 2337390,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scfxm3",
        "expected_optimum": -54901.25454975146,
        "description": "whatever",
    },
    "netlib_scfxm2": {
        "m": 1574,
        "n": 660,
        "size": 1038840,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scfxm2",
        "expected_optimum": -36660.26156499881,
        "description": "whatever",
    },
    "netlib_scfxm1": {
        "m": 787,
        "n": 330,
        "size": 259710,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scfxm1",
        "expected_optimum": -18416.759028348944,
        "description": "whatever",
    },
    "netlib_scagr7": {
        "m": 276,
        "n": 129,
        "size": 35604,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scagr7",
        "expected_optimum": 2331389.824330984,
        "description": "whatever",
    },
    "netlib_scagr25": {
        "m": 996,
        "n": 471,
        "size": 469116,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_scagr25",
        "expected_optimum": 14753433.060768526,
        "description": "whatever",
    },
    "netlib_sc50b": {
        "m": 98,
        "n": 50,
        "size": 4900,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_sc50b",
        "expected_optimum": 70.0,
        "description": "whatever",
    },
    "netlib_sc50a": {
        "m": 98,
        "n": 50,
        "size": 4900,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_sc50a",
        "expected_optimum": 64.57507705856449,
        "description": "whatever",
    },
    "netlib_sc205": {
        "m": 408,
        "n": 205,
        "size": 83640,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_sc205",
        "expected_optimum": 52.202061211707225,
        "description": "whatever",
    },
    "netlib_sc105": {
        "m": 208,
        "n": 105,
        "size": 21840,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_sc105",
        "expected_optimum": 52.202061211707246,
        "description": "whatever",
    },
    "netlib_lotfi": {
        "m": 477,
        "n": 153,
        "size": 72981,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_lotfi",
        "expected_optimum": 25.264708000000006,
        "description": "whatever",
    },
    "netlib_kb2": {
        "m": 108,
        "n": 52,
        "size": 5616,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_kb2",
        "expected_optimum": 1749.900198893992,
        "description": "whatever",
    },
    "netlib_israel": {
        "m": 324,
        "n": 174,
        "size": 56376,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_israel",
        "expected_optimum": 896644.8218630458,
        "description": "whatever",
    },
    "netlib_grow7": {
        "m": 721,
        "n": 420,
        "size": 302820,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_grow7",
        "expected_optimum": 47787811.814710684,
        "description": "whatever",
    },
    "netlib_grow22": {
        "m": 2266,
        "n": 1320,
        "size": 2991120,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_grow22",
        "expected_optimum": 160834336.48256317,
        "description": "whatever",
    },
    "netlib_grow15": {
        "m": 1545,
        "n": 900,
        "size": 1390500,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_grow15",
        "expected_optimum": 106870941.29357532,
        "description": "whatever",
    },
    "netlib_gfrd-pnc": {
        "m": 1966,
        "n": 874,
        "size": 1718284,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_gfrd-pnc",
        "expected_optimum": -6902237.379548814,
        "description": "whatever",
    },
    "netlib_ganges": {
        "m": 3387,
        "n": 1706,
        "size": 5778222,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_ganges",
        "expected_optimum": 109585.74345963489,
        "description": "whatever",
    },
    "netlib_forplan": {
        "m": 625,
        "n": 183,
        "size": 114375,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_forplan",
        "expected_optimum": 1163.915768685812,
        "description": "whatever",
    },
    "netlib_fit1p": {
        "m": 2703,
        "n": 1026,
        "size": 2773278,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_fit1p",
        "expected_optimum": -9146.378092420935,
        "description": "whatever",
    },
    "netlib_fit1d": {
        "m": 2087,
        "n": 1050,
        "size": 2191350,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_fit1d",
        "expected_optimum": 9146.378092420928,
        "description": "whatever",
    },
    "netlib_fffff800": {
        "m": 1468,
        "n": 524,
        "size": 769232,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_fffff800",
        "expected_optimum": -555680.7779460382,
        "description": "whatever",
    },
    "netlib_e226": {
        "m": 521,
        "n": 223,
        "size": 116183,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_e226",
        "expected_optimum": 18.751929066370547,
        "description": "whatever",
    },
    "netlib_czprob": {
        "m": 4682,
        "n": 1158,
        "size": 5421756,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_czprob",
        "expected_optimum": -2185197.1641155668,
        "description": "whatever",
    },
    "netlib_capri": {
        "m": 843,
        "n": 418,
        "size": 352374,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_capri",
        "expected_optimum": -2690.035553719555,
        "description": "whatever",
    },
    "netlib_brandy": {
        "m": 469,
        "n": 220,
        "size": 103180,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_brandy",
        "expected_optimum": -1518.5098964881292,
        "description": "whatever",
    },
    "netlib_blend": {
        "m": 157,
        "n": 74,
        "size": 11618,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_blend",
        "expected_optimum": 30.81214984582821,
        "description": "whatever",
    },
    "netlib_beaconfd": {
        "m": 435,
        "n": 173,
        "size": 75255,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_beaconfd",
        "expected_optimum": -33592.48580719999,
        "description": "whatever",
    },
    "netlib_bandm": {
        "m": 777,
        "n": 305,
        "size": 236985,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_bandm",
        "expected_optimum": 158.62801845009145,
        "description": "whatever",
    },
    "netlib_agg3": {
        "m": 818,
        "n": 516,
        "size": 422088,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_agg3",
        "expected_optimum": -10312058.591095194,
        "description": "whatever",
    },
    "netlib_agg2": {
        "m": 818,
        "n": 516,
        "size": 422088,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_agg2",
        "expected_optimum": 20239263.24727866,
        "description": "whatever",
    },
    "netlib_agg": {
        "m": 698,
        "n": 488,
        "size": 340624,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_agg",
        "expected_optimum": 35991696.93936366,
        "description": "whatever",
    },
    "netlib_afiro": {
        "m": 59,
        "n": 27,
        "size": 1593,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_afiro",
        "expected_optimum": 464.75314285714285,
        "description": "whatever",
    },
    "netlib_adlittle": {
        "m": 154,
        "n": 56,
        "size": 8624,
        "file": os.path.join(PROJECT_ROOT, "benchmarks/input/") + "netlib_adlittle",
        "expected_optimum": -225494.96316238018,
        "description": "whatever",
    },
}


# -----------------------------------------------------------------------------
# SOLVER DEFINITIONS
# -----------------------------------------------------------------------------

SOLVERS = {
    # Sooner or later needed, gurobi baseline
    "bm_v5": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/bm_v5_thrust_max_elem.out"),
        "description": "bm partial gpu",
        "type": "parallel",
        "input_method": "stdin",
    },
    "bm_v8": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/bm_v8_full_gpu.out"),
        "description": "bm all on gpu",
        "type": "parallel",
        "input_method": "stdin",
    },
    "bm_v10": {
        "binary": os.path.join(
            PROJECT_ROOT, "bin_solver/bm_v10_sherman_morris_opt.out"
        ),
        "description": "bm with sherman morris op",
        "type": "parallel",
        "input_method": "stdin",
    },
    "bm_v11": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/bm_v11_sparse.out"),
        "description": "bm with sparse",
        "type": "parallel",
        "input_method": "stdin",
    },
    "tp_v1": {
        "binary": os.path.join(PROJECT_ROOT, "bin_solver/tp_v1.out"),
        "description": "Two phase first",
        "type": "parallel",
        "input_method": "stdin",
    },
    "gurobi": {
        "binary": os.path.join(PROJECT_ROOT, ".venv/bin/python3") + " " + os.path.join(PROJECT_ROOT, "src/gurobi/solver_gurobi.py"),
        "description": "Gurobi commercial solver (simplex)",
        "type": "serial",
        "input_method": "file_arg",
    },
}

# -----------------------------------------------------------------------------
# SUITE HELPERS (Problem Lists)
# -----------------------------------------------------------------------------

# The 30 smallest problems (sorted by size approx < 450k)
SMALL_30_PROBLEMS = [
    "netlib_afiro", "netlib_sc50b", "netlib_sc50a", "netlib_kb2", "netlib_adlittle",
    "netlib_blend", "netlib_share2b", "netlib_sc105", "netlib_stocfor1", "netlib_scagr7",
    "netlib_share1b", "netlib_israel", "netlib_scsd1", "netlib_lotfi", "netlib_beaconfd",
    "netlib_sc205", "netlib_brandy", "netlib_forplan", "netlib_e226", "netlib_vtp.base",
    "netlib_scsd6", "netlib_bandm", "netlib_scfxm1", "netlib_sctap1", "netlib_grow7",
    "netlib_scorpion", "netlib_agg", "netlib_capri", "netlib_agg2", "netlib_agg3"
]

# second largest problem
BIG_1_PROBLEM = ["netlib_ganges"]

# first and 3rd largest problems
BIG_2_PROBLEMS = ["netlib_stocfor2", "netlib_sierra"]

# Solver Groups
SOLVERS_BM = ["bm_v5", "bm_v8", "bm_v10", "bm_v11"]
SOLVERS_TP = ["tp_v1"]

# -----------------------------------------------------------------------------
# BENCHMARK SUITES
# -----------------------------------------------------------------------------

SUITES = {
    # -------------------------------------------------------------------------
    # PART 1: Big M Implementations
    # -------------------------------------------------------------------------
    "bm_small_30": {
        "description": "Big M: 30 Small problems",
        "problems": SMALL_30_PROBLEMS,
        "solvers": SOLVERS_BM,
        "repetitions": 5,
        "warmup_iterations": 10,
        "warmup_problem": "netlib_afiro",
    },
    "bm_large_1": {
        "description": "Big M: 1 Huge problem (Sierra)",
        "problems": BIG_1_PROBLEM,
        "solvers": SOLVERS_BM,
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
    },
    "bm_large_2": {
        "description": "Big M: 2 Large problems (Stocfor2, Ganges)",
        "problems": BIG_2_PROBLEMS,
        "solvers": SOLVERS_BM,
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
    },
    "bm_stocfor2_remaining": {
        "description": "Big M: stocfor2 with v8/v10/v11 (v5 already done)",
        "problems": ["netlib_stocfor2"],
        "solvers": ["bm_v8", "bm_v10", "bm_v11"],
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
    },
    # -------------------------------------------------------------------------
    # PART 2: Two Phase Implementations
    # -------------------------------------------------------------------------
    "tp_small_30": {
        "description": "Two Phase: 30 Small problems",
        "problems": SMALL_30_PROBLEMS,
        "solvers": SOLVERS_TP,
        "repetitions": 5,
        "warmup_iterations": 0,
        "warmup_problem": "netlib_afiro",
    },
    "tp_large_1": {
        "description": "Two Phase: 1 Huge problem (Sierra)",
        "problems": BIG_1_PROBLEM,
        "solvers": SOLVERS_TP,
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
    },
    "tp_large_2": {
        "description": "Two Phase: 2 Large problems (Stocfor2, Ganges)",
        "problems": BIG_2_PROBLEMS,
        "solvers": SOLVERS_TP,
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
    },
    # -------------------------------------------------------------------------
    # PART 3: Analysis/Plotting Suites (for visualization only)
    # -------------------------------------------------------------------------
    "medium_5": {
        "description": "5 medium problems around 800K-1.2M size (for all 5 solvers)",
        "problems": [
            "netlib_scsd8",    # 1,249,359 (3147×397)
            "netlib_scfxm2",   # 1,038,840 (1574×660)
            "netlib_ship04l",  # 1,016,256 (2528×402)
            "netlib_standmps", #   982,051 (1673×587)
            "netlib_scrs8",    #   822,220 (1678×490)
        ],
        "solvers": ["bm_v5", "bm_v8", "bm_v10", "bm_v11", "tp_v1"],
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
        "hardware": "NVIDIA GeForce RTX 3060",
    },
    # -------------------------------------------------------------------------
    # Combined 12 problems (7 large + 5 medium) for plotting
    # -------------------------------------------------------------------------
    "all_12": {
        "description": "12 problems (7 large + 5 medium) - all 5 solvers",
        "problems": [
            # Large 7
            "netlib_stocfor2", "netlib_ganges", "netlib_czprob", "netlib_ship12s",
            "netlib_grow22", "netlib_fit1p", "netlib_ship08s",
            # Medium 5
            "netlib_scsd8", "netlib_scfxm2", "netlib_ship04l", "netlib_standmps", "netlib_scrs8",
        ],
        "solvers": ["bm_v5", "bm_v8", "bm_v10", "bm_v11", "tp_v1"],
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
        "hardware": "NVIDIA GeForce RTX 3060",
    },
    "all_12_v5": {
        "description": "12 problems - v5 only",
        "problems": [
            "netlib_stocfor2", "netlib_ganges", "netlib_czprob", "netlib_ship12s",
            "netlib_grow22", "netlib_fit1p", "netlib_ship08s",
            "netlib_scsd8", "netlib_scfxm2", "netlib_ship04l", "netlib_standmps", "netlib_scrs8",
        ],
        "solvers": ["bm_v5"],
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
        "hardware": "NVIDIA GeForce RTX 3060",
    },
    "all_12_v5_v8": {
        "description": "12 problems - v5 and v8",
        "problems": [
            "netlib_stocfor2", "netlib_ganges", "netlib_czprob", "netlib_ship12s",
            "netlib_grow22", "netlib_fit1p", "netlib_ship08s",
            "netlib_scsd8", "netlib_scfxm2", "netlib_ship04l", "netlib_standmps", "netlib_scrs8",
        ],
        "solvers": ["bm_v5", "bm_v8"],
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
        "hardware": "NVIDIA GeForce RTX 3060",
    },
    "all_12_v5_v8_v10": {
        "description": "12 problems - v5, v8, v10",
        "problems": [
            "netlib_stocfor2", "netlib_ganges", "netlib_czprob", "netlib_ship12s",
            "netlib_grow22", "netlib_fit1p", "netlib_ship08s",
            "netlib_scsd8", "netlib_scfxm2", "netlib_ship04l", "netlib_standmps", "netlib_scrs8",
        ],
        "solvers": ["bm_v5", "bm_v8", "bm_v10"],
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
        "hardware": "NVIDIA GeForce RTX 3060",
    },
    "all_12_v5_v8_v10_v11": {
        "description": "12 problems - v5, v8, v10, v11",
        "problems": [
            "netlib_stocfor2", "netlib_ganges", "netlib_czprob", "netlib_ship12s",
            "netlib_grow22", "netlib_fit1p", "netlib_ship08s",
            "netlib_scsd8", "netlib_scfxm2", "netlib_ship04l", "netlib_standmps", "netlib_scrs8",
        ],
        "solvers": ["bm_v5", "bm_v8", "bm_v10", "bm_v11"],
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
        "hardware": "NVIDIA GeForce RTX 3060",
    },
    "next_5_large": {
        "description": "Next 5 largest problems without results",
        "problems": ["netlib_czprob", "netlib_ship12s", "netlib_grow22", "netlib_fit1p", "netlib_ship08s"],
        "solvers": ["bm_v5", "bm_v8", "bm_v10", "bm_v11", "tp_v1"],
        "repetitions": 3,
        "warmup_iterations": 0,
        "warmup_problem": None,
        "hardware": "NVIDIA GeForce RTX 3060",
    },
    "gurobi_all": {
        "description": "Gurobi on all problems (local baseline)",
        "problems": [
            "netlib_afiro", "netlib_sc50a", "netlib_sc50b", "netlib_kb2", "netlib_adlittle",
            "netlib_blend", "netlib_share2b", "netlib_sc105", "netlib_stocfor1", "netlib_scagr7",
            "netlib_share1b", "netlib_israel", "netlib_scsd1", "netlib_lotfi", "netlib_beaconfd",
            "netlib_sc205", "netlib_brandy", "netlib_forplan", "netlib_e226", "netlib_vtp.base",
            "netlib_scsd6", "netlib_bandm", "netlib_scfxm1", "netlib_sctap1", "netlib_grow7",
            "netlib_scorpion", "netlib_agg", "netlib_capri", "netlib_agg2", "netlib_agg3",
            "netlib_stair", "netlib_scagr25", "netlib_fffff800", "netlib_ship04s", "netlib_standata",
            "netlib_standgub", "netlib_scrs8", "netlib_standmps", "netlib_ship04l", "netlib_scfxm2",
            "netlib_scsd8", "netlib_grow15", "netlib_gfrd-pnc", "netlib_fit1d", "netlib_ship08s",
            "netlib_scfxm3", "netlib_fit1p", "netlib_grow22", "netlib_ship12s", "netlib_czprob",
            "netlib_ganges", "netlib_stocfor2", "netlib_sierra",
        ],
        "solvers": ["gurobi"],
        "repetitions": 10,
        "warmup_iterations": 20,
        "warmup_problem": "netlib_afiro",
        "hardware": "Apple M4 Max CPU",
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
    # Big-M solvers (warm colors progression)
    "bm_v5": "#E63946",   # Red (baseline Big-M)
    "bm_v8": "#F4A261",   # Orange (full GPU)
    "bm_v10": "#E9C46A",  # Yellow (Sherman-Morris)
    "bm_v11": "#2A9D8F",  # Teal (sparse)
    # Two-Phase solvers
    "tp_v1": "#264653",   # Dark blue
    # cuda_slow solvers (legacy)
    "cuda_slow_v1_cpu": "#E63946",  # Red
    "cuda_slow_v1_lu": "#F4A261",  # Orange
    "cuda_slow_v2": "#E9C46A",  # Yellow
    "cuda_slow_v3": "#2A9D8F",  # Teal
    "cuda_slow_v5": "#264653",  # Dark blue
    "cuda_slow_v6": "#8338EC",  # Purple
    "cuda_slow_v7": "#FF006E",  # Pink
    "cuda_slow_v8": "#3A86FF",  # Blue
    # cuda_fast solvers (legacy)
    "cuda_fast_v1": "#06D6A0",  # Mint
    "cuda_fast_v2": "#118AB2",  # Ocean blue
    "cuda_fast_v3": "#073B4C",  # Dark teal
    "cuda_fast_v4": "#EF476F",  # Coral
    "cuda_fast_v1_fix": "#FFD166",  # Gold
    "cuda_fast_quadratic": "#9B5DE5",  # Lavender
}
