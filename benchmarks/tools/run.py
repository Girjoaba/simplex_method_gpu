"""
Execute benchmarks for a single solver on a single problem with multiple repetitions.

This module handles the low-level execution:
- Running solver binary N times
- Measuring wall-clock time
- Parsing solver output
- Validating correctness
- Collecting measurements
"""

import subprocess
import time
import sys
import os
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.util import parse_solver_output, validate_optimum, format_time, format_ci


def benchmark_problem(solver_binary: str, problem_file: str, problem_file_ending: str, problem_name: str,
                     expected_optimum: float, num_repetitions: int = 100,
                     verbose: bool = True, input_method: str = 'stdin') -> List[Dict]:
    """
    Run solver on a single problem multiple times and collect measurements.

    Args:
        solver_binary: Path to solver executable
        problem_file: Path to problem file (canonical format)
        problem_name: Name of problem (for output)
        expected_optimum: Expected optimal value (for validation)
        num_repetitions: Number of times to run solver
        verbose: Print progress information
        input_method: How to pass problem to solver ('stdin' or 'file_arg')

    Returns:
        List of measurement dictionaries, one per run

    Raises:
        FileNotFoundError: If solver binary or problem file doesn't exist
        RuntimeError: If solver fails or produces incorrect result
    """
    # Very ugly fix...
    problem_file = problem_file + problem_file_ending

    # Parse solver command (may be "python3 script.py" or just "binary")
    solver_cmd = solver_binary.split()

    # Validate inputs
    if not os.path.exists(solver_cmd[0]) and not any(os.path.exists(os.path.join(p, solver_cmd[0])) for p in os.environ.get('PATH', '').split(':')):
        # Check if it's in PATH (like python3)
        import shutil
        if not shutil.which(solver_cmd[0]):
            raise FileNotFoundError(f"Solver binary not found: {solver_cmd[0]}")
    if len(solver_cmd) > 1 and not os.path.exists(solver_cmd[-1]):
        raise FileNotFoundError(f"Solver script not found: {solver_cmd[-1]}")
    if not os.path.exists(problem_file):
        raise FileNotFoundError(f"Problem file not found: {problem_file}")

    measurements = []
    solver_name = os.path.splitext(os.path.basename(solver_binary))[0]

    if verbose:
        print(f"Running {solver_name} on {problem_name} ({num_repetitions} repetitions)...", end='', flush=True)

    for run_id in range(num_repetitions):
        # Measure wall-clock time
        start_time = time.perf_counter()

        try:
            if input_method == 'stdin':
                with open(problem_file, 'r') as f:
                    result = subprocess.run(
                        solver_cmd,
                        stdin=f,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
            else:  # file_arg
                result = subprocess.run(
                    solver_cmd + [problem_file],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Solver timed out on {problem_name} (run {run_id})")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Check for errors
        if result.returncode != 0:
            raise RuntimeError(f"Solver failed on {problem_name} (run {run_id}):\n{result.stderr}")

        # Parse output
        optimum, iterations = parse_solver_output(result.stdout)

        if optimum is None:
            raise RuntimeError(f"Failed to parse solver output on {problem_name} (run {run_id}):\n{result.stdout}")

        # Validate correctness
        if not validate_optimum(optimum, expected_optimum):
            raise RuntimeError(
                f"Incorrect optimum on {problem_name} (run {run_id}): "
                f"got {optimum}, expected {expected_optimum}"
            )

        # Record measurement
        measurements.append({
            'solver': solver_name,
            'problem': problem_name,
            'optimum': optimum,
            'iterations': iterations,
            'time_sec': elapsed_time,
            'run_id': run_id
        })

        # Progress indicator
        if verbose and (run_id + 1) % max(1, num_repetitions // 10) == 0:
            print(f" {run_id + 1}/{num_repetitions}", end='', flush=True)

    if verbose:
        # Compute quick statistics for feedback
        times = [m['time_sec'] for m in measurements]
        mean_time = sum(times) / len(times)
        print(f" ✓ (mean: {format_time(mean_time)})")

    return measurements


def run_single_benchmark(solver_name: str, problem_name: str,
                        num_repetitions: int = 100, verbose: bool = True) -> List[Dict]:
    """
    Convenience function that loads config and runs benchmark.

    Args:
        solver_name: Name of solver from config.SOLVERS
        problem_name: Name of problem from config.PROBLEMS
        num_repetitions: Number of repetitions
        verbose: Print progress

    Returns:
        List of measurements
    """
    from config import SOLVERS, PROBLEMS

    if solver_name not in SOLVERS:
        raise ValueError(f"Unknown solver: {solver_name}. Available: {list(SOLVERS.keys())}")
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown problem: {problem_name}. Available: {list(PROBLEMS.keys())}")

    solver = SOLVERS[solver_name]
    problem = PROBLEMS[problem_name]

    # Determine file extension based on solver type
    # bm_* and gurobi use .canonical, tp_* uses .twophase
    if solver_name.startswith('tp'):
        problem_file_ending = ".twophase"
    else:
        problem_file_ending = ".canonical"

    measurements = benchmark_problem(
        solver_binary=solver['binary'],
        problem_file=problem['file'],
        problem_file_ending=problem_file_ending,
        problem_name=problem_name,
        expected_optimum=problem['expected_optimum'],
        num_repetitions=num_repetitions,
        verbose=verbose,
        input_method=solver.get('input_method', 'stdin')
    )

    # Add problem metadata
    for m in measurements:
        m['m'] = problem['m']
        m['n'] = problem['n']

    return measurements
