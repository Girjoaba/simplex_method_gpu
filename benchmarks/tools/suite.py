"""
Orchestrate benchmark suites across multiple problems and solvers.

This module coordinates running benchmarks:
- Load suite configuration
- Run benchmarks for all problems × solvers
- Save measurements to TSV files
- Skip already-completed measurements (unless --force)
- Display progress and summary statistics
"""

import os
import sys
import subprocess
import time
from typing import List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SUITES, SOLVERS, PROBLEMS
from tools.run import benchmark_problem
from tools.util import save_measurements, aggregate_statistics, format_time, format_ci
import pandas as pd


def run_warmup(suite, solver_name: str, verbose: bool = True) -> Tuple[float, float]:
    """
    Run warmup iterations to eliminate cold-start effects.

    Warmup measurements are NOT saved. Returns first-10 vs last-10 comparison.

    Args:
        suite: Suite configuration dict
        solver_name: Name of solver to warmup
        verbose: Print warmup progress

    Returns:
        (first_10_avg, last_10_avg) in seconds
    """
    warmup_problem = PROBLEMS[suite['warmup_problem']]
    solver = SOLVERS[solver_name]
    n_warmup = suite['warmup_iterations']

    if verbose:
        print()
        print("=" * 70)
        print(f"Warmup Phase ({n_warmup} iterations on {suite['warmup_problem']})")
        print("=" * 70)
        print("Warming up...", end='', flush=True)

    times = []

    for i in range(n_warmup):
        # Run solver and measure time
        start_time = time.perf_counter()

        # Use stdin for input (matching run.py behavior)
        with open(warmup_problem['file'], 'r') as f:
            result = subprocess.run(
                [solver['binary']],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=300
            )

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times.append(elapsed)

        # Progress indicator every 10 iterations
        if verbose and (i + 1) % 10 == 0:
            print(f" {i + 1}", end='', flush=True)

        # Check for errors (but don't validate optimum - faster warmup)
        if result.returncode != 0:
            raise RuntimeError(f"Solver failed during warmup: {result.stderr}")

    # Compute first 10 vs last 10 comparison
    first_10_avg = sum(times[:10]) / 10
    last_10_avg = sum(times[-10:]) / 10
    diff = last_10_avg - first_10_avg
    pct_change = (diff / first_10_avg) * 100

    if verbose:
        print(" ✓")
        print()
        print(f"First 10 runs:  avg = {first_10_avg * 1000:.1f}ms")
        print(f"Last 10 runs:   avg = {last_10_avg * 1000:.1f}ms")
        print(f"Difference:     {diff * 1000:+.1f}ms ({pct_change:+.1f}%)", end='')

        if abs(pct_change) < 1.0:
            print(" ✓ Steady state achieved")
        elif pct_change < 0:
            print(" ← Cold start eliminated")
        else:
            print(" ⚠ System may not be warmed up")

        print()
        print("System ready for benchmarking.")
        print("=" * 70)

    return first_10_avg, last_10_avg


def run_suite(suite_name: str, measurements_dir: str = 'measurements',
              force: bool = False, verbose: bool = True):
    """
    Run a complete benchmark suite.

    Args:
        suite_name: Name of suite from config.SUITES
        measurements_dir: Directory to save measurements
        force: If True, re-run even if measurements exist
        verbose: Print progress information

    Raises:
        ValueError: If suite_name is unknown
    """
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite: {suite_name}. Available: {list(SUITES.keys())}")

    suite = SUITES[suite_name]

    if verbose:
        print("=" * 70)
        print(f"Running suite: {suite_name}")
        print(f"Description: {suite['description']}")
        print(f"Problems: {len(suite['problems'])}")
        print(f"Solvers: {', '.join(suite['solvers'])}")
        print(f"Repetitions per problem: {suite['repetitions']}")
        print("=" * 70)

    # Run warmup phase if configured (use first solver only)
    if 'warmup_iterations' in suite and suite['warmup_iterations'] > 0:
        first_solver = suite['solvers'][0]
        run_warmup(suite, first_solver, verbose=verbose)

    if verbose:
        print()
        print("=" * 70)
        print("Benchmark Phase")
        print("=" * 70)
        print()

    # Track statistics
    total_problems = len(suite['problems']) * len(suite['solvers'])
    completed = 0
    skipped = 0

    # Run benchmarks for each solver × problem combination
    for solver_name in suite['solvers']:
        solver = SOLVERS[solver_name]

        # Create output directory for this solver
        solver_dir = os.path.join(measurements_dir, solver_name)
        os.makedirs(solver_dir, exist_ok=True)

        for i, problem_name in enumerate(suite['problems'], 1):
            problem = PROBLEMS[problem_name]

            # Output file path
            output_file = os.path.join(solver_dir, f"{problem_name}.tsv")

            # Check if already measured
            if os.path.exists(output_file) and not force:
                if verbose:
                    print(f"[{i}/{len(suite['problems'])}] {problem_name} ({problem['m']}×{problem['n']})... "
                          f"SKIPPED (already measured, use --force to re-run)")
                skipped += 1
                continue

            # Run benchmark
            try:
                measurements = benchmark_problem(
                    solver_binary=solver['binary'],
                    problem_file=problem['file'],
                    problem_name=problem_name,
                    expected_optimum=problem['expected_optimum'],
                    num_repetitions=suite['repetitions'],
                    verbose=False  # We'll print our own progress
                )

                # Add problem metadata
                for m in measurements:
                    m['m'] = problem['m']
                    m['n'] = problem['n']

                # Save measurements
                save_measurements(measurements, output_file)

                # Display summary statistics
                if verbose:
                    df = pd.DataFrame(measurements)
                    stats = aggregate_statistics(df)
                    print(f"[{i}/{len(suite['problems'])}] {problem_name} ({problem['m']}×{problem['n']})... "
                          f"✓ (mean: {format_time(stats['mean'])}, "
                          f"95% CI: {format_ci(stats['ci_lower'], stats['ci_upper'])})")

                completed += 1

            except Exception as e:
                print(f"[{i}/{len(suite['problems'])}] {problem_name} ({problem['m']}×{problem['n']})... "
                      f"❌ FAILED: {e}")
                raise

    # Summary
    if verbose:
        print()
        print("=" * 70)
        print(f"Suite complete!")
        print(f"Total: {total_problems}, Completed: {completed}, Skipped: {skipped}")
        print(f"Measurements saved to: {os.path.abspath(measurements_dir)}/")
        print("=" * 70)


def list_suites(verbose: bool = True) -> List[str]:
    """
    List all available benchmark suites.

    Args:
        verbose: If True, print detailed information

    Returns:
        List of suite names
    """
    if verbose:
        print("Available benchmark suites:")
        print()
        for name, suite in SUITES.items():
            print(f"  {name}:")
            print(f"    Description: {suite['description']}")
            print(f"    Problems: {len(suite['problems'])} ({', '.join(suite['problems'])})")
            print(f"    Solvers: {', '.join(suite['solvers'])}")
            print(f"    Repetitions: {suite['repetitions']}")
            print()

    return list(SUITES.keys())
