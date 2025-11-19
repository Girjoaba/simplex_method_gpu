#!/usr/bin/env python3
"""
Benchmark CLI for Simplex solver performance testing.

Commands:
  run <solver> <problem>           Run single benchmark (single solver, single problem)
  suite <suite_name>               Run benchmark suite (multiple problems, multiple solvers)
  analyze <solvers...>             Generate plots from measurements
  list                             List available problems, solvers, and suites

Examples:
  python bench.py run glop sample
  python bench.py suite baseline
  python bench.py analyze glop --suite baseline
  python bench.py list
"""

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROBLEMS, SOLVERS, SUITES
from tools.run import run_single_benchmark
from tools.suite import run_suite, list_suites
from tools.analyze import create_time_vs_problem_plot, print_summary_table
from tools.util import save_measurements


def cmd_run(args):
    """Run single benchmark (one solver, one problem)."""
    print(f"Running {args.solver} on {args.problem}...")
    print()

    measurements = run_single_benchmark(
        solver_name=args.solver,
        problem_name=args.problem,
        num_repetitions=args.repetitions,
        verbose=True
    )

    # Save measurements
    output_dir = os.path.join(args.measurements_dir, args.solver)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.problem}.tsv")

    save_measurements(measurements, output_file)
    print(f"Saved measurements to: {output_file}")


def cmd_suite(args):
    """Run benchmark suite."""
    run_suite(
        suite_name=args.suite_name,
        measurements_dir=args.measurements_dir,
        force=args.force,
        verbose=True
    )


def cmd_analyze(args):
    """Generate plots from measurements."""
    if not args.solvers:
        print("Error: Must specify at least one solver")
        print("Example: python bench.py analyze glop --suite baseline")
        sys.exit(1)

    if not args.suite:
        print("Error: Must specify --suite")
        print("Example: python bench.py analyze glop --suite baseline")
        sys.exit(1)

    print(f"Analyzing measurements for: {', '.join(args.solvers)}")
    print(f"Suite: {args.suite}")
    print()

    # Print summary table
    print_summary_table(
        solver_names=args.solvers,
        suite_name=args.suite,
        measurements_dir=args.measurements_dir
    )

    # Generate plots
    create_time_vs_problem_plot(
        solver_names=args.solvers,
        suite_name=args.suite,
        measurements_dir=args.measurements_dir,
        plots_dir=args.plots_dir,
        show_iterations=(len(args.solvers) == 1)
    )

    print()
    print(f"Analysis complete! Check {args.plots_dir}/ for plots.")


def cmd_list(args):
    """List available problems, solvers, and suites."""
    print()
    print("=" * 70)
    print("AVAILABLE PROBLEMS")
    print("=" * 70)
    for name, problem in PROBLEMS.items():
        print(f"  {name:<12} {problem['m']}×{problem['n']:<10} {problem['description']}")

    print()
    print("=" * 70)
    print("AVAILABLE SOLVERS")
    print("=" * 70)
    for name, solver in SOLVERS.items():
        print(f"  {name:<12} {solver['description']}")
        print(f"               Binary: {solver['binary']}")

    print()
    print("=" * 70)
    print("AVAILABLE SUITES")
    print("=" * 70)
    list_suites(verbose=True)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark infrastructure for Simplex solver performance testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Global options
    parser.add_argument('--measurements-dir', default='measurements',
                       help='Directory to store/read measurements (default: measurements/)')
    parser.add_argument('--plots-dir', default='plots',
                       help='Directory to save plots (default: plots/)')
    parser.add_argument('--repetitions', type=int, default=100,
                       help='Number of repetitions per problem (default: 100)')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Command: run
    parser_run = subparsers.add_parser('run', help='Run single benchmark')
    parser_run.add_argument('solver', choices=list(SOLVERS.keys()),
                           help='Solver to benchmark')
    parser_run.add_argument('problem', choices=list(PROBLEMS.keys()),
                           help='Problem to solve')
    parser_run.set_defaults(func=cmd_run)

    # Command: suite
    parser_suite = subparsers.add_parser('suite', help='Run benchmark suite')
    parser_suite.add_argument('suite_name', choices=list(SUITES.keys()),
                             help='Suite to run')
    parser_suite.add_argument('--force', action='store_true',
                             help='Re-run even if measurements exist')
    parser_suite.set_defaults(func=cmd_suite)

    # Command: analyze
    parser_analyze = subparsers.add_parser('analyze', help='Generate plots from measurements')
    parser_analyze.add_argument('solvers', nargs='+',
                               help='Solvers to analyze (e.g., glop gurobi)')
    parser_analyze.add_argument('--suite', required=True, choices=list(SUITES.keys()),
                               help='Suite to analyze')
    parser_analyze.set_defaults(func=cmd_analyze)

    # Command: list
    parser_list = subparsers.add_parser('list', help='List available problems, solvers, and suites')
    parser_list.set_defaults(func=cmd_list)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
