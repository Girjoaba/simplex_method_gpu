"""
Analysis and plotting for benchmark results.

This module handles:
- Loading measurements from TSV files
- Computing summary statistics with confidence intervals
- Generating publication-quality plots
- Exporting data alongside plots for reproducibility
"""

import os
import sys
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROBLEMS, SUITES, PLOT_STYLE, SOLVER_COLORS
from tools.util import read_measurements, aggregate_statistics


def load_solver_measurements(solver_name: str, problem_names: List[str],
                             measurements_dir: str = 'measurements') -> Dict[str, pd.DataFrame]:
    """
    Load measurements for a solver across multiple problems.

    Args:
        solver_name: Name of solver
        problem_names: List of problem names
        measurements_dir: Directory containing measurements

    Returns:
        Dictionary mapping problem_name -> DataFrame
    """
    measurements = {}

    for problem_name in problem_names:
        filepath = os.path.join(measurements_dir, solver_name, f"{problem_name}.tsv")

        if not os.path.exists(filepath):
            print(f"Warning: Measurements not found for {solver_name}/{problem_name} at {filepath}")
            continue

        measurements[problem_name] = read_measurements(filepath)

    return measurements


def create_time_vs_problem_plot(solver_names: List[str], suite_name: str,
                                measurements_dir: str = 'measurements',
                                plots_dir: str = 'plots',
                                show_iterations: bool = True):
    """
    Create log-log scatter plot showing mean solve time vs problem size.

    X-axis: Problem size (m×n) on logarithmic scale
    Y-axis: Mean time (seconds) on logarithmic scale
    Error bars: 95% confidence intervals (vertical)
    No interpolation: Points not connected (per style guideline #12)

    Args:
        solver_names: List of solver names to plot
        suite_name: Name of suite (determines which problems to include)
        measurements_dir: Directory with measurement TSV files
        plots_dir: Directory to save plots
        show_iterations: If True, also create iteration count plot
    """
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite: {suite_name}")

    suite = SUITES[suite_name]
    problem_names = suite['problems']

    # Collect statistics for all solvers
    solver_stats = {}
    for solver_name in solver_names:
        measurements = load_solver_measurements(solver_name, problem_names, measurements_dir)

        stats_by_problem = {}
        for problem_name in problem_names:
            if problem_name not in measurements:
                continue
            df = measurements[problem_name]
            stats_by_problem[problem_name] = aggregate_statistics(df)

        solver_stats[solver_name] = stats_by_problem

    # Set up plot style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])

    # Marker styles for different solvers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # Plot scatter points for each solver
    for solver_idx, solver_name in enumerate(solver_names):
        stats = solver_stats[solver_name]

        sizes = []
        means = []
        errors_lower = []
        errors_upper = []
        labels = []

        for problem_name in problem_names:
            if problem_name in stats:
                problem = PROBLEMS[problem_name]
                s = stats[problem_name]

                sizes.append(problem['size'])  # m × n
                means.append(s['mean'])
                # Error bars: distance from mean to CI bounds
                errors_lower.append(s['mean'] - s['ci_lower'])
                errors_upper.append(s['ci_upper'] - s['mean'])
                labels.append(f"{problem_name} ({problem['m']}×{problem['n']})")

        if not sizes:
            continue

        color = SOLVER_COLORS.get(solver_name, f'C{solver_idx}')
        marker = markers[solver_idx % len(markers)]

        # Plot scatter points with error bars (no line connecting them)
        ax.errorbar(sizes, means,
                    yerr=[errors_lower, errors_upper],
                    fmt=marker,  # Marker style, no line
                    color=color,
                    label=solver_name,
                    markersize=8,
                    capsize=4,
                    capthick=1.5,
                    elinewidth=1.5,
                    alpha=0.8)

    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Custom y-axis formatter and ticks for better readability
    from matplotlib.ticker import FuncFormatter, FixedLocator

    def time_formatter(y, pos):
        """Format y-axis as readable time values (ms or s)"""
        if y >= 1.0:
            return f'{y:.1f}s'
        elif y >= 0.01:
            return f'{y*1000:.0f}ms'
        else:
            return f'{y*1000:.1f}ms'

    # Set explicit tick locations at readable intervals
    y_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))

    # Formatting
    ax.set_xlabel('Problem Size, m×n (log scale)', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel('Mean Time (log scale)', fontsize=PLOT_STYLE['label_fontsize'])

    # Title with chip name and repetition/warmup info from suite
    n_problems = len([p for p in problem_names if p in solver_stats.get(solver_names[0], {})])
    title_line1 = 'Simplex Solver on Apple M4 Max CPU'
    if 'warmup_iterations' in suite and 'warmup_problem' in suite:
        warmup_info = f', {suite["warmup_iterations"]}x {suite["warmup_problem"]} warmup'
    else:
        warmup_info = ''
    title_line2 = f'{n_problems} problems, {suite["repetitions"]} reps{warmup_info}'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=PLOT_STYLE['title_fontsize'])

    ax.tick_params(axis='both', labelsize=PLOT_STYLE['tick_fontsize'])
    ax.legend(fontsize=PLOT_STYLE['legend_fontsize'])
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()

    # Save plot
    os.makedirs(plots_dir, exist_ok=True)
    plot_name = f"{suite_name}_time"
    plt.savefig(os.path.join(plots_dir, f"{plot_name}.pdf"), dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, f"{plot_name}.png"), dpi=PLOT_STYLE['dpi'], bbox_inches='tight')

    print(f"Saved plots: {plots_dir}/{plot_name}.{{pdf,png}}")

    # Export data
    export_data = []
    for problem_name in problem_names:
        problem = PROBLEMS[problem_name]
        row = {
            'problem': problem_name,
            'm': problem['m'],
            'n': problem['n'],
            'size': problem['size']
        }
        for solver_name in solver_names:
            if problem_name in solver_stats[solver_name]:
                s = solver_stats[solver_name][problem_name]
                row[f'{solver_name}_mean'] = s['mean']
                row[f'{solver_name}_ci_lower'] = s['ci_lower']
                row[f'{solver_name}_ci_upper'] = s['ci_upper']
                row[f'{solver_name}_std'] = s['std']
                row[f'{solver_name}_median'] = s['median']
                row[f'{solver_name}_mean_iterations'] = s['mean_iterations']
        export_data.append(row)

    export_df = pd.DataFrame(export_data)
    export_file = os.path.join(plots_dir, f"{plot_name}_data.tsv")
    export_df.to_csv(export_file, sep='\t', index=False, float_format='%.16e')
    print(f"Saved data: {export_file}")

    plt.close()

    # Optionally create iteration count plot
    if show_iterations and len(solver_names) == 1:
        create_iterations_plot(solver_names[0], suite_name, measurements_dir, plots_dir)


def create_iterations_plot(solver_name: str, suite_name: str,
                           measurements_dir: str = 'measurements',
                           plots_dir: str = 'plots'):
    """
    Create scatter plot showing iteration counts vs problem size (log-x axis).

    Args:
        solver_name: Name of solver
        suite_name: Name of suite
        measurements_dir: Directory with measurements
        plots_dir: Directory to save plots
    """
    suite = SUITES[suite_name]
    problem_names = suite['problems']

    measurements = load_solver_measurements(solver_name, problem_names, measurements_dir)

    # Collect iteration statistics
    sizes = []
    mean_iters = []
    problem_labels = []

    for problem_name in problem_names:
        if problem_name not in measurements:
            continue

        df = measurements[problem_name]
        stats = aggregate_statistics(df)

        problem = PROBLEMS[problem_name]
        sizes.append(problem['size'])
        mean_iters.append(stats['mean_iterations'])
        problem_labels.append(f"{problem_name} ({problem['m']}×{problem['n']})")

    # Create plot
    fig, ax = plt.subplots(figsize=PLOT_STYLE['figure_size'])

    color = SOLVER_COLORS.get(solver_name, 'C0')

    # Scatter plot with log-x axis
    ax.scatter(sizes, mean_iters, color=color, s=100, alpha=0.8, edgecolors='black', linewidths=1.5)

    # Set logarithmic x-axis
    ax.set_xscale('log')

    ax.set_xlabel('Problem Size, m×n (log scale)', fontsize=PLOT_STYLE['label_fontsize'])
    ax.set_ylabel('Iteration Count', fontsize=PLOT_STYLE['label_fontsize'])

    # Title with chip name and problem count
    n_problems = len(sizes)
    title_line1 = 'Simplex Solver Iterations on Apple M4 Max CPU'
    title_line2 = f'{n_problems} problems'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=PLOT_STYLE['title_fontsize'])

    ax.tick_params(axis='both', labelsize=PLOT_STYLE['tick_fontsize'])
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()

    # Save
    plot_name = f"{suite_name}_iterations_{solver_name}"
    plt.savefig(os.path.join(plots_dir, f"{plot_name}.pdf"), dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, f"{plot_name}.png"), dpi=PLOT_STYLE['dpi'], bbox_inches='tight')

    print(f"Saved iteration plot: {plots_dir}/{plot_name}.{{pdf,png}}")

    plt.close()


def print_summary_table(solver_names: List[str], suite_name: str,
                       measurements_dir: str = 'measurements'):
    """
    Print summary statistics table to console.

    Args:
        solver_names: List of solver names
        suite_name: Name of suite
        measurements_dir: Directory with measurements
    """
    suite = SUITES[suite_name]
    problem_names = suite['problems']

    print()
    print("=" * 90)
    print(f"Summary Statistics: {suite_name}")
    print("=" * 90)
    print()

    for solver_name in solver_names:
        print(f"Solver: {solver_name}")
        print("-" * 90)
        print(f"{'Problem':<12} {'Size':<12} {'Mean Time':<15} {'95% CI':<30} {'Iters':<10}")
        print("-" * 90)

        measurements = load_solver_measurements(solver_name, problem_names, measurements_dir)

        for problem_name in problem_names:
            if problem_name not in measurements:
                continue

            problem = PROBLEMS[problem_name]
            df = measurements[problem_name]
            stats = aggregate_statistics(df)

            size_str = f"{problem['m']}×{problem['n']}"
            time_str = f"{stats['mean']:.6f}s"
            ci_str = f"[{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}]"
            iter_str = f"{stats['mean_iterations']:.1f}"

            print(f"{problem_name:<12} {size_str:<12} {time_str:<15} {ci_str:<30} {iter_str:<10}")

        print()
