"""
Utility functions for benchmark infrastructure.

Provides:
- Solver output parsing
- Statistical calculations (mean, CI using bootstrap)
- TSV I/O operations
- Data validation
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def parse_solver_output(stdout: str) -> Tuple[Optional[float], Optional[int]]:
    """
    Parse solver output to extract optimum value and iteration count.

    Expected format:
        Optimum found: 9.0000000000000000
        Iterations: 2

    Returns:
        (optimum, iterations) or (None, None) if parsing fails
    """
    optimum = None
    iterations = None

    # Parse optimum value
    optimum_match = re.search(r'Optimum found:\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', stdout)
    if optimum_match:
        optimum = float(optimum_match.group(1))

    # Parse iteration count
    iter_match = re.search(r'Iterations:\s+(\d+)', stdout)
    if iter_match:
        iterations = int(iter_match.group(1))

    return optimum, iterations


def validate_optimum(computed: float, expected: float, rtol: float = 1e-4) -> bool:
    """
    Validate that computed optimum matches expected value within relative tolerance.

    Args:
        computed: Optimum value from solver
        expected: Expected optimum from groundtruth
        rtol: Relative tolerance (default: 1e-4, same as test script)

    Returns:
        True if values match within tolerance
    """
    if expected == 0:
        return abs(computed) < 1e-10
    return abs((computed - expected) / expected) < rtol


def calculate_bootstrap_ci(values: np.ndarray, confidence: float = 0.95,
                           n_bootstrap: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """
    Calculate confidence interval using bootstrap resampling.

    This is a non-parametric method that doesn't assume normality of the data,
    making it suitable for timing measurements which may be skewed.

    Args:
        values: Array of measurements
        confidence: Confidence level (default: 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        (lower_bound, upper_bound) of confidence interval
    """
    rng = np.random.RandomState(seed)
    bootstrap_means = []

    n = len(values)
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return lower, upper


def aggregate_statistics(measurements_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics from raw measurements.

    Args:
        measurements_df: DataFrame with 'time_sec' and 'iterations' columns

    Returns:
        Dictionary with keys: mean, median, std, ci_lower, ci_upper, min, max,
                             mean_iterations, median_iterations
    """
    times = measurements_df['time_sec'].values
    iterations = measurements_df['iterations'].values

    # Time statistics (use arithmetic mean for costs)
    mean_time = np.mean(times)
    median_time = np.median(times)
    std_time = np.std(times, ddof=1)  # Sample std deviation
    min_time = np.min(times)
    max_time = np.max(times)

    # Confidence interval
    ci_lower, ci_upper = calculate_bootstrap_ci(times, confidence=0.95)

    # Iteration statistics (handle None values)
    valid_iters = [x for x in iterations if x is not None]
    if valid_iters:
        mean_iters = np.mean(valid_iters)
        median_iters = np.median(valid_iters)
    else:
        mean_iters = None
        median_iters = None

    return {
        'mean': mean_time,
        'median': median_time,
        'std': std_time,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'min': min_time,
        'max': max_time,
        'mean_iterations': mean_iters,
        'median_iterations': median_iters
    }


def read_measurements(filepath: str) -> pd.DataFrame:
    """
    Read measurements from TSV file.

    Args:
        filepath: Path to .tsv file

    Returns:
        pandas DataFrame with measurement data
    """
    return pd.read_csv(filepath, sep='\t')


def save_measurements(data: List[Dict], filepath: str, metadata: Optional[Dict] = None):
    """
    Save measurements to TSV file.

    Args:
        data: List of measurement dictionaries
        filepath: Output path for .tsv file
        metadata: Optional dict with problem metadata (m, n, etc.)
    """
    df = pd.DataFrame(data)

    # Add metadata columns if provided
    if metadata:
        for key, value in metadata.items():
            if key not in df.columns:
                df[key] = value

    # Ensure consistent column ordering
    column_order = ['solver', 'problem', 'm', 'n', 'optimum', 'iterations', 'time_sec', 'run_id']
    existing_cols = [col for col in column_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in column_order]
    df = df[existing_cols + other_cols]

    # Write to file
    df.to_csv(filepath, sep='\t', index=False, float_format='%.16e')


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1.23ms", "45.6μs")
    """
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f}ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:.2f}μs"
    else:
        return f"{seconds * 1e9:.2f}ns"


def format_ci(lower: float, upper: float) -> str:
    """
    Format confidence interval in human-readable format.

    Args:
        lower: Lower bound
        upper: Upper bound

    Returns:
        Formatted string (e.g., "[1.2ms, 1.5ms]")
    """
    return f"[{format_time(lower)}, {format_time(upper)}]"
