# Simplex Solver Benchmarking Infrastructure

Directly based off of my ASL benchmarking infra

## Quick Start

```bash
# 1. Install dependencies
cd benchmarks
pip install -r requirements.txt

# 2. List available suites, problems, and solvers
python bench.py list

# 3. Run baseline suite (5 problems × 100 reps each)
python bench.py suite baseline

# 4. Generate plots
python bench.py analyze glop --suite baseline

# 5. View results
open plots/baseline_time.pdf
```

## Architecture

```
benchmarks/
├── bench.py              # CLI entry point
├── config.py             # Problem/solver/suite definitions
├── requirements.txt      # Python dependencies
├── tools/
│   ├── run.py           # Execute single problem with repetitions
│   ├── suite.py         # Orchestrate multiple problems
│   ├── analyze.py       # Statistical analysis and plotting
│   └── util.py          # Helper functions (parsing, stats, I/O)
├── measurements/         # Raw timing data (TSV files)
└── plots/               # Generated plots (PDF + PNG + data.tsv)
```

## Commands

### Run Single Benchmark

```bash
python bench.py run <solver> <problem>

# Example: Benchmark GLOP on sample problem (100 reps)
python bench.py run glop sample

# Custom repetitions
python bench.py run glop sample --repetitions 1000
```

### Run Benchmark Suite

```bash
python bench.py suite <suite_name> [--force]

# Example: Run baseline suite
python bench.py suite baseline

# Re-run even if measurements exist
python bench.py suite baseline --force
```

### Analyze and Plot

```bash
python bench.py analyze <solvers...> --suite <suite_name>

# Example: Analyze GLOP baseline
python bench.py analyze glop --suite baseline

# Compare multiple solvers (when available)
python bench.py analyze glop gurobi --suite baseline
```

### List Available Components

```bash
python bench.py list
```

## Configuration

### Adding New Problems

Edit `config.py`:

```python
PROBLEMS = {
    'new_problem': {
        'm': 100,
        'n': 200,
        'size': 20000,
        'file': os.path.join(PROJECT_ROOT, 'test/input/new_problem.canonical'),
        'expected_optimum': 42.0,
        'description': 'Description of problem'
    }
}
```

### Adding New Solvers

Edit `config.py`:

```python
SOLVERS = {
    'gurobi': {
        'binary': os.path.join(PROJECT_ROOT, 'gurobi_baseline/gurobi_canonical'),
        'description': 'Gurobi Optimizer',
        'type': 'serial'
    }
}
```

The solver binary must:
- Accept canonical file as command-line argument
- Output format: `Optimum found: <value>` and `Iterations: <count>`

### Creating New Suites

Edit `config.py`:

```python
SUITES = {
    'custom_suite': {
        'description': 'Custom benchmark suite',
        'problems': ['sample', 'afiro', 'e226'],
        'solvers': ['glop', 'gurobi'],
        'repetitions': 200
    }
}
```

## Output Format

### Measurements TSV

`measurements/{solver}/{problem}.tsv`:

```
solver  problem  m    n    optimum              iterations  time_sec    run_id
glop    sample   2    4    9.0000000000000000   2          0.000143    0
glop    sample   2    4    9.0000000000000000   2          0.000139    1
...
```

### Plot Data TSV

`plots/{suite_name}_time_data.tsv`:

```
problem  m    n    size   glop_mean  glop_ci_lower  glop_ci_upper  glop_std  glop_median  glop_mean_iterations
sample   2    4    8      0.000141   0.000139       0.000143       0.000002  0.000141     2.0
...
```

## Statistical Methodology

Following best practices from Torsten presentation/paper:

1. When publishing parallel speedup, state whether the base case is a single parallel process or the best serial execution, and report the base case’s absolute performance. (p.3)
2. If you report only subsets (benchmarks, kernels, or resources), explain why; don’t cherry-pick. (p.3)
3. Use the arithmetic mean only for costs (e.g., time); use the harmonic mean for rates (e.g., flop/s). (pp.3–4)
4. Avoid averaging ratios; summarize the underlying costs or rates instead. If only ratios are available, use the geometric mean (with caution). (p.4)
5. Say whether measurements are deterministic. For nondeterministic data, report confidence intervals. (p.4)
6. Don’t assume normality of the data (e.g., just because n is large) without diagnostic checks. (p.5)
7. Compare nondeterministic data statistically (e.g., non-overlapping CIs, ANOVA/Kruskal-Wallis), not by eyeballing means/medians alone. (p.6)
8. Check whether mean/median are the right summaries; for some cases (e.g., worst-case latency) report other percentiles. (p.6)
9. Document all varying factors, their levels, and the full experimental setup (hardware, software, inputs, methods) to enable interpretability/reproducibility. (p.8)
10. For parallel timing, report the measurement, any synchronization, and how you summarize across processes. (p.8)
11. Where possible, show upper-bound models (e.g., ideal scaling, Amdahl/overheads) to put results in context. (p.9)
12. Plot enough information to interpret results (incl. uncertainty when needed). Only connect points if interpolation is valid. (p.10) 
