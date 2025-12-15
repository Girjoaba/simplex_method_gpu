# Simplex Solver Benchmarking Infrastructure

## Quick Start

**Key files:**
- `bench.py` — CLI entry point for all commands
- `config.py` — Define problems, solvers, and benchmark suites
- `tools/run.py` — Run a solver on one problem N times
- `tools/suite.py` — Run a suite (multiple problems × solvers)
- `tools/analyze.py` — Generate plots from measurements
- `tools/util.py` — Parsing, statistics, file I/O helpers

```bash
cd benchmarks
pip install -r requirements.txt

# List available suites, problems, and solvers
python bench.py list

# Run a benchmark suite
python bench.py suite medium_5

# Generate plots
python bench.py analyze bm_v5 bm_v8 --suite medium_5

# View results
open plots/medium_5_time.pdf
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
├── plots/               # Generated plots (PDF + PNG + data.tsv)
└── tables/              # Generated markdown tables
```

## Commands

### Run Single Benchmark

```bash
python bench.py run <solver> <problem>

# Example: Benchmark bm_v5 on netlib_afiro (100 reps)
python bench.py run bm_v5 netlib_afiro

# Custom repetitions
python bench.py run bm_v5 netlib_afiro --repetitions 50
```

### Run Benchmark Suite

```bash
python bench.py suite <suite_name> [--force]

# Example: Run medium_5 suite
python bench.py suite medium_5

# Re-run even if measurements exist
python bench.py suite medium_5 --force
```

### Analyze and Plot

```bash
python bench.py analyze <solvers...> --suite <suite_name>

# Example: Analyze single solver
python bench.py analyze bm_v5 --suite all_12

# Compare multiple solvers
python bench.py analyze bm_v5 bm_v8 bm_v10 bm_v11 tp_v1 --suite all_12
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
        'm': 100,              # rows
        'n': 200,              # columns
        'size': 20000,         # m * n (for x-axis ordering)
        'file': os.path.join(PROJECT_ROOT, 'benchmarks/input/new_problem'),
        'expected_optimum': 42.0,
        'description': 'Description of problem'
    }
}
```

Note: File path should omit extension. The system appends `.canonical` or `.twophase` based on solver type.

### Adding New Solvers

Edit `config.py`:

```python
SOLVERS = {
    'my_solver': {
        'binary': os.path.join(PROJECT_ROOT, 'bin_solver/my_solver.out'),
        'description': 'My solver description',
        'type': 'parallel',        # or 'serial'
        'input_method': 'stdin',   # or 'file_arg'
    }
}
```

Solver requirements:
- `input_method: 'stdin'`: Read problem from stdin
- `input_method: 'file_arg'`: Accept problem file as command-line argument
- Output format: `Optimum found: <value>` (iterations optional)

### Creating New Suites

Edit `config.py`:

```python
SUITES = {
    'custom_suite': {
        'description': 'Custom benchmark suite',
        'problems': ['netlib_afiro', 'netlib_blend', 'netlib_sc50a'],
        'solvers': ['bm_v5', 'bm_v8'],
        'repetitions': 3,
        'warmup_iterations': 10,       # 0 to disable
        'warmup_problem': 'netlib_afiro',
        'hardware': 'NVIDIA GeForce RTX 3060',  # for plot title
    }
}
```

## Output Format

### Measurements TSV

`measurements/{solver}/{problem}.tsv`:

```
solver  problem       m     n     optimum               iterations  time_sec     run_id
bm_v5   netlib_afiro  59    27    4.6475314285714285e+02            0.00234      0
bm_v5   netlib_afiro  59    27    4.6475314285714285e+02            0.00231      1
```

### Plot Data TSV

`plots/{suite_name}_time_data.tsv`:

```
problem       m    n    size   bm_v5_mean  bm_v5_ci_lower  bm_v5_ci_upper  ...
netlib_afiro  59   27   1593   0.00232     0.00229         0.00235         ...
```
