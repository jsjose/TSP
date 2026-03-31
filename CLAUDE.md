# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research/benchmarking project implementing and comparing multiple algorithms for the **Traveling Salesman Problem (TSP)**. The project is organized as a Python package. Solvers range from a novel quantum-inspired SPSA approach to classical exact solvers (CPLEX, OR-Tools).

## Repository Structure

```
TSP/
├── algorithms/                 # Production solver package
│   ├── __init__.py             # Re-exports all public solvers
│   ├── spsa.py                 # SingleQubitTSP + two_opt_refinement()
│   ├── genetic.py              # GeneticTSPSolver (PMX crossover, scramble mutation)
│   ├── ortools.py              # ORToolsTSPSolver (Google OR-Tools wrapper)
│   ├── cplex.py                # CPLEXTSPSolver (IBM CPLEX / MTZ formulation)
│   ├── decomposition.py        # solve_tsp_decomposition() (k-means + sub-solve)
│   ├── held_karp.py            # Held-Karp DP solver with MLX GPU support
│   ├── montecarlo.py           # Batch distance calculation via NumPy vectorization
│   └── lkh.py                  # LKH3Solver (Lin-Kernighan-Helsgott wrapper)
├── utils/                      # Shared utilities package
│   ├── __init__.py             # Re-exports all utilities
│   ├── helpers.py              # print_progress, calculate_total_distance, etc.
│   └── loader.py               # TSPLIB file parser; defines TSPInstance dataclass
├── experiments/                # Standalone research/experimentation scripts
│   ├── spsa_standalone.py      # SPSA research script (self-contained)
│   ├── benchmark_advanced.py   # 2nd-order SPSA benchmarking
│   ├── held_karp_dev.py        # Held-Karp experimentation
│   ├── decomposition_demo.py   # K-Means decomposition visualization
│   ├── bruteforce_cuda.py      # CUDA brute-force solver
│   └── bruteforce_mlx.py       # MLX (Apple Silicon) brute-force solver
├── tsp/                        # Legacy solver registry package
│   ├── registry.py             # Solver registry (maps names → classes)
│   ├── base.py                 # Base solver interface
│   └── store.py                # Result storage
├── test_tsplib.py              # Main benchmark runner
├── main.py                     # Entry point — calls run_benchmark()
├── cli.py                      # CLI interface
├── pyproject.toml              # Project metadata and dependencies
├── requirements.txt            # Pinned dev dependencies (pytest, torch, etc.)
├── uv.lock                     # UV lock file
├── .python-version             # Specifies Python 3.12
├── ReadMe.md                   # Algorithm descriptions and benchmark results table
└── tsplib/                     # TSPLIB data files (gitignored, must obtain separately)
    ├── *.tsp                   # Problem instances (burma14, bayg29, att48, …)
    └── solutions               # Optimal costs in "name : cost" format
```

## Environment Setup

**Required:** Python >= 3.11 (project uses 3.12 via `.python-version`)

### Install with UV (preferred)
```bash
uv sync
```

### Install with pip
```bash
pip install -r requirements.txt
```

### Notable dependencies
| Package | Purpose |
|---|---|
| `numpy==2.3.4` | Core array math (pinned exact version) |
| `ortools` | Google OR-Tools TSP solver |
| `cplex` / `docplex` | IBM CPLEX (requires separate license; Python < 3.13) |
| `mlx` | Apple Silicon GPU acceleration |
| `scipy`, `scikit-learn` | Distance metrics, k-means clustering |
| `matplotlib` | Route visualization |
| `black` | Code formatter |
| `pytest` | Test runner |

## Common Commands

### Running the Full Benchmark

```bash
python main.py
# or equivalently:
python test_tsplib.py
```

Runs all solvers against standard TSPLIB instances and prints a results table (cost, time, gap % vs. optimal). TSPLIB data files must be present in `tsplib/` (gitignored).

### CLI: Solve Single Instance

```bash
tsp-cli solve tsplib/berlin52.tsp --solver ortools
tsp-cli solve tsplib/att48.tsp --solver spsa --store
```

Run a single solver on a single instance. Use `--store` to save results to the database.

### CLI: Benchmark Multiple Instances

```bash
tsp-cli benchmark --instances tsplib/burma14.tsp tsplib/att48.tsp
tsp-cli benchmark --instances tsplib/berlin52.tsp --solvers ortools genetic --store
```

Run selected solvers on multiple instances. By default, all eligible solvers run (respecting size thresholds).

### CLI: Query Results

```bash
tsp-cli results
tsp-cli results --instance berlin52 --solver ortools --last 5
```

Query the stored results database. Use `--instance`, `--solver`, and `--last N` to filter.

### Code Formatting

```bash
black algorithms/ utils/ experiments/ *.py
```

Format code with Black (pinned version in pyproject.toml). The project does not currently have automated linting.

## Key Algorithms

### 1. SingleQubitTSP (`algorithms/spsa.py`)
Quantum-inspired heuristic. Each city's "preference weights" are a softmax over an n×n parameter matrix, optimized by SPSA.
- `solve_hybrid()` — Phase 1: 1SPSA (global), Phase 2: 2SPSA (curvature), Phase 3: 2-opt
- `solve_refined()` — Multi-start refined SPSA
- `two_opt_refinement(path, matrix)` — Standalone 2-opt local search
- Distance matrix stored as `self.B`; normalized version as `self.normalized_B`
- Greedy decoding in `decode_path()` ensures valid Hamiltonian cycle

### 2. GeneticTSPSolver (`algorithms/genetic.py`)
Population-based evolutionary solver.
- Partially Mapped Crossover (PMX) for valid permutation offspring
- Scramble mutation for diversity
- Selection from full population (not top-50) to reduce premature convergence

### 3. ORToolsTSPSolver (`algorithms/ortools.py`)
Wrapper around Google OR-Tools `RoutingModel`.
- Uses Guided Local Search metaheuristic
- Configurable time limits (longer for large instances)

### 4. CPLEXTSPSolver (`algorithms/cplex.py`)
Integer programming via IBM CPLEX.
- Miller-Tucker-Zemlin (MTZ) subtour elimination constraints
- Binary edge variables + sequence variables
- Fastest exact solver for small instances (< 0.1s on burma14)

### 5. solve_tsp_decomposition (`algorithms/decomposition.py`)
Divide-and-conquer heuristic.
- k-means clustering splits cities into sub-clusters
- Solves each cluster with Refined SPSA
- Stitches solutions via centroid TSP
- Requires coordinate data (`coords`); skipped when unavailable

### 6. Held-Karp (`algorithms/held_karp.py`)
Exact DP solver: O(n² × 2ⁿ). Only practical for very small instances.
- `solve_tsp_held_karp_mlx()` — Apple Silicon MLX GPU version
- `solve_tsp_held_karp_cpu()` — Pure NumPy CPU version

### 7. LKH3Solver (`algorithms/lkh.py`)

Lin-Kernighan-Helsgott heuristic wrapper. Requires LKH3 binary installation.

- Wrapper around the external LKH3 solver
- Provides high-quality solutions for medium to large instances
- Requires separate LKH3 tool (installed via `pip install lkh`)

## Data Loading (`utils/loader.py`)

```python
import utils.loader as tsp_loader

solutions_map = tsp_loader.load_solutions_file("tsplib/solutions")
instance = tsp_loader.load_tsp("tsplib/berlin52.tsp", solutions_map)
# instance.name, instance.dimension, instance.distance_matrix, instance.optimal_cost, instance.coords
```

Supported EDGE_WEIGHT_TYPEs: `EUC_2D`, `GEO`, `ATT`, `EXPLICIT` (FULL_MATRIX, UPPER_ROW, LOWER_DIAG_ROW).

## Size-Based Algorithm Selection

The benchmark applies size thresholds automatically:

| Instance size | Algorithms run |
|---|---|
| n < 100 | SPSA+2Opt, Genetic, OR-Tools, CPLEX, Decomposition |
| n >= 100 | OR-Tools only (SPSA/Genetic/Decomposition are skipped) |

When adding a new solver, follow this same pattern in `test_tsplib.py`.

## Code Conventions

### Naming
- **Classes:** `PascalCase` — `SingleQubitTSP`, `GeneticTSPSolver`, `TSPInstance`
- **Functions/methods:** `snake_case` — `solve_refined()`, `two_opt_refinement()`
- **Private methods:** leading underscore — `_crossover()`, `_mutate()`
- **Distance matrix:** conventionally stored as `self.B` (academic convention)
- **Problem size:** stored as `self.n`

### Return Interface
All public solver methods return `tuple[list[int], float]` → `(path, cost)`:
- `path` — list of city indices starting and ending at 0, e.g. `[0, 3, 1, 4, 2, 0]`
- `cost` — total tour distance as a float

### Class Pattern
Every solver class follows this structure:
```python
class MySolver:
    def __init__(self, cost_matrix: np.ndarray):
        self.B = np.array(cost_matrix)
        self.n = len(cost_matrix)
        # ... setup

    def solve(self) -> tuple[list[int], float]:
        # Returns (path, cost) where path is list of city indices
        ...
```

### Docstrings
Use Google-style docstrings with Args/Returns sections for public methods.

### Dependencies
Use try/except for optional hardware-specific imports to enable graceful degradation:
```python
try:
    import mlx.core as mx
except ImportError:
    mx = None
```

### Shared Utilities
- `utils.helpers.print_progress(...)` — progress bar output
- `utils.helpers.calculate_total_distance(path, matrix)` — computes tour cost

### Import Conventions
- **Within `algorithms/`**: relative imports → `from .montecarlo import solve_tsp_batch`
- **Cross-package**: absolute from root → `from utils.helpers import calculate_total_distance`
- **In `experiments/`**: add `sys.path.insert(0, '..')` at the top of each script

## Package Structure

```python
# algorithms/__init__.py re-exports all solvers:
from algorithms import (
    SingleQubitTSP,
    two_opt_refinement,
    GeneticTSPSolver,
    ORToolsTSPSolver,
    CPLEXTSPSolver,
    solve_tsp_decomposition,
    solve_tsp_held_karp_mlx,
    solve_tsp_held_karp_cpu,
)

# utils/__init__.py re-exports all utilities:
from utils import (
    load_tsp,
    load_solutions_file,
    TSPInstance,
    calculate_total_distance,
    print_progress,
)
```

## TSPLIB Data (gitignored)

The `tsplib/` directory is excluded from version control. To run benchmarks, obtain the standard TSPLIB instances from [http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and place them in `tsplib/`. The solutions file format is:
```
burma14 : 3323
bayg29 : 1610
att48 : 10628
```

## Benchmark Results (Reference)

From `ReadMe.md` — representative gaps vs. optimal:

| Instance | n | SPSA+2Opt gap | Genetic gap | OR-Tools gap | CPLEX gap |
|---|---|---|---|---|---|
| burma14 | 14 | 3.76% | 4.57% | 0.00% | 0.00% |
| bayg29 | 29 | 0.93% | 16.02% | 0.00% | 0.00% |
| att48 | 48 | 4.25% | 58.17% | 0.00% | — |
| berlin52 | 52 | 11.07% | 61.71% | 0.00% | — |
| ali535 | 535 | skip | skip | 5.45% | — |
| dsj1000 | 1000 | skip | skip | 4.29% | — |

CPLEX requires Python < 3.13 and a valid IBM CPLEX license.

## Adding a New Solver

1. Create `algorithms/<algorithm>.py` with a class following the solver pattern above
2. Add a re-export line in `algorithms/__init__.py`
3. Import it in `test_tsplib.py`
4. Add a benchmark block inside `run_benchmark()` respecting the size threshold pattern
5. Register it in `tsp/registry.py` if CLI support is needed
6. Update the results table in `ReadMe.md`

## Git Workflow

- Main development branch: `master`
- Feature branches: `claude/<description>-<id>`
- Commit messages are imperative, concise, and describe the algorithm change (e.g., `"Introduce CPLEX and ORTOOLS"`, `"add decomposition algorithm"`)
