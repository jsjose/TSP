# TSP Solvers

A research/benchmarking project implementing and comparing multiple algorithms for the **Traveling Salesman Problem (TSP)**. Solvers range from a novel quantum-inspired SPSA approach to classical exact solvers (CPLEX, OR-Tools).

## Algorithms

| Solver | Module | Type |
|---|---|---|
| `SingleQubitTSP` | `tsp_spsa.py` | Quantum-inspired heuristic (SPSA) |
| `GeneticTSPSolver` | `tsp_ga.py` | Evolutionary algorithm (PMX + scramble mutation) |
| `ORToolsTSPSolver` | `tsp_ortools.py` | Guided Local Search (Google OR-Tools) |
| `CPLEXTSPSolver` | `tsp_cplex.py` | Exact ILP (IBM CPLEX, MTZ formulation) |
| `solve_tsp_decomposition` | `tsp_decomposition.py` | Divide-and-conquer (k-means clustering) |
| Held-Karp | `tsp_held_karp.py` | Exact DP — O(n² × 2ⁿ), small instances only |

### SingleQubitTSP (`tsp_spsa.py`)
Quantum-inspired heuristic. Each city's preference weights are a softmax over an n×n parameter matrix, optimized by SPSA.
- `solve_hybrid()` — Phase 1: 1SPSA (global), Phase 2: 2SPSA (curvature), Phase 3: 2-opt
- `solve_refined()` — Multi-start refined SPSA with 2-opt post-processing
- `two_opt_refinement(path, matrix)` — Standalone 2-opt local search

### GeneticTSPSolver (`tsp_ga.py`)
Population-based evolutionary solver.
- **PMX crossover** — Partially Mapped Crossover for valid permutation offspring
- **Scramble mutation** — Randomly shuffles a subsequence for diversity
- Selection draws from the full population (not top-N) to prevent premature convergence

### ORToolsTSPSolver (`tsp_ortools.py`)
Wrapper around Google OR-Tools `RoutingModel` with Guided Local Search. Time limits scale with instance size.

### CPLEXTSPSolver (`tsp_cplex.py`)
Integer programming via IBM CPLEX using Miller-Tucker-Zemlin (MTZ) subtour elimination. Fastest exact solver for small instances. Requires Python < 3.13 and a valid IBM CPLEX license.

### solve_tsp_decomposition (`tsp_decomposition.py`)
Divide-and-conquer heuristic: splits cities with k-means clustering, solves each cluster with Refined SPSA, then stitches sub-tours via a centroid TSP. Skipped when coordinate data is unavailable.

## Size-Based Algorithm Selection

The benchmark applies size thresholds automatically:

| Instance size | Algorithms run |
|---|---|
| n < 100 | SPSA+2Opt, Genetic, OR-Tools, CPLEX, Decomposition |
| n ≥ 100 | OR-Tools only |

## How to Run

```bash
python test_tsplib.py
```

TSPLIB data files must be present in `tsplib/` (gitignored). Obtain them from [TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) and place them in `tsplib/`. The solutions file format is:
```
burma14 : 3323
bayg29 : 1610
att48 : 10628
```

## Benchmark Results

```
==============================================================================================================
INSTANCE     | METHOD          | COST       | TIME (s)   | GAP (%)    | OPTIMAL
--------------------------------------------------------------------------------------------------------------
burma14      | SPSA+2Opt       | 3448.0     | 0.46       | 3.76       | 3323.0
             | Genetic Alg     | 3475.0     | 0.77       | 4.57       | 3323.0
             | OR-Tools        | 3323.0     | 5.01       | 0.00       | 3323.0
             | CPLEX           | 3323.0     | 0.06       | 0.00       | 3323.0
             | Decomposition   | 5150.0     | 0.08       | 54.98      | 3323.0
--------------------------------------------------------------------------------------------------------------
bayg29       | SPSA+2Opt       | 1625.0     | 0.97       | 0.93       | 1610.0
             | Genetic Alg     | 1868.0     | 1.53       | 16.02      | 1610.0
             | OR-Tools        | 1610.0     | 5.00       | 0.00       | 1610.0
             | CPLEX           | 1610.0     | 0.24       | 0.00       | 1610.0
             | Decomposition   | 3066.0     | 0.11       | 90.43      | 1610.0
--------------------------------------------------------------------------------------------------------------
att48        | SPSA+2Opt       | 11080.0    | 1.96       | 4.25       | 10628.0
             | Genetic Alg     | 16810.0    | 2.91       | 58.17      | 10628.0
             | OR-Tools        | 10628.0    | 9.00       | 0.00       | 10628.0
             | Decomposition   | 17548.0    | 0.18       | 65.11      | 10628.0
--------------------------------------------------------------------------------------------------------------
berlin52     | SPSA+2Opt       | 8377.0     | 2.27       | 11.07      | 7542.0
             | Genetic Alg     | 12196.0    | 3.26       | 61.71      | 7542.0
             | OR-Tools        | 7542.0     | 10.00      | 0.00       | 7542.0
             | Decomposition   | 13561.0    | 0.19       | 79.81      | 7542.0
--------------------------------------------------------------------------------------------------------------
brazil58     | SPSA+2Opt       | 25704.0    | 3.29       | 1.22       | 25395.0
             | Genetic Alg     | 45591.0    | 3.82       | 79.53      | 25395.0
             | OR-Tools        | 25395.0    | 11.00      | 0.00       | 25395.0
--------------------------------------------------------------------------------------------------------------
dantzig42    | SPSA+2Opt       | 747.0      | 1.53       | 6.87       | 699.0
             | Genetic Alg     | 1050.0     | 2.47       | 50.21      | 699.0
             | OR-Tools        | 699.0      | 8.00       | 0.00       | 699.0
             | Decomposition   | 1242.0     | 0.16       | 77.68      | 699.0
--------------------------------------------------------------------------------------------------------------
ali535       | SPSA+2Opt       | Skip       | Skip       | Skip       | 202339.0
             | Genetic Alg     | Skip       | Skip       | Skip       | 202339.0
             | OR-Tools        | 213364.0   | 107.00     | 5.45       | 202339.0
--------------------------------------------------------------------------------------------------------------
dsj1000      | SPSA+2Opt       | Skip       | Skip       | Skip       | 18660188.0
             | Genetic Alg     | Skip       | Skip       | Skip       | 18660188.0
             | OR-Tools        | 19459947.0 | 200.01     | 4.29       | 18660188.0
--------------------------------------------------------------------------------------------------------------
brg180       | SPSA+2Opt       | Skip       | Skip       | Skip       | 1950.0
             | Genetic Alg     | Skip       | Skip       | Skip       | 1950.0
             | OR-Tools        | 1950.0     | 36.00      | 0.00       | 1950.0
--------------------------------------------------------------------------------------------------------------
```

## Setup

```bash
uv sync        # preferred
# or
pip install -r requirements.txt
```

Python 3.12 required (see `.python-version`). CPLEX requires Python < 3.13.
