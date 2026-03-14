"""
TSP solver algorithms package.

Available solvers:
- SingleQubitTSP      — Quantum-inspired SPSA (spsa.py)
- GeneticTSPSolver    — Genetic algorithm with PMX crossover (genetic.py)
- ORToolsTSPSolver    — Google OR-Tools wrapper (ortools.py)
- CPLEXTSPSolver      — IBM CPLEX integer programming (cplex.py)
- solve_tsp_decomposition — K-Means divide-and-conquer (decomposition.py)
- solve_tsp_held_karp_mlx — Held-Karp DP on Apple Silicon GPU (held_karp.py)
- solve_tsp_held_karp_cpu — Held-Karp DP on CPU (held_karp.py)

Standalone refinement:
- two_opt_refinement  — 2-opt edge-swap local search (spsa.py)
"""

from algorithms.spsa import SingleQubitTSP, two_opt_refinement
from algorithms.genetic import GeneticTSPSolver
from algorithms.ortools import ORToolsTSPSolver
from algorithms.cplex import CPLEXTSPSolver
from algorithms.decomposition import solve_tsp_decomposition
from algorithms.held_karp import solve_tsp_held_karp_mlx, solve_tsp_held_karp_cpu

__all__ = [
    "SingleQubitTSP",
    "two_opt_refinement",
    "GeneticTSPSolver",
    "ORToolsTSPSolver",
    "CPLEXTSPSolver",
    "solve_tsp_decomposition",
    "solve_tsp_held_karp_mlx",
    "solve_tsp_held_karp_cpu",
]
