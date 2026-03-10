"""Solver registry — maps string names to solver classes.

Usage:
    from tsp.registry import get_solver, list_solvers

    SolverClass = get_solver("ortools")
    solver = SolverClass(cost_matrix)
    path, cost = solver.solve()
"""

def _import_solvers():
    """Lazy imports so missing optional deps only fail when that solver is used."""
    registry = {}

    try:
        from lib.tsp_spsa import SingleQubitTSP
        registry["spsa"] = SingleQubitTSP
    except ImportError:
        pass

    try:
        from lib.tsp_ga import GeneticTSPSolver
        registry["genetic"] = GeneticTSPSolver
    except ImportError:
        pass

    try:
        from lib.tsp_ortools import ORToolsTSPSolver
        registry["ortools"] = ORToolsTSPSolver
    except ImportError:
        pass

    try:
        from lib.tsp_cplex import CPLEXTSPSolver
        registry["cplex"] = CPLEXTSPSolver
    except ImportError:
        pass

    try:
        from lib.tsp_lkh import LKH3Solver
        registry["lkh"] = LKH3Solver
    except ImportError:
        pass

    try:
        from lib.tsp_held_karp import solve_tsp_held_karp_mlx as _hk

        class _HeldKarpWrapper:
            """Thin wrapper so Held-Karp fits the BaseSolver interface."""
            def __init__(self, cost_matrix):
                import numpy as np
                self.B = np.array(cost_matrix)
                self.n = len(cost_matrix)

            def solve(self):
                cost, path = _hk(self.B)
                return path, cost

        registry["held_karp"] = _HeldKarpWrapper
    except ImportError:
        pass

    return registry


# Solver metadata for help text and size-gating
SOLVER_META = {
    "spsa":        {"label": "SPSA+2Opt",      "max_n": 99},
    "genetic":     {"label": "Genetic Alg",     "max_n": 99},
    "ortools":     {"label": "OR-Tools",        "max_n": None},
    "cplex":       {"label": "CPLEX",           "max_n": 99},
    "lkh":         {"label": "LKH-3",            "max_n": None},
    "held_karp":   {"label": "Held-Karp",       "max_n": 20},
}


def list_solvers() -> list[str]:
    """Return names of all available (importable) solvers."""
    return list(_import_solvers().keys())


def get_solver(name: str):
    """Return the solver class for *name*, or raise ValueError."""
    registry = _import_solvers()
    if name not in registry:
        available = ", ".join(registry.keys()) or "(none)"
        raise ValueError(f"Unknown solver '{name}'. Available: {available}")
    return registry[name]


def solvers_for_n(n: int) -> list[str]:
    """Return solver names that are eligible for an instance of size n."""
    registry = _import_solvers()
    result = []
    for name, meta in SOLVER_META.items():
        if name not in registry:
            continue
        max_n = meta.get("max_n")
        if max_n is None or n <= max_n:
            result.append(name)
    return result
