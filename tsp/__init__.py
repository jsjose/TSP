"""TSP solver library.

Quick-start
-----------
    import lib.tsp_loader as tsp_loader
    from tsp import solve, list_solvers

    instance = tsp_loader.load_tsp("tsplib/berlin52.tsp")
    path, cost = solve(instance.distance_matrix, solver="ortools")
"""

from tsp.registry import get_solver, list_solvers, solvers_for_n
from tsp.store import ResultStore


def solve(cost_matrix, solver: str = "ortools", **kwargs) -> tuple[list[int], float]:
    """Convenience wrapper: instantiate *solver* and call solve().

    Args:
        cost_matrix: NxN distance matrix (list-of-lists or np.ndarray).
        solver: Solver key. See list_solvers() for available options.
        **kwargs: Extra keyword arguments forwarded to solver.solve().

    Returns:
        (path, cost) tuple.
    """
    cls = get_solver(solver)
    return cls(cost_matrix).solve(**kwargs)


__all__ = ["solve", "list_solvers", "solvers_for_n", "get_solver", "ResultStore"]
