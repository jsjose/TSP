import numpy as np
from docplex.mp.model import Model


class CPLEXTSPSolver:
    """TSP solver using IBM CPLEX with Miller-Tucker-Zemlin (MTZ) subtour elimination.

    Requires a valid IBM CPLEX installation (Python < 3.13).

    Args:
        dist_matrix: NxN distance/cost matrix.
    """

    def __init__(self, dist_matrix) -> None:
        self.B = np.array(dist_matrix)
        self.n = len(dist_matrix)

    def solve(
        self, log_output: bool = False
    ) -> tuple[list[int] | None, int | None]:
        """Solves TSP to optimality via binary programming.

        Variables:
            x[i,j]: Binary edge selection variables.
            u[i]: Sequence rank variables (MTZ subtour elimination).

        Args:
            log_output: Whether to print CPLEX solver logs.

        Returns:
            (path, cost): Optimal tour and its integer cost,
            or (None, None) if no solution was found.
        """
        try:
            import cplex
        except ImportError:
            raise RuntimeError(
                "CPLEX library not found. Please install it via 'pip install cplex'."
            )

        # Fix for 'module cplex has no attribute exceptions'
        if not hasattr(cplex, "exceptions"):
            if hasattr(cplex, "CplexError"):
                # Monkey-patch for docplex compatibility
                class CplexExceptions:
                    CplexError = cplex.CplexError

                cplex.exceptions = CplexExceptions
            else:
                raise RuntimeError(
                    "The imported 'cplex' module is invalid. "
                    "Check if you have a file named 'cplex.py' in your folder shadowing the library."
                )

        mdl = Model(name="TSP_MTZ")

        # x[i,j] = 1 if edge from i to j is used
        x = mdl.binary_var_matrix(self.n, self.n, name="x")
        # u[i] = sequence rank of city i (MTZ variables)
        u = mdl.continuous_var_list(self.n, lb=0, ub=self.n - 1, name="u")

        # Objective: minimize total tour cost
        mdl.minimize(
            mdl.sum(x[i, j] * self.B[i, j] for i in range(self.n) for j in range(self.n))
        )

        # Assignment constraints: each city has exactly one outgoing and one incoming edge
        for i in range(self.n):
            mdl.add_constraint(
                mdl.sum(x[i, j] for j in range(self.n) if i != j) == 1
            )
        for j in range(self.n):
            mdl.add_constraint(
                mdl.sum(x[i, j] for i in range(self.n) if i != j) == 1
            )

        # MTZ subtour elimination: prevents u[i] - u[j] from cycling without the depot
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i != j:
                    mdl.add_constraint(
                        u[i] - u[j] + self.n * x[i, j] <= self.n - 1
                    )

        solution = mdl.solve(log_output=log_output)

        if solution:
            path = [0]
            while len(path) < self.n:
                curr = path[-1]
                for j in range(self.n):
                    if solution.get_value(x[curr, j]) > 0.5:
                        path.append(j)
                        break
            path.append(0)
            return path, int(solution.objective_value)

        return None, None
