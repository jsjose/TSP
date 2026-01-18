from docplex.mp.model import Model
import numpy as np
import time

class CPLEXTSPSolver:
    def __init__(self, dist_matrix):
        self.B = np.array(dist_matrix)
        self.n = len(dist_matrix)

    def solve(self, log_output=False):
        try:
            import cplex
        except ImportError:
            raise RuntimeError("CPLEX library not found. Please install it via 'pip install cplex'.")

        # Fix for 'module cplex has no attribute exceptions'
        if not hasattr(cplex, 'exceptions'):
            if hasattr(cplex, 'CplexError'):
                # Monkey-patch for docplex compatibility
                class CplexExceptions:
                    CplexError = cplex.CplexError
                cplex.exceptions = CplexExceptions
            else:
                raise RuntimeError("The imported 'cplex' module is invalid. Check if you have a file named 'cplex.py' in your folder shadowing the library.")

        # 1. Create Model
        mdl = Model(name='TSP_MTZ')
        
        # 2. Variables
        # x[i,j] = 1 if edge from i to j is used
        x = mdl.binary_var_matrix(self.n, self.n, name='x')
        # u[i] = sequence rank of city i (MTZ variables)
        u = mdl.continuous_var_list(self.n, lb=0, ub=self.n-1, name='u')

        # 3. Objective
        mdl.minimize(mdl.sum(x[i, j] * self.B[i, j] for i in range(self.n) for j in range(self.n)))

        # 4. Assignment Constraints
        for i in range(self.n):
            mdl.add_constraint(mdl.sum(x[i, j] for j in range(self.n) if i != j) == 1) # Out-degree
        for j in range(self.n):
            mdl.add_constraint(mdl.sum(x[i, j] for i in range(self.n) if i != j) == 1) # In-degree

        # 5. MTZ Subtour Elimination
        # Prevents u[i] - u[j] from looping without the depot
        for i in range(1, self.n):
            for j in range(1, self.n):
                if i != j:
                    mdl.add_constraint(u[i] - u[j] + self.n * x[i, j] <= self.n - 1)

        # 6. Solve
        solution = mdl.solve(log_output=log_output)
        
        if solution:
            # Decode Path
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