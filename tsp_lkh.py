import lkh
import numpy as np
import time
import tsplib95
import os
import tempfile

class LKH3Solver:
    def __init__(self, dist_matrix):
        self.B = np.array(dist_matrix)
        self.n = len(dist_matrix)

    def solve(self):
        """
        Solves TSP using the LKH-3 heuristic via an external process bridge.
        """
        # 1. Create a tsplib95 problem from the distance matrix
        problem = tsplib95.models.StandardProblem()
        problem.name = 'TSP'
        problem.type = 'TSP'
        problem.dimension = self.n
        problem.edge_weight_type = 'EXPLICIT'
        problem.edge_weight_format = 'FULL_MATRIX'

        # tsplib95 expects edge_weights as a list of rows (list of lists)
        problem.edge_weights = self.B.astype(int).tolist()
        
        # 2. Write problem to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsp') as tmp_file:
            problem.write(tmp_file)
            tmp_path = tmp_file.name
        
        start_time = time.time()
        
        # 3. Execute LKH-3 via wrapper (expects file path)
        try:
            solution = lkh.solve(tmp_path, runs=10, max_trials=1000)
        except Exception as e:
            print(f"LKH execution failed: {e}")
            solution = None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        execution_time = time.time() - start_time
        
        if solution:
            # Handle potential return types (list of nodes or list of lists)
            path_indices = solution[0] if isinstance(solution[0], list) else solution
            
            # LKH returns 1-based indexing; convert back to 0-based
            path = [node - 1 for node in path_indices]
            path.append(path[0]) # Close the loop
            
            # Calculate final cost
            cost = sum(self.B[path[i]][path[i+1]] for i in range(self.n))
            return path, int(cost)

        return None, None
