import lkh
import numpy as np
import time
import os
import tempfile

class LKH3Solver:
    def __init__(self, dist_matrix):
        self.B = np.array(dist_matrix)
        self.n = len(dist_matrix)

    def _write_tsp_file(self, path):
        """Write a TSPLIB EXPLICIT FULL_MATRIX problem file."""
        with open(path, 'w') as f:
            f.write("NAME: TSP\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {self.n}\n")
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for row in self.B.astype(int).tolist():
                f.write(" ".join(str(v) for v in row) + "\n")
            f.write("EOF\n")

    def solve(self):
        """
        Solves TSP using the LKH-3 heuristic via an external process bridge.
        """
        # 1. Write problem to temporary file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tsp')
        os.close(tmp_fd)
        self._write_tsp_file(tmp_path)
        
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
