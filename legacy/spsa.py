import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools
import time # Importar el módulo time
from math import factorial

import numpy as np

def print_progress(iteration, total, start_time, prefix='Progress:', length=30):
    elapsed_time = time.time() - start_time
    progress_ratio = iteration / total
    if progress_ratio > 0:
        eta_seconds = elapsed_time / progress_ratio - elapsed_time
        minutes, seconds = divmod(int(eta_seconds), 60)
        eta_str = f" ETA: {minutes:02d}m {seconds:02d}s"
    else:
        eta_str = ""
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} [{bar}] {percent:.1f}%{eta_str}')
    sys.stdout.flush()

# 1. Four-City Symmetric TSP (Example from Fig. 4)
# The paper uses this to demonstrate the basic functionality.
dist_4_city = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
# Result: Optimal path c1 -> c2 -> c4 -> c3 -> c1 (Total Dist: 80)

# 2. Five-City Asymmetric TSP (Eq. 17 in Appendix A)
# In asymmetric TSP, distance(i,j) != distance(j,i).
dist_5_asymmetric = [
    [0, 10, 8, 9, 7],
    [10, 0, 10, 5, 6],
    [8, 10, 0, 8, 9],
    [9, 5, 8, 0, 6],
    [7, 6, 9, 6, 0]
]
# Result: Optimal path c1 -> c3 -> c4 -> c2 -> c5 -> c1 (Total Dist: 34)

# 3. Eight-City Symmetric TSP (Eq. 18 in Appendix A)
# This is one of the complex cases solved in Figure 6.
dist_8_city = [
    [0, 15, 10, 20, 12, 18, 14, 22],
    [15, 0, 16, 25, 13, 11, 21, 19],
    [10, 16, 0, 30, 17, 24, 15, 28],
    [20, 25, 30, 0, 22, 14, 16, 12],
    [12, 13, 17, 22, 0, 26, 18, 14],
    [18, 11, 24, 14, 26, 0, 12, 10],
    [14, 21, 15, 16, 18, 12, 0, 17],
    [22, 19, 28, 12, 14, 10, 17, 0]
]
# Paper Result: c2 -> c8 -> c6 -> c4 -> c3 -> c7 -> c5 -> c1 -> c2

class SingleQubitTSP:
    def __init__(self, cost_matrix):
        self.B = np.array(cost_matrix)
        self.n = len(cost_matrix)
        # Map cities to equitorial angles (longitude phi)
        self.phi = np.linspace(0, 2 * np.pi, self.n, endpoint=False)

    def get_city_state(self, i):
        """Returns the state |P_ii> on the equator."""
        phi = self.phi[i]
        # State: 1/sqrt(2) * (|0> + exp(i*phi)|1>)
        return np.array([1, np.exp(1j * phi)]) / np.sqrt(2)

    def get_intermediate_state(self, i, j):
        """Returns state |P_ij> encoding distance s_ij."""
        phi_i = self.phi[i]
        dist = self.B[i][j]
        # Normalize distance to an angle theta in [0, pi/2]
        # Brachistochrone mapping: theta proportional to distance
        theta = (dist / (np.max(self.B) * 1.1)) * (np.pi / 2)
        # State: cos(theta/2)|0> + exp(i*phi)sin(theta/2)|1>
        return np.array([np.cos(theta / 2), np.exp(1j * phi_i) * np.sin(theta / 2)])

    def rotation_operator(self, target_state, current_state):
        """Creates a unitary that moves current_state towards target_state."""
        # This is a simplified version of the paper's U_u and U_d
        # We find the rotation that aligns the vectors
        # For simulation, we can model the 'strength' alpha as a projection
        return np.outer(target_state, current_state.conj())

    def cost_function(self, params):
        """Calculates the expected cost for the given parameters."""
        # Use normalized_B if available for better convergence, else B
        matrix = getattr(self, 'normalized_B', self.B)
        total_dist = 0
        for i in range(self.n):
            row = params[i]
            weights = np.exp(row - np.max(row))
            weights /= np.sum(weights)
            total_dist += np.sum(weights * matrix[i])
        return total_dist

    def solve_spsa(self, iterations=100, a=0.1, c=0.01):
        """Simultaneous Perturbation Stochastic Approximation to find optimal path."""
        num_cities = self.n
        # Parameters: one 'strength' alpha for every possible edge (i,j)
        params = np.random.rand(num_cities, num_cities)

        start_time = time.time() # Start time for ETA calculation

        def cost_fn(p):
            # Calculate 'Expected Distance' based on parameter weights
            # In a real quantum run, this would be derived from State Tomography
            total_dist = 0
            for i in range(num_cities):
                weights = np.exp(p[i]) / np.sum(np.exp(p[i]))  # Softmax
                total_dist += np.sum(weights * self.B[i])
            return total_dist

        # SPSA Loop
        for k in range(1, iterations + 1):
            ak = a / (k + 1) ** 0.602
            ck = c / (k + 1) ** 0.101
            delta = np.random.choice([-1, 1], size=params.shape)

            f_plus = cost_fn(params + ck * delta)
            f_minus = cost_fn(params - ck * delta)

            grad = (f_plus - f_minus) / (2 * ck * delta)
            params = params - ak * grad

            if k % 10 == 0 or k == iterations:
                print_progress(k, iterations, start_time)
        print()

        # Decode optimal path from optimized parameters
        path = [0]
        visited = {0}
        for _ in range(num_cities - 1):
            curr = path[-1]
            # Pick next city with highest weight that hasn't been visited
            options = np.argsort(-params[curr])
            for opt in options:
                if opt not in visited:
                    path.append(opt)
                    visited.add(opt)
                    break
        path.append(0)  # Return to start
        return path, params

    def visualize_bloch(self, path):
        """Visualizes the cities and path on the Bloch sphere."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Draw Bloch Sphere surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='whitesmoke', alpha=0.1)
        ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.2)

        # Plot Cities on Equator
        coords = []
        start_time = time.time()
        for i in range(self.n):
            print_progress(i + 1, self.n, start_time, prefix="Visualizing:")
            phi = self.phi[i]
            # Bloch coordinates for state 1/sqrt(2)(|0> + e^(iphi)|1>) are (x,y,z) = (cos(phi), sin(phi), 0)
            xc, yc, zc = np.cos(phi), np.sin(phi), 0
            coords.append((xc, yc, zc))
            ax.scatter(xc, yc, zc, color='red', s=50)
            ax.text(xc * 1.1, yc * 1.1, zc, str(i), fontsize=10, fontweight='bold')
        print()

        # Plot Path
        for k in range(len(path) - 1):
            p1 = coords[path[k]]
            p2 = coords[path[k+1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='blue', linewidth=1.5)

        ax.set_title("TSP Path on Bloch Sphere")
        ax.set_axis_off()
        plt.show()

    def solve_brute_force(self):
        """Finds the exact optimal path using brute-force search with a progress bar."""
        cities = list(range(1, self.n))
        min_cost = float('inf')
        best_path = []

        total_permutations = factorial(self.n - 1)
        start_time = time.time() # Start time for ETA calculation
        
        for k, perm in enumerate(itertools.permutations(cities)):
            current_path = [0] + list(perm) + [0]
            current_cost = 0
            for i in range(len(current_path) - 1):
                current_cost += self.B[current_path[i]][current_path[i+1]]
            
            if current_cost < min_cost:
                min_cost = current_cost
                best_path = current_path

            if k % 10000 == 0 or k == total_permutations - 1: # Update progress every 10,000 permutations or at the end
                print_progress(k + 1, total_permutations, start_time)
        print() # New line after completion
        
        return best_path, min_cost

    def simulate_state_tomography(self, params):
        """
        Simulates state tomography to calculate the probability of each valid path.
        Returns a sorted list of (path, probability) tuples.
        """
        # Calculate transition probabilities (Softmax)
        probs = np.zeros_like(params)
        for i in range(self.n):
            # Use stable softmax
            row_exp = np.exp(params[i] - np.max(params[i]))
            probs[i] = row_exp / np.sum(row_exp)

        # Iterate all valid tours (fixing start at 0)
        cities = list(range(1, self.n))
        path_probabilities = []

        for perm in itertools.permutations(cities):
            path = [0] + list(perm) + [0]
            prob = 1.0
            for k in range(len(path) - 1):
                prob *= probs[path[k]][path[k+1]]
            path_probabilities.append((path, prob))

        # Sort by probability descending
        path_probabilities.sort(key=lambda x: x[1], reverse=True)
        return path_probabilities

    def normalize_distances(self):
        """Scales distances to [0, pi/2] for optimal Bloch sphere distribution."""
        min_val = np.min(self.B[self.B > 0]) # Ignore self-loops
        max_val = np.max(self.B)
        # Map to range [0.1, 1.4] radians to stay away from extreme poles
        self.normalized_B = 0.1 + (self.B - min_val) * (1.3 / (max_val - min_val))
        np.fill_diagonal(self.normalized_B, 0)

    def get_spsa_params(self, k, a=0.1, c=0.01, A=10):
        """Calculates decaying step sizes according to paper's logic."""
        ak = a / (k + 1 + A)**0.602
        ck = c / (k + 1)**0.101
        return ak, ck

    def update_with_momentum(self, params, grad, momentum, beta=0.9, ak=0.01):
        """Applies momentum-based update to the parameters."""
        momentum = beta * momentum + (1 - beta) * grad
        params = params - ak * momentum
        return params, momentum     
    
    def decode_path(self, weights_matrix):
        """Greedy decoding to ensure a valid TSP cycle is returned."""
        n = self.n
        path = [0]
        visited = {0}
        for _ in range(n - 1):
            curr = path[-1]
            # Sort potential next cities by weight (highest first)
            probs = weights_matrix[curr]
            options = np.argsort(-probs)
            for opt in options:
                if opt not in visited:
                    path.append(opt)
                    visited.add(opt)
                    break
        path.append(0)
        return path
    
    def solve_refined(self, trials=5, iterations=500):
        self.normalize_distances()
        best_global_path = None
        min_global_dist = float('inf')

        start_time_global = time.time() # Start time for global ETA

        for trial in range(trials):
            # Initialize random weights for edges
            params = np.random.randn(self.n, self.n) * 0.1
            momentum = np.zeros_like(params)
            
            for k in range(iterations):
                ak, ck = self.get_spsa_params(k)
                delta = np.random.choice([-1, 1], size=params.shape)
                
                # Estimate gradients
                f_plus = self.cost_function(params + ck * delta)
                f_minus = self.cost_function(params - ck * delta)
                grad = (f_plus - f_minus) / (2 * ck * delta)
                
                # Update
                params, momentum = self.update_with_momentum(params, grad, momentum, ak=ak)
            
            # Check this trial's result
            trial_path = self.decode_path(params)
            trial_dist = sum(self.B[trial_path[i]][trial_path[i+1]] for i in range(self.n))
            
            if trial_dist < min_global_dist:
                min_global_dist = trial_dist
                best_global_path = trial_path

            # Update progress bar for trials
            prefix = f"Refined SPSA Progress (Trial {trial + 1}/{trials}):"
            print_progress(trial + 1, trials, start_time_global, prefix=prefix)
        print() # New line after completion
                
        return best_global_path, min_global_dist

def calculate_total_distance(path, dist_matrix):
    """Computes the total length of the TSP tour."""
    distance = 0
    for i in range(len(path) - 1):
        distance += dist_matrix[path[i]][path[i+1]]
    return distance

def two_opt_refinement(path, dist_matrix):
    """
    Iteratively improves a TSP path by swapping edges (2-opt).
    """
    best_path = list(path)
    best_dist = calculate_total_distance(best_path, dist_matrix)
    improved = True
    
    while improved:
        improved = False
        start_time = time.time()
        total_steps = len(best_path) - 3
        # Loop through all possible pairs to swap (excluding adjacent edges)
        for i in range(1, len(best_path) - 2):
            print_progress(i, total_steps, start_time, prefix="2-opt Refinement:")
            for j in range(i + 1, len(best_path) - 1):
                # Potential new path: reverse the segment between i and j
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                new_dist = calculate_total_distance(new_path, dist_matrix)
                
                if new_dist < best_dist:
                    best_path = new_path
                    best_dist = new_dist
                    improved = True
                    break # Restart with the improved path
            if improved:
                break
    print()
                
    return best_path, best_dist

def generate_random_tsp(n_cities, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(n_cities, 2) * 100
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

# --- Example Usage ---
test_cases = [
    ("4-City Symmetric", dist_4_city),
    ("5-City Asymmetric", dist_5_asymmetric),
    ("8-City Symmetric", dist_8_city),
    ("6-City Random", generate_random_tsp(6)),
    ("7-City Random", generate_random_tsp(7)),
    ("11-City Random", generate_random_tsp(11)),
    ("13-City Random", generate_random_tsp(13)),
    ("17-City Random", generate_random_tsp(17))
]

for name, matrix in test_cases:
    print(f"\n--- {name} ---")
    solver = SingleQubitTSP(matrix)
    
    print("Solving with SPSA...")
    best_path, params = solver.solve_spsa(iterations=2000)
    spsa_cost = sum(solver.B[best_path[i]][best_path[i+1]] for i in range(len(best_path)-1))
    print(f"SPSA Path: {' -> '.join(map(str, best_path))}")
    print(f"SPSA Cost: {spsa_cost}")

    if solver.n <= 8:
        print("Calculating path probabilities (State Tomography)...")
        path_probs = solver.simulate_state_tomography(params)
        print("Top 3 Probable Paths:")
        for p, prob in path_probs[:3]:
            cost = sum(solver.B[p[i]][p[i+1]] for i in range(len(p)-1))
            print(f"  Path: {' -> '.join(map(str, p))} | Prob: {prob:.4e} | Cost: {cost}")
    else:
        print(f"Skipping State Tomography for {solver.n} cities (too many permutations).")  

    print("Solving with Refined SPSA...")
    refined_path, refined_cost = solver.solve_refined(trials=3, iterations=1000)
    print(f"Refined Path: {' -> '.join(map(str, refined_path))}")
    print(f"Refined Cost: {refined_cost}") 

    print("Applying 2-opt Refinement...")
    optimized_path, final_cost = two_opt_refinement(refined_path, solver.B)
    print(f"Optimized Path after 2-opt: {' -> '.join(map(str, optimized_path))}")
    print(f"Optimized Cost after 2-opt: {final_cost}")  

    if solver.n < 13:
        print("Calculating exact solution (Brute Force)...")
        bf_path, bf_cost = solver.solve_brute_force()
        print(f"Exact Path: {' -> '.join(map(str, bf_path))}")
        print(f"Exact Cost: {bf_cost}")
    else:
        print(f"Skipping Brute Force for {solver.n} cities (too computationally expensive).")
