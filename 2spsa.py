import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools
import time
from math import factorial
import datetime

# --- Utility: Progress Bar ---
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

class SingleQubitTSP:
    def __init__(self, cost_matrix):
        self.B = np.array(cost_matrix)
        self.n = len(cost_matrix)
        self.phi = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        self.normalize_distances()

    def normalize_distances(self):
        """Scales distances for optimal Bloch sphere mapping."""
        min_val = np.min(self.B[self.B > 0]) if np.any(self.B > 0) else 0
        max_val = np.max(self.B)
        self.normalized_B = 0.1 + (self.B - min_val) * (1.3 / (max_val - min_val))
        np.fill_diagonal(self.normalized_B, 0)

    def cost_function(self, params):
        """Calculates expected cost using softmax to simulate qubit state populations."""
        matrix = getattr(self, 'normalized_B', self.B)
        total_dist = 0
        for i in range(self.n):
            row = params[i]
            # Stable softmax
            weights = np.exp(row - np.max(row))
            weights /= np.sum(weights)
            total_dist += np.sum(weights * matrix[i])
        return total_dist

    def decode_path(self, params):
        """Greedy decoding to ensure a valid Hamiltonian cycle."""
        path = [0]
        visited = {0}
        for _ in range(self.n - 1):
            curr = path[-1]
            options = np.argsort(-params[curr])
            for opt in options:
                if opt not in visited:
                    path.append(opt)
                    visited.add(opt)
                    break
        path.append(0)
        return path

    def solve_hybrid(self, iters_1st=1000, iters_2nd=500):
        """
        Phase 1: 1SPSA (Global Search)
        Phase 2: 2SPSA (Curvature-based fine-tuning)
        Phase 3: 2-Opt (Classical untangling)
        """
        params = np.random.randn(self.n, self.n) * 0.1
        hessian = np.eye(self.n**2)
        start_time = time.time()

        # --- Phase 1: 1SPSA (Global Exploration) ---
        for k in range(1, iters_1st + 1):
            ak = 0.1 / (k + 10)**0.602
            ck = 0.01 / k**0.101
            delta = np.random.choice([-1, 1], size=params.shape)
            
            grad = (self.cost_function(params + ck*delta) - 
                    self.cost_function(params - ck*delta)) / (2*ck*delta)
            params -= ak * grad
            
            if k % 50 == 0: print_progress(k, iters_1st + iters_2nd, start_time, prefix="Hybrid Phase 1 (1SPSA):")

        # --- Phase 2: 2SPSA (Precision Convergence) ---
        for k in range(iters_1st + 1, iters_1st + iters_2nd + 1):
            ak = 0.05 / (k + 10)**0.602
            ck = 0.01 / k**0.101
            delta = np.random.choice([-1, 1], size=params.shape)
            tilde_delta = np.random.choice([-1, 1], size=params.shape)
            
            y_plus = self.cost_function(params + ck*delta)
            y_minus = self.cost_function(params - ck*delta)
            y_plus_h = self.cost_function(params + ck*delta + ck*tilde_delta)
            y_minus_h = self.cost_function(params - ck*delta + ck*tilde_delta)
            
            grad = (y_plus - y_minus) / (2 * ck * delta)
            g_diff = ((y_plus_h - y_plus)/ck - (y_minus_h - y_minus)/ck).flatten()
            delta_flat = delta.flatten()
            
            H_k = 0.5 * (np.outer(g_diff, delta_flat) + np.outer(delta_flat, g_diff))
            hessian = 0.9 * hessian + 0.1 * H_k
            
            # Matrix regularization and inversion
            inv_H = np.linalg.inv(hessian + (1e-4)*np.eye(len(hessian)))
            params -= ak * (inv_H @ grad.flatten()).reshape(params.shape)
            
            if k % 50 == 0: print_progress(k, iters_1st + iters_2nd, start_time, prefix="Hybrid Phase 2 (2SPSA):")
        print("\nOptimization complete.")

        # --- Phase 3: 2-Opt Refinement ---
        rough_path = self.decode_path(params)
        print("Starting Phase 3 (2-Opt Refinement)...")
        final_path, final_cost = self.run_2opt(rough_path)
        
        return final_path, final_cost

    def run_2opt(self, path):
        """Iteratively untangles edge crossings classically."""
        best_path = list(path)
        best_dist = sum(self.B[best_path[i]][best_path[i+1]] for i in range(self.n))
        improved = True
        while improved:
            improved = False
            for i in range(1, self.n - 1):
                for j in range(i + 1, self.n):
                    new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                    new_dist = sum(self.B[new_path[k]][new_path[k+1]] for k in range(self.n))
                    if new_dist < best_dist:
                        best_path, best_dist = new_path, new_dist
                        improved = True
                        break
                if improved: break
        return best_path, best_dist

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

    def visualize_bloch(self, path):
        """Draws the final TSP path as geodesics on the Bloch Sphere equator."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Sphere Surface
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color='whitesmoke', alpha=0.1)
        ax.plot_wireframe(x, y, z, color='lightgray', alpha=0.1)

        # Cities (Equatorial Mapping)
        coords = []
        for i in range(self.n):
            phi = self.phi[i]
            xc, yc, zc = np.cos(phi), np.sin(phi), 0
            coords.append((xc, yc, zc))
            ax.scatter(xc, yc, zc, color='red', s=60, edgecolors='black')
            ax.text(xc*1.1, yc*1.1, zc, f"C{i}", fontsize=12, fontweight='bold')

        # Tour Path
        for k in range(len(path) - 1):
            p1, p2 = coords[path[k]], coords[path[k+1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color='blue', linewidth=2.5, alpha=0.8)

        ax.set_title("Hybrid TSP Path on Bloch Sphere", fontsize=15)
        ax.set_axis_off()
        plt.show()

# --- Execution ---
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

dist_4_city = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

dist_5_asymmetric = [
    [0, 10, 8, 9, 7],
    [10, 0, 10, 5, 6],
    [8, 10, 0, 8, 9],
    [9, 5, 8, 0, 6],
    [7, 6, 9, 6, 0]
]

dist_8_city = [
    [0, 15, 10, 20, 12, 18, 14, 22], [15, 0, 16, 25, 13, 11, 21, 19],
    [10, 16, 0, 30, 17, 24, 15, 28], [20, 25, 30, 0, 22, 14, 16, 12],
    [12, 13, 17, 22, 0, 26, 18, 14], [18, 11, 24, 14, 26, 0, 12, 10],
    [14, 21, 15, 16, 18, 12, 0, 17], [22, 19, 28, 12, 14, 10, 17, 0]
]

test_cases = [
    ("4-City Symmetric", dist_4_city),
    ("5-City Asymmetric", dist_5_asymmetric),
    ("8-City Symmetric", dist_8_city),
    ("6-City Random", generate_random_tsp(6)),
    ("7-City Random", generate_random_tsp(7)),
    ("10-City Random", generate_random_tsp(10)),
    ("11-City Random", generate_random_tsp(11)),
    ("12-City Random", generate_random_tsp(12)),
    ("13-City Random", generate_random_tsp(13)),
    ("17-City Random", generate_random_tsp(17)),
    ("20-City Random", generate_random_tsp(20)),
    ("50-City Random", generate_random_tsp(50)),
    ("100-City Random", generate_random_tsp(100))
]

results_summary = []

for name, matrix in test_cases:
    print(f"\n--- {name} ---")
    solver = SingleQubitTSP(matrix)
    
    best_path = None
    best_cost = None
    hybrid_time = None

    if solver.n < 50:
        print("Solving with Hybrid SPSA...")
        start_time = time.time()
        best_path, best_cost = solver.solve_hybrid(iters_1st=1000, iters_2nd=500)
        hybrid_time = time.time() - start_time
        print(f"Hybrid Path: {' -> '.join(map(str, best_path))}")
        print(f"Hybrid Cost: {best_cost:.2f}")
    else:
        print(f"Skipping Hybrid SPSA for {solver.n} cities (too computationally expensive)")

    print("Solving with Refined SPSA...")
    start_time = time.time()
    refined_path, refined_cost = solver.solve_refined(trials=3, iterations=1000)
    refined_time = time.time() - start_time
    print(f"Refined Path: {' -> '.join(map(str, refined_path))}")
    print(f"Refined Cost: {refined_cost}") 

    print("Applying 2-opt Refinement...")
    start_time = time.time()
    optimized_path, final_cost = two_opt_refinement(refined_path, solver.B)
    two_opt_time = time.time() - start_time
    print(f"Optimized Path after 2-opt: {' -> '.join(map(str, optimized_path))}")
    print(f"Optimized Cost after 2-opt: {final_cost}")  

    bf_cost = None
    bf_time = None
    if solver.n < 13:
        print("Calculating exact solution (Brute Force)...")
        start_time = time.time()
        bf_path, bf_cost = solver.solve_brute_force()
        bf_time = time.time() - start_time
        print(f"Exact Path: {' -> '.join(map(str, bf_path))}")
        print(f"Exact Cost: {bf_cost}")
    else:
        print(f"Skipping Brute Force for {solver.n} cities (too computationally expensive).")
    
    results_summary.append({
        "Case": name,
        "Matrix": matrix,
        "Hybrid": (best_cost, hybrid_time, best_path),
        "Refined": (refined_cost, refined_time, refined_path),
        "Refined+2Opt": (final_cost, refined_time + two_opt_time, optimized_path),
        "BruteForce": (bf_cost, bf_time, bf_path)
    })

print("\n" + "="*115)
print(f"{'TEST CASE':<20} | {'HYBRID (Cost/Time)':<25} | {'REFINED (Cost/Time)':<25} | {'REF+2OPT (Cost/Time)':<25} | {'BRUTE FORCE':<15}")
print("-" * 115)
for res in results_summary:
    h_c, h_t, _ = res["Hybrid"]
    r_c, r_t, _ = res["Refined"]
    r2_c, r2_t, _ = res["Refined+2Opt"]
    bf_c, bf_t, _ = res["BruteForce"]
    
    if h_c is not None:
        h_str = f"{h_c:.2f} / {h_t:.2f}s"
    else:
        h_str = "N/A"
    r_str = f"{r_c:.2f} / {r_t:.2f}s"
    r2_str = f"{r2_c:.2f} / {r2_t:.2f}s"
    
    if bf_c is not None:
        bf_str = f"{bf_c:.2f} / {bf_t:.4f}s"
    else:
        bf_str = "N/A"
    
    print(f"{res['Case']:<20} | {h_str:<25} | {r_str:<25} | {r2_str:<25} | {bf_str:<15}")
print("="*115 + "\n")

# --- Export Results to File ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"tsp_results_{timestamp}.txt"

with open(filename, "w") as f:
    f.write(f"TSP Optimization Results - {timestamp}\n")
    f.write("="*115 + "\n")
    f.write(f"{'TEST CASE':<20} | {'HYBRID (Cost/Time)':<25} | {'REFINED (Cost/Time)':<25} | {'REF+2OPT (Cost/Time)':<25} | {'BRUTE FORCE':<15}\n")
    f.write("-" * 115 + "\n")
    
    for res in results_summary:
        h_c, h_t, _ = res["Hybrid"]
        r_c, r_t, _ = res["Refined"]
        r2_c, r2_t, _ = res["Refined+2Opt"]
        bf_c, bf_t, _ = res["BruteForce"]
        
        if h_c is not None:
            h_str = f"{h_c:.2f} / {h_t:.2f}s"
        else:
            h_str = "N/A"
        r_str = f"{r_c:.2f} / {r_t:.2f}s"
        r2_str = f"{r2_c:.2f} / {r2_t:.2f}s"
        
        if bf_c is not None:
            bf_str = f"{bf_c:.2f} / {bf_t:.4f}s"
        else:
            bf_str = "N/A"
        
        f.write(f"{res['Case']:<20} | {h_str:<25} | {r_str:<25} | {r2_str:<25} | {bf_str:<15}\n")
    
    f.write("="*115 + "\n\n")
    f.write("DETAILED RESULTS\n")
    f.write("="*115 + "\n")

    for res in results_summary:
        f.write(f"\n--- {res['Case']} ---\n")
        f.write("Distance Matrix:\n")
        f.write(np.array2string(np.array(res['Matrix']), precision=2, separator=', ', threshold=sys.maxsize) + "\n\n")
        
        if res['Hybrid'][0] is not None:
            f.write(f"Hybrid SPSA:     Cost={res['Hybrid'][0]:.2f}, Time={res['Hybrid'][1]:.4f}s, Path={res['Hybrid'][2]}\n")
        else:
            f.write("Hybrid SPSA:     Skipped\n")
        f.write(f"Refined SPSA:    Cost={res['Refined'][0]:.2f}, Time={res['Refined'][1]:.4f}s, Path={res['Refined'][2]}\n")
        f.write(f"Refined + 2-Opt: Cost={res['Refined+2Opt'][0]:.2f}, Time={res['Refined+2Opt'][1]:.4f}s, Path={res['Refined+2Opt'][2]}\n")
        if res['BruteForce'][0] is not None:
            f.write(f"Brute Force:     Cost={res['BruteForce'][0]:.2f}, Time={res['BruteForce'][1]:.4f}s, Path={res['BruteForce'][2]}\n")
        else:
            f.write("Brute Force:     Skipped\n")
        f.write("-" * 50 + "\n")

print(f"Results exported to {filename}")

# --- Generate Plots ---
print("Generating plots...")
n_vals = []
hybrid_costs, hybrid_times = [], []
refined_costs, refined_times = [], []
r2opt_costs, r2opt_times = [], []
bf_costs, bf_times = [], []

for res in results_summary:
    n = len(res['Matrix'])
    n_vals.append(n)
    
    # Hybrid
    hc, ht, _ = res['Hybrid']
    hybrid_costs.append(hc if hc is not None else np.nan)
    hybrid_times.append(ht if ht is not None else np.nan)
    
    # Refined
    rc, rt, _ = res['Refined']
    refined_costs.append(rc)
    refined_times.append(rt)
    
    # Refined + 2Opt
    r2c, r2t, _ = res['Refined+2Opt']
    r2opt_costs.append(r2c)
    r2opt_times.append(r2t)
    
    # Brute Force
    bfc, bft, _ = res['BruteForce']
    bf_costs.append(bfc if bfc is not None else np.nan)
    bf_times.append(bft if bft is not None else np.nan)

# Sort data by N
perm = np.argsort(n_vals)
n_vals = np.array(n_vals)[perm]
hybrid_costs = np.array(hybrid_costs)[perm]
hybrid_times = np.array(hybrid_times)[perm]
refined_costs = np.array(refined_costs)[perm]
refined_times = np.array(refined_times)[perm]
r2opt_costs = np.array(r2opt_costs)[perm]
r2opt_times = np.array(r2opt_times)[perm]
bf_costs = np.array(bf_costs)[perm]
bf_times = np.array(bf_times)[perm]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Cost Plot
ax1.plot(n_vals, hybrid_costs, 'o-', label='Hybrid SPSA')
ax1.plot(n_vals, refined_costs, 's-', label='Refined SPSA')
ax1.plot(n_vals, r2opt_costs, '^-', label='Refined + 2-Opt')
ax1.plot(n_vals, bf_costs, 'x--', label='Brute Force', color='k', alpha=0.6)
ax1.set_title('Cost vs Number of Cities')
ax1.set_xlabel('Number of Cities')
ax1.set_ylabel('Cost')
ax1.legend()
ax1.grid(True)

# Time Plot
ax2.plot(n_vals, hybrid_times, 'o-', label='Hybrid SPSA')
ax2.plot(n_vals, refined_times, 's-', label='Refined SPSA')
ax2.plot(n_vals, r2opt_times, '^-', label='Refined + 2-Opt')
ax2.plot(n_vals, bf_times, 'x--', label='Brute Force', color='k', alpha=0.6)
ax2.set_title('Time vs Number of Cities')
ax2.set_xlabel('Number of Cities')
ax2.set_ylabel('Time (s)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True)

plot_filename = f"tsp_results_{timestamp}.png"
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")