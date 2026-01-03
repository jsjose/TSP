import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools
import time
from math import factorial
import datetime
import mlx.core as mx
from itertools import combinations
import os
from sklearn.cluster import KMeans

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

def get_batch_distances(perms, dist_matrix):
    """
    Vectorized distance calculation for a batch of permutations.
    perms: (batch_size, n_cities)
    dist_matrix: (n_cities, n_cities)
    """
    # Create indices for 'from' and 'to' cities
    # perms[:, :-1] are starting cities, perms[:, 1:] are destination cities
    starts = perms[:, :-1]
    ends = perms[:, 1:]
    
    # Return to start city to close the loop
    starts_final = perms[:, -1]
    ends_final = perms[:, 0]
    
    # Vectorized gathering of distances from the matrix
    dists = dist_matrix[starts, ends].sum(axis=1)
    dists += dist_matrix[starts_final, ends_final]
    
    return dists

@mx.compile
def solve_tsp_batch(batch_indices, n_cities, dist_matrix):
    """
    Generates a batch of random permutations and finds the best one.
    """
    batch_size = batch_indices.shape[0]
    
    # Generate random permutations using argsort on random noise
    keys = mx.random.uniform(shape=(batch_size, n_cities))
    perms = mx.argsort(keys, axis=1)
    
    dists = get_batch_distances(perms, dist_matrix)
    
    best_idx = mx.argmin(dists)
    return dists[best_idx], perms[best_idx]

def solve_tsp_held_karp_mlx(dist_matrix):
    """
    Solves TSP using Held-Karp (Dynamic Programming) on Apple Silicon GPU (MLX).
    
    Args:
        dist_matrix (np.ndarray): NxN distance matrix.
        
    Returns:
        (cost, path): Tuple containing the minimum cost and the optimal path list.
    """
    n = int(dist_matrix.shape[0])
    
    # 1. Initialize DP Table and Parent Table
    # Rows: 2^n subsets (bitmasks)
    # Cols: n cities (last visited city)
    
    # We use 'inf' for unreachable states
    dp = mx.full((1 << n, n), float('inf'))
    
    # Parent table to store the previous city for path reconstruction
    # We initialize with -1
    parent = mx.full((1 << n, n), -1, dtype=mx.int32)

    # Base case: Starting at city 0 with only city 0 visited
    # Mask 1 (binary ...001) represents set {0}
    dp[1, 0] = 0.0

    # Convert distance matrix to MLX array for GPU ops
    dist_mx = mx.array(dist_matrix)

    # 2. Iterate through subset sizes (from 2 up to N)
    # We build the solution layer by layer.
    for size in range(2, n + 1):
        # Generate all combinations of size-1 cities (excluding 0)
        # We use Numpy for fast bitmask generation on CPU
        combos = list(combinations(range(1, n), size - 1))
        combos_np = np.array(combos, dtype=np.int32)
        
        # Calculate masks: sum(1 << city) | 1
        # 1 << combos_np gives shape (num_combos, size-1)
        # sum(axis=1) gives the mask part for the subset
        # | 1 adds the start city 0
        part_masks = (1 << combos_np).sum(axis=1)
        masks_np = part_masks | 1
        
        masks = mx.array(masks_np)
        
        # Prepare to collect columns for vectorized update
        new_dp_cols = []
        new_parent_cols = []
        
        # Col 0 is dummy (inf)
        B = masks.shape[0]
        new_dp_cols.append(mx.full((B,), float('inf')))
        new_parent_cols.append(mx.full((B,), -1, dtype=mx.int32))
        
        for k in range(1, n):
            # Vectorized prev_mask calculation
            # If k is in mask, prev_mask is valid (size-1).
            # If k is NOT in mask, prev_mask is invalid (size+1), pointing to inf costs.
            prev_masks = masks ^ (1 << k)
            
            # Fetch costs: dp[prev_masks] -> (B, n)
            prev_costs = dp[prev_masks]
            
            # Fetch distances: dist_mx[:, k] -> (n,)
            dists = dist_mx[:, k]
            
            # Add: (B, n) + (n,) -> (B, n)
            total_costs = prev_costs + dists
            
            # Min and Argmin
            min_cost = mx.min(total_costs, axis=1)
            best_prev = mx.argmin(total_costs, axis=1)
            
            new_dp_cols.append(min_cost)
            new_parent_cols.append(best_prev)
            
        # Stack to create (B, n) update block
        update_vals = mx.stack(new_dp_cols, axis=1)
        update_parents = mx.stack(new_parent_cols, axis=1)
        
        # Scatter update
        dp[masks] = update_vals
        parent[masks] = update_parents
        
        # Crucial: Eval to clear graph and free resources
        mx.eval(dp, parent)

    # 3. Final Step: Return to start (City 0)
    # We look at the mask with ALL cities visited ((1<<n) - 1)
    full_mask = (1 << n) - 1
    
    # Calculate cost to return to 0 from any end city k
    last_costs = dp[full_mask, 1:] + dist_mx[1:, 0]
    
    final_cost = mx.min(last_costs).item()
    last_city_index = mx.argmin(last_costs).item() + 1 # +1 because we sliced 1:
    
    # 4. Reconstruct Path Backwards
    path = [0]
    curr_city = last_city_index
    curr_mask = full_mask
    
    # Trace back from the end
    for _ in range(n - 1):
        path.append(curr_city)
        new_city = parent[curr_mask, curr_city].item()
        curr_mask = curr_mask ^ (1 << curr_city)
        curr_city = new_city
        
    path.append(0) # Start city
    
    return final_cost, path[::-1] # Reverse to get Start -> End

def solve_tsp_held_karp_cpu(dist_matrix):
    """
    Solves TSP using Held-Karp (Dynamic Programming) on CPU (NumPy).
    Vectorized inner loop for performance.
    """
    n = len(dist_matrix)
    
    # dp[mask][k] = min cost to visit set 'mask' ending at 'k'
    dp = np.full((1 << n, n), float('inf'))
    parent = np.full((1 << n, n), -1, dtype=np.int32)
    
    dp[1, 0] = 0
    
    for size in range(2, n + 1):
        for subset in combinations(range(1, n), size - 1):
            subset = (0,) + subset
            mask = sum(1 << c for c in subset)
            
            for k in subset:
                if k == 0: continue
                prev_mask = mask ^ (1 << k)
                
                # Vectorized lookup:
                # dp[prev_mask] contains costs to reach prev_mask ending at any m
                # dist_matrix[:, k] contains costs from any m to k
                # We sum them and find the min
                costs = dp[prev_mask] + dist_matrix[:, k]
                best_prev = np.argmin(costs)
                
                dp[mask, k] = costs[best_prev]
                parent[mask, k] = best_prev
                
    full_mask = (1 << n) - 1
    
    # Return to start (0)
    costs = dp[full_mask] + dist_matrix[:, 0]
    last_city = np.argmin(costs)
    min_cost = costs[last_city]
    
    # Reconstruct path
    path = [0]
    curr = last_city
    curr_mask = full_mask
    
    for _ in range(n - 1):
        path.append(curr)
        prev = parent[curr_mask, curr]
        curr_mask ^= (1 << curr)
        curr = prev
    path.append(0)
    
    return min_cost, path[::-1]

def solve_tsp_decomposition(solver, coords, k=5):
    """
    Solves TSP using a Divide and Conquer approach with K-Means clustering.
    Sub-problems are solved using the Refined SPSA solver.
    """
    if coords is None:
        return None, None, None, None

    n = len(coords)
    if n < k:
        k = max(1, n // 2)
        
    # 1. Decomposition Phase
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(coords)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 2. Conquer Phase: Solve high-level TSP for centroids
    # Calculate centroid distance matrix
    centroid_dist_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            centroid_dist_matrix[i][j] = np.linalg.norm(centroids[i] - centroids[j])
            
    # Use Brute Force for centroids (K is small)
    c_solver = SingleQubitTSP(centroid_dist_matrix)
    c_path, _ = c_solver.solve_brute_force()
    
    # Remove return-to-start for iteration
    if c_path[0] == c_path[-1]:
        c_path = c_path[:-1]

    # 3. Stitching Phase
    full_path = []
    
    for cluster_idx in c_path:
        indices = np.where(labels == cluster_idx)[0]
        if len(indices) == 0: continue
        
        if len(indices) == 1:
            full_path.append(indices[0])
            continue
            
        # Create sub-problem
        sub_matrix = solver.B[np.ix_(indices, indices)]
        sub_solver = SingleQubitTSP(sub_matrix)
        
        # Use Refined SPSA for sub-problem (faster than Hybrid for sub-loops)
        # We use fewer trials/iterations for speed
        sp_path, _ = sub_solver.solve_refined(trials=1, iterations=300)
        
        # Map local indices back to global indices
        # sp_path includes return to start, remove it
        for local_idx in sp_path[:-1]:
            full_path.append(indices[local_idx])
            
    full_path.append(full_path[0]) # Close the loop
    
    # Calculate total cost
    cost = sum(solver.B[full_path[i]][full_path[i+1]] for i in range(len(full_path)-1))
    return full_path, cost, labels, centroids

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

    def montecarlo_explore_mlx(self, batch_size=1_000_000):
        """Finds the optimal path using MLX-based random batch search (Monte Carlo)."""
        
        # Calculate total permutations (N-1)!
        total_perms = factorial(self.n - 1)
        
        # For N=12 (40M perms), we need ~40 batches of 1M.
        # For N=13 (480M perms), we need ~480 batches.
        # We cap samples at 500M to prevent infinite runtimes for N>=14.
        max_samples = 500_000_000
        # We aim for 1.5x coverage to account for random collisions
        target_samples = min(int(total_perms * 1.5), max_samples)
        
        iterations = (target_samples + batch_size - 1) // batch_size
        
        print(f"Monte Carlo (MLX): Sampling {target_samples:.1e} paths ({iterations} batches)...")
        
        dist_matrix_mx = mx.array(self.B)
        batch_indices = mx.arange(batch_size)
        
        global_best_dist = float('inf')
        global_best_perm = None
        
        start_time = time.time()
        
        for i in range(iterations):
            best_dist, best_perm = solve_tsp_batch(batch_indices, self.n, dist_matrix_mx)
            mx.eval(best_dist, best_perm)
            
            cost = best_dist.item()
            if cost < global_best_dist:
                global_best_dist = cost
                global_best_perm = best_perm
            
            if iterations > 5 and (i % (iterations // 10 + 1) == 0):
                print_progress(i + 1, iterations, start_time, prefix="MC Sampling:")
        
        print_progress(iterations, iterations, start_time, prefix="MC Sampling:")
        print()
        
        # Convert to Python/Numpy
        path_np = np.array(global_best_perm.tolist())
        
        # Rotate to start at 0
        zero_idx = np.where(path_np == 0)[0][0]
        path_ordered = np.concatenate((path_np[zero_idx:], path_np[:zero_idx]))
        path_final = path_ordered.tolist() + [0]
        
        return path_final, global_best_dist

    def solve_brute_force(self):
        """Finds the exact optimal path using brute-force search (itertools)."""
        cities = list(range(1, self.n))
        min_cost = float('inf')
        best_path = []

        total_permutations = factorial(self.n - 1)
        start_time = time.time()
        
        for k, perm in enumerate(itertools.permutations(cities)):
            current_path = [0] + list(perm) + [0]
            current_cost = 0
            for i in range(len(current_path) - 1):
                current_cost += self.B[current_path[i]][current_path[i+1]]
            
            if current_cost < min_cost:
                min_cost = current_cost
                best_path = current_path

            if k % 100000 == 0 or k == total_permutations - 1:
                print_progress(k + 1, total_permutations, start_time, prefix="Brute Force:")
        print()
        
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

def plot_decomposition_result(coords, labels, centroids, path, filename):
    """Visualizes the clusters and the global stitched path."""
    plt.figure(figsize=(10, 7))
    k = len(centroids)
    colors = plt.cm.rainbow(np.linspace(0, 1, k))
    
    for i in range(k):
        cluster_pts = coords[labels == i]
        plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], color=colors[i], label=f'Cluster {i}', s=50)
        plt.scatter(centroids[i, 0], centroids[i, 1], color=colors[i], marker='x', s=100, linewidths=3)

    # Draw the global path
    path_coords = coords[path]
    plt.plot(path_coords[:, 0], path_coords[:, 1], 'k--', alpha=0.6, linewidth=1.5, label='Global Stitched Path')
    plt.title(f"TSP Decomposition: {len(coords)} Cities, {k} Clusters")
    plt.legend()
    plt.savefig(filename)
    plt.close()

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
    return dist_matrix, coords

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
    ("4-City Symmetric", dist_4_city, None),
    ("5-City Asymmetric", dist_5_asymmetric, None),
    ("8-City Symmetric", dist_8_city, None),
    ("6-City Random", *generate_random_tsp(6)),
    ("7-City Random", *generate_random_tsp(7)),
    ("10-City Random", *generate_random_tsp(10)),
    ("11-City Random", *generate_random_tsp(11)),
    ("12-City Random", *generate_random_tsp(12)),
    ("13-City Random", *generate_random_tsp(13)),
    ("17-City Random", *generate_random_tsp(17)),
    ("20-City Random", *generate_random_tsp(20)),
    ("30-City Random", *generate_random_tsp(30)),
    ("40-City Random", *generate_random_tsp(40)),
    ("50-City Random", *generate_random_tsp(50)),
    ("75-City Random", *generate_random_tsp(75)),
    ("100-City Random", *generate_random_tsp(100))
]

results_summary = []
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for name, matrix, coords in test_cases:
    print(f"\n--- {name} ---")
    solver = SingleQubitTSP(matrix)
    
    best_path = None
    best_cost = None
    hybrid_time = None

    if solver.n <= 50:
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

    decomp_cost = None
    decomp_time = None
    decomp_path = None
    if coords is not None and solver.n >= 10:
        print("Solving with Decomposition (K-Means + Refined SPSA)...")
        start_time = time.time()
        # Determine K based on N (e.g., roughly 5-10 cities per cluster)
        k_clusters = max(2, int(solver.n / 8))
        decomp_path, decomp_cost, labels, centroids = solve_tsp_decomposition(solver, coords, k=k_clusters)
        decomp_time = time.time() - start_time
        print(f"Decomposition Path: {' -> '.join(map(str, decomp_path))}")
        print(f"Decomposition Cost: {decomp_cost:.2f}")
    else:
        print(f"Skipping Decomposition for {name} (No coords or N too small).")

    mc_cost = None
    mc_time = None
    mc_path = None
    # Limit Monte Carlo to n < 50 due to computational constraints
    if solver.n < 50:
        print("Calculating approximate solution (Monte Carlo MLX)...")
        start_time = time.time()
        mc_path, mc_cost = solver.montecarlo_explore_mlx()
        mc_time = time.time() - start_time
        print(f"Monte Carlo Path: {' -> '.join(map(str, mc_path))}")
        print(f"Monte Carlo Cost: {mc_cost:.2f}")
    else:
        print(f"Skipping Monte Carlo for {solver.n} cities.")

    hk_cost = None
    hk_time = None
    hk_path = None
    # Limit Held-Karp to n < 30 due to computational constraints
    # RuntimeError: [metal::malloc] Attempting to allocate 128849018880 bytes which is greater than the maximum allowed buffer size of 17179869184 bytes.
    # This occurs around n=30 due to 2^n memory requirements.
    if solver.n < 30:
        print("Calculating exact solution (Held-Karp MLX)...")
        start_time = time.time()
        hk_cost, hk_path = solve_tsp_held_karp_mlx(solver.B)
        hk_time = time.time() - start_time
        print(f"Held-Karp Path: {' -> '.join(map(str, hk_path))}")
        print(f"Held-Karp Cost: {hk_cost:.2f}")
    else:
        print(f"Skipping Held-Karp for {solver.n} cities.")

    hk_cpu_cost = None
    hk_cpu_time = None
    hk_cpu_path = None
    if solver.n <= 16:
        print("Calculating exact solution (Held-Karp CPU)...")
        start_time = time.time()
        hk_cpu_cost, hk_cpu_path = solve_tsp_held_karp_cpu(solver.B)
        hk_cpu_time = time.time() - start_time
        print(f"Held-Karp CPU Path: {' -> '.join(map(str, hk_cpu_path))}")
        print(f"Held-Karp CPU Cost: {hk_cpu_cost:.2f}")
    else:
        print(f"Skipping Held-Karp CPU for {solver.n} cities.")

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
        "Decomposition": (decomp_cost, decomp_time, decomp_path),
        "MonteCarlo": (mc_cost, mc_time, mc_path),
        "HeldKarp": (hk_cost, hk_time, hk_path),
        "HeldKarpCPU": (hk_cpu_cost, hk_cpu_time, hk_cpu_path),
        "BruteForce": (bf_cost, bf_time, bf_path)
    })

print("\n" + "="*215)
print(f"{'TEST CASE':<20} | {'HYBRID (Cost/Time)':<20} | {'REFINED (Cost/Time)':<20} | {'REF+2OPT (Cost/Time)':<20} | {'DECOMP (Cost/Time)':<20} | {'MONTE CARLO':<20} | {'HELD KARP':<20} | {'HK CPU':<20} | {'BRUTE FORCE':<15}")
print("-" * 215)
for res in results_summary:
    h_c, h_t, _ = res["Hybrid"]
    r_c, r_t, _ = res["Refined"]
    r2_c, r2_t, _ = res["Refined+2Opt"]
    dc_c, dc_t, _ = res["Decomposition"]
    mc_c, mc_t, _ = res["MonteCarlo"]
    hk_c, hk_t, _ = res["HeldKarp"]
    hk_cpu_c, hk_cpu_t, _ = res["HeldKarpCPU"]
    bf_c, bf_t, _ = res["BruteForce"]
    
    if h_c is not None:
        h_str = f"{h_c:.2f} / {h_t:.2f}s"
    else:
        h_str = "N/A"
    r_str = f"{r_c:.2f} / {r_t:.2f}s"
    r2_str = f"{r2_c:.2f} / {r2_t:.2f}s"
    
    if dc_c is not None:
        dc_str = f"{dc_c:.2f} / {dc_t:.2f}s"
    else:
        dc_str = "N/A"

    if mc_c is not None:
        mc_str = f"{mc_c:.2f} / {mc_t:.2f}s"
    else:
        mc_str = "N/A"

    if hk_c is not None:
        hk_str = f"{hk_c:.2f} / {hk_t:.2f}s"
    else:
        hk_str = "N/A"

    if hk_cpu_c is not None:
        hk_cpu_str = f"{hk_cpu_c:.2f} / {hk_cpu_t:.2f}s"
    else:
        hk_cpu_str = "N/A"

    if bf_c is not None:
        bf_str = f"{bf_c:.2f} / {bf_t:.4f}s"
    else:
        bf_str = "N/A"
    
    print(f"{res['Case']:<20} | {h_str:<20} | {r_str:<20} | {r2_str:<20} | {dc_str:<20} | {mc_str:<20} | {hk_str:<20} | {hk_cpu_str:<20} | {bf_str:<15}")
print("="*215 + "\n")

# --- Export Results to File ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(output_dir, f"tsp_results_{timestamp}.txt")

with open(filename, "w") as f:
    f.write(f"TSP Optimization Results - {timestamp}\n")
    f.write("="*215 + "\n")
    f.write(f"{'TEST CASE':<20} | {'HYBRID (Cost/Time)':<20} | {'REFINED (Cost/Time)':<20} | {'REF+2OPT (Cost/Time)':<20} | {'DECOMP (Cost/Time)':<20} | {'MONTE CARLO':<20} | {'HELD KARP':<20} | {'HK CPU':<20} | {'BRUTE FORCE':<15}\n")
    f.write("-" * 215 + "\n")
    
    for res in results_summary:
        h_c, h_t, _ = res["Hybrid"]
        r_c, r_t, _ = res["Refined"]
        r2_c, r2_t, _ = res["Refined+2Opt"]
        dc_c, dc_t, _ = res["Decomposition"]
        mc_c, mc_t, _ = res["MonteCarlo"]
        hk_c, hk_t, _ = res["HeldKarp"]
        hk_cpu_c, hk_cpu_t, _ = res["HeldKarpCPU"]
        bf_c, bf_t, _ = res["BruteForce"]
        
        if h_c is not None:
            h_str = f"{h_c:.2f} / {h_t:.2f}s"
        else:
            h_str = "N/A"
        r_str = f"{r_c:.2f} / {r_t:.2f}s"
        r2_str = f"{r2_c:.2f} / {r2_t:.2f}s"
        
        if dc_c is not None:
            dc_str = f"{dc_c:.2f} / {dc_t:.2f}s"
        else:
            dc_str = "N/A"

        if mc_c is not None:
            mc_str = f"{mc_c:.2f} / {mc_t:.2f}s"
        else:
            mc_str = "N/A"

        if hk_c is not None:
            hk_str = f"{hk_c:.2f} / {hk_t:.2f}s"
        else:
            hk_str = "N/A"

        if hk_cpu_c is not None:
            hk_cpu_str = f"{hk_cpu_c:.2f} / {hk_cpu_t:.2f}s"
        else:
            hk_cpu_str = "N/A"

        if bf_c is not None:
            bf_str = f"{bf_c:.2f} / {bf_t:.4f}s"
        else:
            bf_str = "N/A"
        
        f.write(f"{res['Case']:<20} | {h_str:<20} | {r_str:<20} | {r2_str:<20} | {dc_str:<20} | {mc_str:<20} | {hk_str:<20} | {hk_cpu_str:<20} | {bf_str:<15}\n")
    
    f.write("="*215 + "\n\n")
    f.write("DETAILED RESULTS\n")
    f.write("="*215 + "\n")

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
        if res['Decomposition'][0] is not None:
            f.write(f"Decomposition:   Cost={res['Decomposition'][0]:.2f}, Time={res['Decomposition'][1]:.4f}s, Path={res['Decomposition'][2]}\n")
        else:
            f.write("Decomposition:   Skipped\n")
        if res['MonteCarlo'][0] is not None:
            f.write(f"Monte Carlo:     Cost={res['MonteCarlo'][0]:.2f}, Time={res['MonteCarlo'][1]:.4f}s, Path={res['MonteCarlo'][2]}\n")
        else:
            f.write("Monte Carlo:     Skipped\n")
        if res['HeldKarp'][0] is not None:
            f.write(f"Held-Karp:       Cost={res['HeldKarp'][0]:.2f}, Time={res['HeldKarp'][1]:.4f}s, Path={res['HeldKarp'][2]}\n")
        else:
            f.write("Held-Karp:       Skipped\n")
        if res['HeldKarpCPU'][0] is not None:
            f.write(f"Held-Karp CPU:   Cost={res['HeldKarpCPU'][0]:.2f}, Time={res['HeldKarpCPU'][1]:.4f}s, Path={res['HeldKarpCPU'][2]}\n")
        else:
            f.write("Held-Karp CPU:   Skipped\n")
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
decomp_costs, decomp_times = [], []
mc_costs, mc_times = [], []
hk_costs, hk_times = [], []
hk_cpu_costs, hk_cpu_times = [], []
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
    
    # Decomposition
    dc, dt, _ = res['Decomposition']
    decomp_costs.append(dc if dc is not None else np.nan)
    decomp_times.append(dt if dt is not None else np.nan)

    # Monte Carlo
    mcc, mct, _ = res['MonteCarlo']
    mc_costs.append(mcc if mcc is not None else np.nan)
    mc_times.append(mct if mct is not None else np.nan)

    # Held-Karp
    hkc, hkt, _ = res['HeldKarp']
    hk_costs.append(hkc if hkc is not None else np.nan)
    hk_times.append(hkt if hkt is not None else np.nan)

    # Held-Karp CPU
    hkcc, hkct, _ = res['HeldKarpCPU']
    hk_cpu_costs.append(hkcc if hkcc is not None else np.nan)
    hk_cpu_times.append(hkct if hkct is not None else np.nan)

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
decomp_costs = np.array(decomp_costs)[perm]
decomp_times = np.array(decomp_times)[perm]
mc_costs = np.array(mc_costs)[perm]
mc_times = np.array(mc_times)[perm]
hk_costs = np.array(hk_costs)[perm]
hk_times = np.array(hk_times)[perm]
hk_cpu_costs = np.array(hk_cpu_costs)[perm]
hk_cpu_times = np.array(hk_cpu_times)[perm]
bf_costs = np.array(bf_costs)[perm]
bf_times = np.array(bf_times)[perm]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Cost Plot
ax1.plot(n_vals, hybrid_costs, 'o-', label='Hybrid SPSA')
ax1.plot(n_vals, refined_costs, 's-', label='Refined SPSA')
ax1.plot(n_vals, r2opt_costs, '^-', label='Refined + 2-Opt')
ax1.plot(n_vals, decomp_costs, 'p-', label='Decomposition', color='green')
ax1.plot(n_vals, mc_costs, 'd--', label='Monte Carlo', color='purple', alpha=0.7)
ax1.plot(n_vals, hk_costs, '*--', label='Held-Karp', color='orange', alpha=0.8)
ax1.plot(n_vals, hk_cpu_costs, '+--', label='Held-Karp CPU', color='brown', alpha=0.8)
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
ax2.plot(n_vals, decomp_times, 'p-', label='Decomposition', color='green')
ax2.plot(n_vals, mc_times, 'd--', label='Monte Carlo', color='purple', alpha=0.7)
ax2.plot(n_vals, hk_times, '*--', label='Held-Karp', color='orange', alpha=0.8)
ax2.plot(n_vals, hk_cpu_times, '+--', label='Held-Karp CPU', color='brown', alpha=0.8)
ax2.plot(n_vals, bf_times, 'x--', label='Brute Force', color='k', alpha=0.6)
ax2.set_title('Time vs Number of Cities')
ax2.set_xlabel('Number of Cities')
ax2.set_ylabel('Time (s)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True)

plot_filename = os.path.join(output_dir, f"tsp_results_{timestamp}.png")
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")

# --- Generate Decomposition Plot for the last applicable case ---
if decomp_path is not None and coords is not None:
    print("Generating Decomposition visualization...")
    decomp_plot_filename = os.path.join(output_dir, f"tsp_decomposition_{timestamp}.png")
    plot_decomposition_result(coords, labels, centroids, decomp_path, decomp_plot_filename)
    print(f"Decomposition plot saved to {decomp_plot_filename}")