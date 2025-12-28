import mlx.core as mx
import numpy as np
from itertools import combinations

def solve_tsp_held_karp_mlx(dist_matrix):
    """
    Solves TSP using Held-Karp (Dynamic Programming) on Apple Silicon GPU (MLX).
    
    Args:
        dist_matrix (np.ndarray): NxN distance matrix.
        
    Returns:
        (cost, path): Tuple containing the minimum cost and the optimal path list.
    """
    n = dist_matrix.shape[0]
    
    # 1. Initialize DP Table and Parent Table
    # Rows: 2^n subsets (bitmasks)
    # Cols: n cities (last visited city)
    # Memory: 2^20 * 20 * 4 bytes ≈ 80MB for N=20 (very efficient)
    
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
        # Generate all combinations of cities for this size that include city 0
        # We use Python's itertools because mask generation is fast enough on CPU
        # and hard to vectorize cleanly.
        subset_indices = [0] + list(range(1, n))
        for subset in combinations(range(1, n), size - 1):
            # The current subset includes 0 + the combinations
            current_subset = (0,) + subset
            
            # Create the bitmask integer for this subset
            # e.g., {0, 2} -> binary 101 -> 5
            mask = sum(1 << city for city in current_subset)
            
            # We process this specific mask. 
            # Optimization: In a fully vectorized production code, we would 
            # batch ALL masks of 'size' together. For clarity/N<20, loop is okay.
            
            # However, to use the GPU effectively, we want to eliminate the inner loop 
            # over 'next_city'. We can't easily vectorise *across* random masks 
            # without complex gathering, but we can vectorise *within* the transition.
            
            # Target cities: k (must be in current_subset and != 0)
            for k in subset: # 'subset' does not include 0, which is correct
                prev_mask = mask ^ (1 << k)
                
                # Vectorized Cost Calculation:
                # We want: min(dp[prev_mask, m] + dist[m, k]) for all m in prev_mask
                # Since dp[prev_mask, m] is INF for m not in prev_mask, 
                # we can just scan the whole row 'prev_mask'.
                
                # 1. Fetch costs to reach the previous state ending at any city m
                prev_costs = dp[prev_mask] # Shape: (n,)
                
                # 2. Fetch distances from any city m to k
                dists_to_k = dist_mx[:, k] # Shape: (n,)
                
                # 3. Add them up
                total_costs = prev_costs + dists_to_k
                
                # 4. Find the min and argmin (best previous city)
                min_cost = mx.min(total_costs)
                best_prev = mx.argmin(total_costs)
                
                # 5. Update DP tables
                # Note: MLX arrays are immutable-ish; we update via index
                # Ideally we collect updates and do a bulk write, but MLX handles this decent enough.
                dp[mask, k] = min_cost
                parent[mask, k] = best_prev

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

# --- Driver Code to Test ---
if __name__ == "__main__":
    # Generate random distance matrix for 12 cities
    # Held-Karp is O(N^2 * 2^N), so N=20 is roughly the limit for quick tests
    num_cities = 12
    np.random.seed(42)
    coords = np.random.rand(num_cities, 2)
    
    # Euclidean distance matrix
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
            
    print(f"Solving TSP for {num_cities} cities using MLX Held-Karp...")
    
    cost, path = solve_tsp_held_karp_mlx(dist_matrix)
    
    print(f"Optimal Cost: {cost:.4f}")
    print(f"Optimal Path: {path}")