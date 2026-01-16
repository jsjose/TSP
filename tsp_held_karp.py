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