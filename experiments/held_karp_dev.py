"""
Standalone Held-Karp development script.

Original implementation used during algorithm development — runs independently
without the production package. Kept for reference and debugging.

Note: Return order here is (cost, path) — the production version in
algorithms/held_karp.py uses the standardized (path, cost) order.

Run from anywhere:
    python experiments/held_karp_dev.py
"""
import mlx.core as mx
import numpy as np
from itertools import combinations


def solve_tsp_held_karp_mlx(dist_matrix):
    """
    Standalone Held-Karp DP solver using Apple Silicon GPU (MLX).

    Returns:
        (cost, path): Note the legacy return order — see algorithms/held_karp.py
        for the production version which returns (path, cost).
    """
    n = dist_matrix.shape[0]

    dp = mx.full((1 << n, n), float('inf'))
    parent = mx.full((1 << n, n), -1, dtype=mx.int32)
    dp[1, 0] = 0.0

    dist_mx = mx.array(dist_matrix)

    for size in range(2, n + 1):
        subset_indices = [0] + list(range(1, n))
        for subset in combinations(range(1, n), size - 1):
            current_subset = (0,) + subset
            mask = sum(1 << city for city in current_subset)

            for k in subset:
                prev_mask = mask ^ (1 << k)
                prev_costs = dp[prev_mask]          # Shape: (n,)
                dists_to_k = dist_mx[:, k]           # Shape: (n,)
                total_costs = prev_costs + dists_to_k
                min_cost = mx.min(total_costs)
                best_prev = mx.argmin(total_costs)
                dp[mask, k] = min_cost
                parent[mask, k] = best_prev

    full_mask = (1 << n) - 1
    last_costs = dp[full_mask, 1:] + dist_mx[1:, 0]
    final_cost = mx.min(last_costs).item()
    last_city_index = mx.argmin(last_costs).item() + 1

    path = [0]
    curr_city = last_city_index
    curr_mask = full_mask
    for _ in range(n - 1):
        path.append(curr_city)
        new_city = parent[curr_mask, curr_city].item()
        curr_mask = curr_mask ^ (1 << curr_city)
        curr_city = new_city
    path.append(0)

    return final_cost, path[::-1]


# ── Driver ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    num_cities = 12
    np.random.seed(42)
    coords = np.random.rand(num_cities, 2)

    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    print(f"Solving TSP for {num_cities} cities using MLX Held-Karp...")
    cost, path = solve_tsp_held_karp_mlx(dist_matrix)
    print(f"Optimal Cost: {cost:.4f}")
    print(f"Optimal Path: {path}")
