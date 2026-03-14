import mlx.core as mx
import numpy as np
from itertools import combinations


def _reconstruct_path(
    parent_np: np.ndarray, last_city: int, full_mask: int, n: int
) -> list[int]:
    """Shared path reconstruction for both MLX and CPU Held-Karp variants.

    Args:
        parent_np: (2^n, n) parent table as a NumPy array.
        last_city: Index of the last city before returning to 0.
        full_mask: Bitmask with all n cities set.
        n: Number of cities.

    Returns:
        Reconstructed tour as a list of city indices (starts and ends at 0).
    """
    path = [0]
    curr, curr_mask = last_city, full_mask
    for _ in range(n - 1):
        path.append(curr)
        prev = int(parent_np[curr_mask, curr])
        curr_mask ^= (1 << curr)
        curr = prev
    path.append(0)
    return path[::-1]


def solve_tsp_held_karp_mlx(dist_matrix: np.ndarray) -> tuple[list[int], float]:
    """Solves TSP using Held-Karp (Dynamic Programming) on Apple Silicon GPU (MLX).

    Time complexity: O(n² × 2ⁿ). Practical limit: n < 30 due to 2ⁿ memory.

    Args:
        dist_matrix: NxN distance matrix.

    Returns:
        (path, cost): Optimal tour and its total cost.
    """
    n = int(dist_matrix.shape[0])

    # Initialize DP and parent tables (rows = 2^n subsets, cols = n cities)
    dp = mx.full((1 << n, n), float("inf"))
    parent = mx.full((1 << n, n), -1, dtype=mx.int32)

    # Base case: start at city 0 with only city 0 visited (mask = 0b...001)
    dp[1, 0] = 0.0

    dist_mx = mx.array(dist_matrix)

    # Build solution layer by layer (subset sizes from 2 to N)
    for size in range(2, n + 1):
        combos = list(combinations(range(1, n), size - 1))
        combos_np = np.array(combos, dtype=np.int32)

        part_masks = (1 << combos_np).sum(axis=1)
        masks_np = part_masks | 1

        masks = mx.array(masks_np)

        new_dp_cols = []
        new_parent_cols = []

        B = masks.shape[0]
        new_dp_cols.append(mx.full((B,), float("inf")))
        new_parent_cols.append(mx.full((B,), -1, dtype=mx.int32))

        for k in range(1, n):
            prev_masks = masks ^ (1 << k)
            prev_costs = dp[prev_masks]
            dists = dist_mx[:, k]

            total_costs = prev_costs + dists
            min_cost = mx.min(total_costs, axis=1)
            best_prev = mx.argmin(total_costs, axis=1)

            new_dp_cols.append(min_cost)
            new_parent_cols.append(best_prev)

        update_vals = mx.stack(new_dp_cols, axis=1)
        update_parents = mx.stack(new_parent_cols, axis=1)

        dp[masks] = update_vals
        parent[masks] = update_parents

        # Eval to clear graph and free GPU resources
        mx.eval(dp, parent)

    # Final step: return to city 0 from the best ending city
    full_mask = (1 << n) - 1
    last_costs = dp[full_mask, 1:] + dist_mx[1:, 0]

    final_cost = float(mx.min(last_costs).item())
    last_city = int(mx.argmin(last_costs).item()) + 1  # +1 because we sliced [1:]

    # Reconstruct path using shared helper (convert MLX parent to numpy first)
    parent_np = np.array(parent.tolist())
    path = _reconstruct_path(parent_np, last_city, full_mask, n)

    return path, final_cost


def solve_tsp_held_karp_cpu(dist_matrix: np.ndarray) -> tuple[list[int], float]:
    """Solves TSP using Held-Karp (Dynamic Programming) on CPU (NumPy).

    Vectorized inner loop for performance.
    Time complexity: O(n² × 2ⁿ). Practical limit: n ≤ 16 for reasonable speed.

    Args:
        dist_matrix: NxN distance matrix.

    Returns:
        (path, cost): Optimal tour and its total cost.
    """
    n = len(dist_matrix)

    # dp[mask][k] = min cost to visit set 'mask' ending at city k
    dp = np.full((1 << n, n), float("inf"))
    parent = np.full((1 << n, n), -1, dtype=np.int32)

    dp[1, 0] = 0

    for size in range(2, n + 1):
        for subset in combinations(range(1, n), size - 1):
            subset = (0,) + subset
            mask = sum(1 << c for c in subset)

            for k in subset:
                if k == 0:
                    continue
                prev_mask = mask ^ (1 << k)

                # Vectorized: find the best predecessor city
                costs = dp[prev_mask] + dist_matrix[:, k]
                best_prev = np.argmin(costs)

                dp[mask, k] = costs[best_prev]
                parent[mask, k] = best_prev

    full_mask = (1 << n) - 1

    # Return to start (city 0)
    costs = dp[full_mask] + dist_matrix[:, 0]
    last_city = int(np.argmin(costs))
    min_cost = float(costs[last_city])

    path = _reconstruct_path(parent, last_city, full_mask, n)

    return path, min_cost
