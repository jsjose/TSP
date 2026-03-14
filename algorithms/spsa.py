import numpy as np
import time
import sys
import itertools
from math import factorial

import mlx.core as mx

from utils.helpers import print_progress, calculate_total_distance
from .montecarlo import solve_tsp_batch


class SingleQubitTSP:
    """Quantum-inspired TSP solver using Simultaneous Perturbation Stochastic Approximation.

    Each city's preference weights are a softmax over an n×n parameter matrix,
    optimized via SPSA with optional 2nd-order curvature correction.

    Args:
        cost_matrix: NxN distance/cost matrix.
    """

    def __init__(self, cost_matrix) -> None:
        self.B = np.array(cost_matrix)
        self.n = len(cost_matrix)
        self.phi = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        self.normalize_distances()

    def normalize_distances(self) -> None:
        """Scales distances to [0.1, 1.4] for optimal Bloch sphere mapping."""
        min_val = np.min(self.B[self.B > 0]) if np.any(self.B > 0) else 0
        max_val = np.max(self.B)
        self.normalized_B = 0.1 + (self.B - min_val) * (1.3 / (max_val - min_val))
        np.fill_diagonal(self.normalized_B, 0)

    def cost_function(self, params: np.ndarray) -> float:
        """Calculates expected cost using softmax to simulate qubit state populations.

        Args:
            params: (n, n) parameter matrix encoding edge preferences.

        Returns:
            Scalar expected tour cost.
        """
        matrix = getattr(self, "normalized_B", self.B)
        total_dist = 0
        for i in range(self.n):
            row = params[i]
            # Numerically stable softmax
            weights = np.exp(row - np.max(row))
            weights /= np.sum(weights)
            total_dist += np.sum(weights * matrix[i])
        return total_dist

    def decode_path(self, params: np.ndarray) -> list[int]:
        """Greedy decoding to ensure a valid Hamiltonian cycle.

        Args:
            params: (n, n) parameter matrix.

        Returns:
            Tour as a list of city indices, starting and ending at 0.
        """
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

    def get_spsa_params(
        self, k: int, a: float = 0.1, c: float = 0.01, A: int = 10
    ) -> tuple[float, float]:
        """Calculates decaying step sizes according to the SPSA paper's schedule.

        Args:
            k: Current iteration index.
            a: Step size scaling constant.
            c: Perturbation size scaling constant.
            A: Stability constant.

        Returns:
            (ak, ck): Step size and perturbation size for iteration k.
        """
        ak = a / (k + 1 + A) ** 0.602
        ck = c / (k + 1) ** 0.101
        return ak, ck

    def update_with_momentum(
        self,
        params: np.ndarray,
        grad: np.ndarray,
        momentum: np.ndarray,
        beta: float = 0.9,
        ak: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Applies momentum-based gradient update.

        Args:
            params: Current parameter matrix.
            grad: Estimated gradient.
            momentum: Current momentum vector.
            beta: Momentum decay factor.
            ak: Learning rate.

        Returns:
            (updated_params, updated_momentum).
        """
        momentum = beta * momentum + (1 - beta) * grad
        params = params - ak * momentum
        return params, momentum

    def solve_hybrid(
        self, iters_1st: int = 1000, iters_2nd: int = 500
    ) -> tuple[list[int], float]:
        """Two-phase SPSA optimization followed by 2-opt refinement.

        Phase 1: 1SPSA (global exploration).
        Phase 2: 2SPSA (curvature-aware fine-tuning).
        Phase 3: 2-opt (classical edge-swap refinement).

        Args:
            iters_1st: Number of iterations for Phase 1.
            iters_2nd: Number of iterations for Phase 2.

        Returns:
            (path, cost): Best tour found and its total cost.
        """
        params = np.random.randn(self.n, self.n) * 0.1
        hessian = np.eye(self.n**2)
        start_time = time.time()

        # --- Phase 1: 1SPSA (Global Exploration) ---
        for k in range(1, iters_1st + 1):
            ak = 0.1 / (k + 10) ** 0.602
            ck = 0.01 / k ** 0.101
            delta = np.random.choice([-1, 1], size=params.shape)

            grad = (
                self.cost_function(params + ck * delta)
                - self.cost_function(params - ck * delta)
            ) / (2 * ck * delta)
            params -= ak * grad

            if k % 50 == 0:
                print_progress(k, iters_1st + iters_2nd, start_time, prefix="Hybrid Phase 1 (1SPSA):")

        # --- Phase 2: 2SPSA (Precision Convergence) ---
        for k in range(iters_1st + 1, iters_1st + iters_2nd + 1):
            ak = 0.05 / (k + 10) ** 0.602
            ck = 0.01 / k ** 0.101
            delta = np.random.choice([-1, 1], size=params.shape)
            tilde_delta = np.random.choice([-1, 1], size=params.shape)

            y_plus = self.cost_function(params + ck * delta)
            y_minus = self.cost_function(params - ck * delta)
            y_plus_h = self.cost_function(params + ck * delta + ck * tilde_delta)
            y_minus_h = self.cost_function(params - ck * delta + ck * tilde_delta)

            grad = (y_plus - y_minus) / (2 * ck * delta)
            g_diff = ((y_plus_h - y_plus) / ck - (y_minus_h - y_minus) / ck).flatten()
            delta_flat = delta.flatten()

            H_k = 0.5 * (np.outer(g_diff, delta_flat) + np.outer(delta_flat, g_diff))
            hessian = 0.9 * hessian + 0.1 * H_k

            # Matrix regularization and inversion
            inv_H = np.linalg.inv(hessian + (1e-4) * np.eye(len(hessian)))
            params -= ak * (inv_H @ grad.flatten()).reshape(params.shape)

            if k % 50 == 0:
                print_progress(k, iters_1st + iters_2nd, start_time, prefix="Hybrid Phase 2 (2SPSA):")

        print("\nOptimization complete.")

        # --- Phase 3: 2-Opt Refinement ---
        rough_path = self.decode_path(params)
        print("Starting Phase 3 (2-Opt Refinement)...")
        final_path, final_cost = self.run_2opt(rough_path)

        return final_path, final_cost

    def run_2opt(self, path: list[int]) -> tuple[list[int], float]:
        """Iteratively untangles edge crossings via 2-opt swaps.

        Delegates to the standalone two_opt_refinement() function.

        Args:
            path: Current tour as list of city indices.

        Returns:
            (improved_path, cost): Locally optimal tour and its cost.
        """
        return two_opt_refinement(path, self.B)

    def montecarlo_explore_mlx(
        self, batch_size: int = 1_000_000
    ) -> tuple[list[int], float]:
        """Finds a good path via MLX-based random batch sampling (Monte Carlo).

        Targets 1.5× coverage of the total permutation space, capped at 500M samples.

        Args:
            batch_size: Number of random permutations to evaluate per batch.

        Returns:
            (path, cost): Best tour found and its total cost.
        """
        total_perms = factorial(self.n - 1)

        max_samples = 500_000_000
        target_samples = min(int(total_perms * 1.5), max_samples)
        iterations = (target_samples + batch_size - 1) // batch_size

        print(f"Monte Carlo (MLX): Sampling {target_samples:.1e} paths ({iterations} batches)...")

        dist_matrix_mx = mx.array(self.B)
        batch_indices = mx.arange(batch_size)

        global_best_dist = float("inf")
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

        # Convert to Python list and rotate to start at city 0
        path_np = np.array(global_best_perm.tolist())
        zero_idx = np.where(path_np == 0)[0][0]
        path_ordered = np.concatenate((path_np[zero_idx:], path_np[:zero_idx]))
        path_final = path_ordered.tolist() + [0]

        return path_final, global_best_dist

    def solve_brute_force(self) -> tuple[list[int], float]:
        """Finds the exact optimal path via brute-force enumeration (itertools).

        Only practical for n < 13 due to (n-1)! growth.

        Returns:
            (path, cost): Optimal tour and its total cost.
        """
        cities = list(range(1, self.n))
        min_cost = float("inf")
        best_path: list[int] = []

        total_permutations = factorial(self.n - 1)
        start_time = time.time()

        for k, perm in enumerate(itertools.permutations(cities)):
            current_path = [0] + list(perm) + [0]
            current_cost = calculate_total_distance(current_path, self.B)

            if current_cost < min_cost:
                min_cost = current_cost
                best_path = current_path

            if k % 100000 == 0 or k == total_permutations - 1:
                print_progress(k + 1, total_permutations, start_time, prefix="Brute Force:")
        print()

        return best_path, min_cost

    def solve_refined(
        self, trials: int = 5, iterations: int = 500
    ) -> tuple[list[int], float]:
        """Multi-start SPSA with momentum to reduce sensitivity to initialization.

        Args:
            trials: Number of independent SPSA runs.
            iterations: SPSA iterations per trial.

        Returns:
            (path, cost): Best tour found across all trials.
        """
        self.normalize_distances()
        best_global_path: list[int] | None = None
        min_global_dist = float("inf")

        start_time_global = time.time()

        for trial in range(trials):
            params = np.random.randn(self.n, self.n) * 0.1
            momentum = np.zeros_like(params)

            for k in range(iterations):
                ak, ck = self.get_spsa_params(k)
                delta = np.random.choice([-1, 1], size=params.shape)

                f_plus = self.cost_function(params + ck * delta)
                f_minus = self.cost_function(params - ck * delta)
                grad = (f_plus - f_minus) / (2 * ck * delta)

                params, momentum = self.update_with_momentum(params, grad, momentum, ak=ak)

            trial_path = self.decode_path(params)
            trial_dist = calculate_total_distance(trial_path, self.B)

            if trial_dist < min_global_dist:
                min_global_dist = trial_dist
                best_global_path = trial_path

            prefix = f"Refined SPSA Progress (Trial {trial + 1}/{trials}):"
            print_progress(trial + 1, trials, start_time_global, prefix=prefix)
        print()

        return best_global_path, min_global_dist


def two_opt_refinement(
    path: list[int], dist_matrix: np.ndarray
) -> tuple[list[int], float]:
    """Iteratively improves a TSP path by swapping edges (2-opt).

    Restarts the full scan on each improvement (first-improvement strategy).

    Args:
        path: Initial tour as list of city indices.
        dist_matrix: NxN distance matrix.

    Returns:
        (path, cost): Locally optimal tour and its total cost.
    """
    best_path = list(path)
    best_dist = calculate_total_distance(best_path, dist_matrix)
    improved = True

    while improved:
        improved = False
        start_time = time.time()
        total_steps = len(best_path) - 3
        for i in range(1, len(best_path) - 2):
            print_progress(i, total_steps, start_time, prefix="2-opt Refinement:")
            for j in range(i + 1, len(best_path) - 1):
                new_path = best_path[:i] + best_path[i:j + 1][::-1] + best_path[j + 1:]
                new_dist = calculate_total_distance(new_path, dist_matrix)

                if new_dist < best_dist:
                    best_path = new_path
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    print()

    return best_path, best_dist
