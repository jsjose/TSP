"""
Standalone SPSA research script.

Self-contained version of the SingleQubitTSP solver for teaching and reference.
Does not depend on the production package — runs from any directory.
"""
import sys
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
from math import factorial


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


def calculate_total_distance(path, dist_matrix):
    distance = 0
    for i in range(len(path) - 1):
        distance += dist_matrix[path[i]][path[i+1]]
    return distance


def two_opt_refinement(path, dist_matrix):
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
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
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


def generate_random_tsp(n_cities, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(n_cities, 2) * 100
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix


# ── Problem Instances ────────────────────────────────────────────────────────

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
    [0, 15, 10, 20, 12, 18, 14, 22],
    [15, 0, 16, 25, 13, 11, 21, 19],
    [10, 16, 0, 30, 17, 24, 15, 28],
    [20, 25, 30, 0, 22, 14, 16, 12],
    [12, 13, 17, 22, 0, 26, 18, 14],
    [18, 11, 24, 14, 26, 0, 12, 10],
    [14, 21, 15, 16, 18, 12, 0, 17],
    [22, 19, 28, 12, 14, 10, 17, 0]
]


class SingleQubitTSP:
    """Standalone (self-contained) version of the quantum-inspired SPSA solver."""

    def __init__(self, cost_matrix):
        self.B = np.array(cost_matrix)
        self.n = len(cost_matrix)
        self.phi = np.linspace(0, 2 * np.pi, self.n, endpoint=False)

    def get_city_state(self, i):
        phi = self.phi[i]
        return np.array([1, np.exp(1j * phi)]) / np.sqrt(2)

    def get_intermediate_state(self, i, j):
        phi_i = self.phi[i]
        dist = self.B[i][j]
        theta = (dist / (np.max(self.B) * 1.1)) * (np.pi / 2)
        return np.array([np.cos(theta / 2), np.exp(1j * phi_i) * np.sin(theta / 2)])

    def rotation_operator(self, target_state, current_state):
        return np.outer(target_state, current_state.conj())

    def cost_function(self, params):
        matrix = getattr(self, 'normalized_B', self.B)
        total_dist = 0
        for i in range(self.n):
            row = params[i]
            weights = np.exp(row - np.max(row))
            weights /= np.sum(weights)
            total_dist += np.sum(weights * matrix[i])
        return total_dist

    def solve_spsa(self, iterations=100, a=0.1, c=0.01):
        num_cities = self.n
        params = np.random.rand(num_cities, num_cities)
        start_time = time.time()

        def cost_fn(p):
            total_dist = 0
            for i in range(num_cities):
                weights = np.exp(p[i]) / np.sum(np.exp(p[i]))
                total_dist += np.sum(weights * self.B[i])
            return total_dist

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

        path = [0]
        visited = {0}
        for _ in range(num_cities - 1):
            curr = path[-1]
            options = np.argsort(-params[curr])
            for opt in options:
                if opt not in visited:
                    path.append(opt)
                    visited.add(opt)
                    break
        path.append(0)
        return path, params

    def simulate_state_tomography(self, params):
        probs = np.zeros_like(params)
        for i in range(self.n):
            row_exp = np.exp(params[i] - np.max(params[i]))
            probs[i] = row_exp / np.sum(row_exp)
        cities = list(range(1, self.n))
        path_probabilities = []
        for perm in itertools.permutations(cities):
            path = [0] + list(perm) + [0]
            prob = 1.0
            for k in range(len(path) - 1):
                prob *= probs[path[k]][path[k+1]]
            path_probabilities.append((path, prob))
        path_probabilities.sort(key=lambda x: x[1], reverse=True)
        return path_probabilities

    def normalize_distances(self):
        min_val = np.min(self.B[self.B > 0])
        max_val = np.max(self.B)
        self.normalized_B = 0.1 + (self.B - min_val) * (1.3 / (max_val - min_val))
        np.fill_diagonal(self.normalized_B, 0)

    def get_spsa_params(self, k, a=0.1, c=0.01, A=10):
        ak = a / (k + 1 + A) ** 0.602
        ck = c / (k + 1) ** 0.101
        return ak, ck

    def update_with_momentum(self, params, grad, momentum, beta=0.9, ak=0.01):
        momentum = beta * momentum + (1 - beta) * grad
        params = params - ak * momentum
        return params, momentum

    def decode_path(self, weights_matrix):
        n = self.n
        path = [0]
        visited = {0}
        for _ in range(n - 1):
            curr = path[-1]
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
            trial_dist = sum(self.B[trial_path[i]][trial_path[i+1]] for i in range(self.n))
            if trial_dist < min_global_dist:
                min_global_dist = trial_dist
                best_global_path = trial_path
            prefix = f"Refined SPSA Progress (Trial {trial + 1}/{trials}):"
            print_progress(trial + 1, trials, start_time_global, prefix=prefix)
        print()
        return best_global_path, min_global_dist

    def solve_brute_force(self):
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
            if k % 10000 == 0 or k == total_permutations - 1:
                print_progress(k + 1, total_permutations, start_time)
        print()
        return best_path, min_cost


# ── Test Runner ───────────────────────────────────────────────────────────────

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
        print(f"Skipping State Tomography for {solver.n} cities.")

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
        print(f"Skipping Brute Force for {solver.n} cities.")
