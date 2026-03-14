"""
GPU Brute-Force TSP solver using NVIDIA CUDA (Numba).

Uses factoradic permutation mapping to assign each CUDA thread a unique
permutation. Practical for N ≤ 12-14 due to factorial explosion.

Requires NVIDIA GPU and `pip install numba`.

Run from anywhere:
    python experiments/bruteforce_cuda.py
"""
import numpy as np
from numba import cuda, int64
import math


# 1. Factoradic mapping: Converts a thread index to a unique city permutation
@cuda.jit(device=True)
def get_permutation(n, index, perm, cities):
    # Initialize available cities
    for i in range(n):
        cities[i] = i

    # Factoradic selection logic
    temp_index = index
    for i in range(n):
        fact = 1
        for j in range(1, n - i):
            fact *= j

        pos = temp_index // fact
        perm[i] = cities[pos]

        # Remove used city by shifting
        for j in range(pos, n - 1):
            cities[j] = cities[j + 1]

        temp_index %= fact


@cuda.jit
def gpu_brute_force_tsp(dist_matrix, n, start_permutation, total_perms, global_min):
    idx = cuda.grid(1)
    current_idx = start_permutation + idx

    if current_idx < total_perms:
        # Local thread memory (max 20 cities)
        perm = cuda.local.array(20, int64)
        cities = cuda.local.array(20, int64)

        get_permutation(n, current_idx, perm, cities)

        # Calculate total distance for this permutation
        dist = 0.0
        for i in range(n - 1):
            dist += dist_matrix[perm[i], perm[i + 1]]
        dist += dist_matrix[perm[n - 1], perm[0]]  # Return to start

        # Atomic update to find the global minimum across all threads
        cuda.atomic.min(global_min, 0, dist)


# ── Execution ─────────────────────────────────────────────────────────────────

n_cities = 12
dist_matrix = np.random.rand(n_cities, n_cities).astype(np.float32)
total_perms = math.factorial(n_cities)

d_dist_matrix = cuda.to_device(dist_matrix)
d_global_min = cuda.to_device(np.array([float('inf')], dtype=np.float32))

threads_per_block = 256
blocks_per_grid = (min(total_perms, 10**7) + (threads_per_block - 1)) // threads_per_block

gpu_brute_force_tsp[blocks_per_grid, threads_per_block](d_dist_matrix, n_cities, 0, total_perms, d_global_min)

print(f"Shortest path found: {d_global_min.copy_to_host()[0]}")
