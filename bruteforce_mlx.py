import mlx.core as mx
import numpy as np
import math

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

n = 10 
dist_matrix = mx.random.uniform(shape=(n, n))

n = 5
dist_matrix = mx.array([
    [0, 10, 8, 9, 7],
    [10, 0, 10, 5, 6],
    [8, 10, 0, 8, 9],
    [9, 5, 8, 0, 6],
    [7, 6, 9, 6, 0]
])
# --- High Level Logic for M2 ---

batch_size = 1_000_000

# 1. Generate a batch of random permutations to test performance
# (In a real brute force, you'd iterate through all permutations)
batch_indices = mx.arange(batch_size)

best_dist, best_perm = solve_tsp_batch(batch_indices, n, dist_matrix)
print(f"Best distance in batch: {best_dist.item()}")
print(f"Best path: {best_perm}")