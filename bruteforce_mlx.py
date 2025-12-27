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
    # This is a simplified version of generating permutations via vector math.
    # For real brute force, you would map batch_indices to permutations here.
    # For demonstration, we'll create a random batch of permutations:
    
    # Note: Generating actual permutations vectorially is complex.
    # Typically, we'd use a pre-calculated block of permutations.
    pass

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
test_perms = mx.array(np.array([np.random.permutation(n) for _ in range(batch_size)]))

# 2. Run the vectorized distance calculation on GPU
distances = get_batch_distances(test_perms, dist_matrix)

# 3. Find the best in this batch
best_idx = mx.argmin(distances)
print(f"Best distance in batch: {distances[best_idx].item()}")