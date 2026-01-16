import mlx.core as mx

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