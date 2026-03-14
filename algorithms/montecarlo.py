import mlx.core as mx


def get_batch_distances(perms: mx.array, dist_matrix: mx.array) -> mx.array:
    """Vectorized distance calculation for a batch of permutations.

    Args:
        perms: Shape (batch_size, n_cities) — integer permutation indices.
        dist_matrix: Shape (n_cities, n_cities) — distance matrix.

    Returns:
        mx.array of shape (batch_size,) with total tour distance per permutation.
    """
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
def solve_tsp_batch(
    batch_indices: mx.array, n_cities: int, dist_matrix: mx.array
) -> tuple[mx.array, mx.array]:
    """Generates a batch of random permutations and finds the best one.

    Args:
        batch_indices: Arange of batch size, used to trigger compilation.
        n_cities: Number of cities in the TSP instance.
        dist_matrix: Shape (n_cities, n_cities) — distance matrix as MLX array.

    Returns:
        (best_dist, best_perm): Best tour distance and corresponding permutation.
    """
    batch_size = batch_indices.shape[0]

    # Generate random permutations using argsort on random noise
    keys = mx.random.uniform(shape=(batch_size, n_cities))
    perms = mx.argsort(keys, axis=1)

    dists = get_batch_distances(perms, dist_matrix)

    best_idx = mx.argmin(dists)
    return dists[best_idx], perms[best_idx]
