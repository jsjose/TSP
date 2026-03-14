import numpy as np
from sklearn.cluster import KMeans

from utils.helpers import calculate_total_distance
from .spsa import SingleQubitTSP


def solve_tsp_decomposition(
    solver: SingleQubitTSP,
    coords: np.ndarray,
    k: int = 5,
) -> tuple[list[int], float, np.ndarray, np.ndarray]:
    """Solves TSP using a Divide-and-Conquer approach with K-Means clustering.

    Phase 1: K-Means partitions cities into k clusters.
    Phase 2: Brute-force solves the k-centroid TSP to determine cluster visit order.
    Phase 3: Refined SPSA solves each cluster independently.
    Phase 4: Sub-tours are stitched into a global tour.

    Requires coordinate data (not available for EXPLICIT edge-weight instances).

    Args:
        solver: SingleQubitTSP instance (provides the distance matrix via solver.B).
        coords: Nx2 array of city coordinates.
        k: Number of clusters (default 5).

    Returns:
        (path, cost, labels, centroids): Global tour, its total cost,
        cluster assignments, and centroid positions.
        Returns (None, None, None, None) if coords is unavailable.
    """
    if coords is None or len(coords) == 0:
        return None, None, None, None

    n = len(coords)
    if n < k:
        k = max(1, n // 2)

    # --- Phase 1: Decompose into clusters ---
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(coords)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # --- Phase 2: Solve high-level TSP for centroids ---
    centroid_dist_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            centroid_dist_matrix[i][j] = np.linalg.norm(centroids[i] - centroids[j])

    c_solver = SingleQubitTSP(centroid_dist_matrix)
    c_path, _ = c_solver.solve_brute_force()

    # Remove return-to-start for iteration
    if c_path[0] == c_path[-1]:
        c_path = c_path[:-1]

    # --- Phase 3 & 4: Solve each cluster and stitch ---
    full_path: list[int] = []

    for cluster_idx in c_path:
        indices = np.where(labels == cluster_idx)[0]
        if len(indices) == 0:
            continue

        if len(indices) == 1:
            full_path.append(int(indices[0]))
            continue

        # Create sub-problem from the global distance matrix
        sub_matrix = solver.B[np.ix_(indices, indices)]
        sub_solver = SingleQubitTSP(sub_matrix)

        # Refined SPSA with reduced budget for speed
        sp_path, _ = sub_solver.solve_refined(trials=1, iterations=300)

        # Map local indices back to global indices (exclude closing return to start)
        for local_idx in sp_path[:-1]:
            full_path.append(int(indices[local_idx]))

    full_path.append(full_path[0])  # Close the global tour

    cost = calculate_total_distance(full_path, solver.B)
    return full_path, cost, labels, centroids
