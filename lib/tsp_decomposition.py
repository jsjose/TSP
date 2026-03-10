import numpy as np
from sklearn.cluster import KMeans
from lib.tsp_spsa import SingleQubitTSP

def solve_tsp_decomposition(solver, coords, k=5):
    """
    Solves TSP using a Divide and Conquer approach with K-Means clustering.
    Sub-problems are solved using the Refined SPSA solver.
    """
    if coords is None or len(coords) == 0:
        return None, None, None, None

    n = len(coords)
    if n < k:
        k = max(1, n // 2)
        
    # 1. Decomposition Phase
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(coords)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 2. Conquer Phase: Solve high-level TSP for centroids
    # Calculate centroid distance matrix
    centroid_dist_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            centroid_dist_matrix[i][j] = np.linalg.norm(centroids[i] - centroids[j])
            
    # Use Brute Force for centroids (K is small)
    c_solver = SingleQubitTSP(centroid_dist_matrix)
    c_path, _ = c_solver.solve_brute_force()
    
    # Remove return-to-start for iteration
    if c_path[0] == c_path[-1]:
        c_path = c_path[:-1]

    # 3. Stitching Phase
    full_path = []
    
    for cluster_idx in c_path:
        indices = np.where(labels == cluster_idx)[0]
        if len(indices) == 0: continue
        
        if len(indices) == 1:
            full_path.append(indices[0])
            continue
            
        # Create sub-problem
        sub_matrix = solver.B[np.ix_(indices, indices)]
        sub_solver = SingleQubitTSP(sub_matrix)
        
        # Use Refined SPSA for sub-problem (faster than Hybrid for sub-loops)
        # We use fewer trials/iterations for speed
        sp_path, _ = sub_solver.solve_refined(trials=1, iterations=300)
        
        # Map local indices back to global indices
        # sp_path includes return to start, remove it
        for local_idx in sp_path[:-1]:
            full_path.append(indices[local_idx])
            
    full_path.append(full_path[0]) # Close the loop
    
    # Calculate total cost
    cost = sum(solver.B[full_path[i]][full_path[i+1]] for i in range(len(full_path)-1))
    return full_path, cost, labels, centroids