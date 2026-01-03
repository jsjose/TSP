import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Generate 50 Random Cities
np.random.seed(42)
cities = np.random.rand(50, 2) * 100

# 2. Decomposition Phase (Cluster into 5 subproblems)
k = 5
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(cities)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 3. Conquer Phase: Solve each subproblem
def solve_local_tsp(cluster_cities):
    # For simplicity, we use a Nearest Neighbor heuristic here.
    # In your project, you would insert your 'solve_hybrid' qubit solver here!
    size = len(cluster_cities)
    if size <= 1: return list(range(size))
    
    path = [0]
    unvisited = list(range(1, size))
    while unvisited:
        curr = cluster_cities[path[-1]]
        next_idx = min(unvisited, key=lambda i: np.linalg.norm(curr - cluster_cities[i]))
        path.append(next_idx)
        unvisited.remove(next_idx)
    return path

# 4. Stitching Phase: Connect mini-tours based on Centroid proximity
# Determine the order of clusters by solving a high-level TSP on centroids
centroid_order = solve_local_tsp(centroids)

full_global_path = []
for cluster_idx in centroid_order:
    cluster_indices = np.where(labels == cluster_idx)[0]
    if len(cluster_indices) == 0: continue
    
    # Solve local tour for these indices
    local_cities = cities[cluster_indices]
    local_order = solve_local_tsp(local_cities)
    
    # Map local relative indices back to global city indices
    global_local_path = [cluster_indices[i] for i in local_order]
    full_global_path.extend(global_local_path)

full_global_path.append(full_global_path[0]) # Close the loop

# 5. Visualization
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(k):
    cluster_pts = cities[labels == i]
    plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], c=colors[i], label=f'Cluster {i}')

# Draw the global path
path_coords = cities[full_global_path]
plt.plot(path_coords[:, 0], path_coords[:, 1], 'k--', alpha=0.5, label='Global Stitched Path')
plt.title("50-City TSP: Subproblem Decomposition via K-Means")
plt.legend()
plt.show()