import time
import sys
import numpy as np
import matplotlib.pyplot as plt

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
    """Computes the total length of the TSP tour."""
    distance = 0
    for i in range(len(path) - 1):
        distance += dist_matrix[path[i]][path[i+1]]
    return distance

def generate_random_tsp(n_cities, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(n_cities, 2) * 100
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix, coords

def plot_decomposition_result(coords, labels, centroids, path, filename):
    """Visualizes the clusters and the global stitched path."""
    plt.figure(figsize=(10, 7))
    k = len(centroids)
    colors = plt.cm.rainbow(np.linspace(0, 1, k))
    
    for i in range(k):
        cluster_pts = coords[labels == i]
        plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], color=colors[i], label=f'Cluster {i}', s=50)
        plt.scatter(centroids[i, 0], centroids[i, 1], color=colors[i], marker='x', s=100, linewidths=3)

    # Draw the global path
    path_coords = coords[path]
    plt.plot(path_coords[:, 0], path_coords[:, 1], 'k--', alpha=0.6, linewidth=1.5, label='Global Stitched Path')
    plt.title(f"TSP Decomposition: {len(coords)} Cities, {k} Clusters")
    plt.legend()
    plt.savefig(filename)
    plt.close()