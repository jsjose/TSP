import tsp_loader
from tsp_decomposition import solve_tsp_decomposition
from tsp_spsa import SingleQubitTSP, two_opt_refinement
import time

# 1. Load the master solutions file once
# Assume 'solutions.txt' contains:
# a280 : 2579
# att48 : 10628
solutions_map = tsp_loader.load_solutions_file("tsplib/solutions")

# 2. Load a specific problem instance
# The loader will look up 'a280' inside solutions_map automatically
instance = tsp_loader.load_tsp("tsplib/dantzig42.tsp", solutions_map)
instance = tsp_loader.load_tsp("tsplib/ali535.tsp", solutions_map)
instance = tsp_loader.load_tsp("tsplib/att48.tsp", solutions_map)
instance = tsp_loader.load_tsp("tsplib/bayg29.tsp", solutions_map)
instance = tsp_loader.load_tsp("tsplib/brazil58.tsp", solutions_map)
instance = tsp_loader.load_tsp("tsplib/brg180.tsp", solutions_map)
instance = tsp_loader.load_tsp("tsplib/burma14.tsp", solutions_map)
instance = tsp_loader.load_tsp("tsplib/dsj1000.tsp", solutions_map)

print(f"Loaded: {instance.name}")
print(f"Dimension: {instance.dimension}")
print(f"Target Optimal Cost: {instance.optimal_cost}")

# 3. Use in your algorithm
# ... run your solver ...
# my_cost = run_solver(instance.distance_matrix)
# gap = (my_cost - instance.optimal_cost) / instance.optimal_cost * 100
# print(f"Optimality Gap: {gap:.2f}%")

solver = SingleQubitTSP(instance.distance_matrix)

print("Solving with Refined SPSA...")
start_time = time.time()
refined_path, refined_cost = solver.solve_refined(trials=3, iterations=1000)
refined_time = time.time() - start_time
print(f"Refined Path: {' -> '.join(map(str, refined_path))}")
print(f"Refined Cost: {refined_cost}") 

if instance.optimal_cost is not None:
    gap = (refined_cost - instance.optimal_cost) / instance.optimal_cost * 100
    print(f"Optimality Gap: {gap:.2f}%")
else:
    print("Optimality Gap: N/A")

print("Applying 2-opt Refinement...")
start_time = time.time()
optimized_path, final_cost = two_opt_refinement(refined_path, solver.B)
two_opt_time = time.time() - start_time
print(f"Optimized Path after 2-opt: {' -> '.join(map(str, optimized_path))}")
print(f"Optimized Cost after 2-opt: {final_cost}")  

if instance.optimal_cost is not None:
    gap = (final_cost - instance.optimal_cost) / instance.optimal_cost * 100
    print(f"Optimality Gap: {gap:.2f}%")
else:
    print("Optimality Gap: N/A")

decomp_cost = None
decomp_time = None
decomp_path = None
if instance.coords is not None and len(instance.coords) > 0 and solver.n >= 10:
    print("Solving with Decomposition (K-Means + Refined SPSA)...")
    start_time = time.time()
    # Determine K based on N (e.g., roughly 5-10 cities per cluster)
    k_clusters = max(2, int(solver.n / 8))
    decomp_path, decomp_cost, labels, centroids = solve_tsp_decomposition(solver, instance.coords, k=k_clusters)
    decomp_time = time.time() - start_time
    print(f"Decomposition Path: {' -> '.join(map(str, decomp_path))}")
    print(f"Decomposition Cost: {decomp_cost:.2f}")
else:
    print(f"Skipping Decomposition for {instance.name} (No coords or N too small).")

if decomp_cost is not None and instance.optimal_cost:
    gap = (decomp_cost - instance.optimal_cost) / instance.optimal_cost * 100
    print(f"Optimality Gap: {gap:.2f}%")

bf_cost = None
bf_time = None
if solver.n < 13:
    print("Calculating exact solution (Brute Force)...")
    start_time = time.time()
    bf_path, bf_cost = solver.solve_brute_force()
    bf_time = time.time() - start_time
    print(f"Exact Path: {' -> '.join(map(str, bf_path))}")
    print(f"Exact Cost: {bf_cost}")
else:
    print(f"Skipping Brute Force for {solver.n} cities (too computationally expensive).")