import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools
import time
from math import factorial
import datetime
import mlx.core as mx
from itertools import combinations
import os
from sklearn.cluster import KMeans

from tsp_utils import print_progress, generate_random_tsp, plot_decomposition_result
from tsp_spsa import SingleQubitTSP, two_opt_refinement
from tsp_decomposition import solve_tsp_decomposition
from tsp_held_karp import solve_tsp_held_karp_mlx, solve_tsp_held_karp_cpu

dist_4_city = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

dist_5_asymmetric = [
    [0, 10, 8, 9, 7],
    [10, 0, 10, 5, 6],
    [8, 10, 0, 8, 9],
    [9, 5, 8, 0, 6],
    [7, 6, 9, 6, 0]
]

dist_8_city = [
    [0, 15, 10, 20, 12, 18, 14, 22], [15, 0, 16, 25, 13, 11, 21, 19],
    [10, 16, 0, 30, 17, 24, 15, 28], [20, 25, 30, 0, 22, 14, 16, 12],
    [12, 13, 17, 22, 0, 26, 18, 14], [18, 11, 24, 14, 26, 0, 12, 10],
    [14, 21, 15, 16, 18, 12, 0, 17], [22, 19, 28, 12, 14, 10, 17, 0]
]

test_cases = [
    ("4-City Symmetric", dist_4_city, None),
    ("5-City Asymmetric", dist_5_asymmetric, None),
    ("8-City Symmetric", dist_8_city, None),
    ("6-City Random", *generate_random_tsp(6)),
    ("7-City Random", *generate_random_tsp(7)),
    ("10-City Random", *generate_random_tsp(10)),
    ("11-City Random", *generate_random_tsp(11)),
    ("12-City Random", *generate_random_tsp(12)),
    ("13-City Random", *generate_random_tsp(13)),
    ("17-City Random", *generate_random_tsp(17)),
    ("20-City Random", *generate_random_tsp(20)),
    ("30-City Random", *generate_random_tsp(30)),
    ("40-City Random", *generate_random_tsp(40)),
    ("50-City Random", *generate_random_tsp(50)),
    ("75-City Random", *generate_random_tsp(75)),
    ("100-City Random", *generate_random_tsp(100))
]

results_summary = []
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for name, matrix, coords in test_cases:
    print(f"\n--- {name} ---")
    solver = SingleQubitTSP(matrix)
    
    best_path = None
    best_cost = None
    hybrid_time = None

    if solver.n <= 50:
        print("Solving with Hybrid SPSA...")
        start_time = time.time()
        best_path, best_cost = solver.solve_hybrid(iters_1st=1000, iters_2nd=500)
        hybrid_time = time.time() - start_time
        print(f"Hybrid Path: {' -> '.join(map(str, best_path))}")
        print(f"Hybrid Cost: {best_cost:.2f}")
    else:
        print(f"Skipping Hybrid SPSA for {solver.n} cities (too computationally expensive)")

    print("Solving with Refined SPSA...")
    start_time = time.time()
    refined_path, refined_cost = solver.solve_refined(trials=3, iterations=1000)
    refined_time = time.time() - start_time
    print(f"Refined Path: {' -> '.join(map(str, refined_path))}")
    print(f"Refined Cost: {refined_cost}") 

    print("Applying 2-opt Refinement...")
    start_time = time.time()
    optimized_path, final_cost = two_opt_refinement(refined_path, solver.B)
    two_opt_time = time.time() - start_time
    print(f"Optimized Path after 2-opt: {' -> '.join(map(str, optimized_path))}")
    print(f"Optimized Cost after 2-opt: {final_cost}")  

    decomp_cost = None
    decomp_time = None
    decomp_path = None
    if coords is not None and solver.n >= 10:
        print("Solving with Decomposition (K-Means + Refined SPSA)...")
        start_time = time.time()
        # Determine K based on N (e.g., roughly 5-10 cities per cluster)
        k_clusters = max(2, int(solver.n / 8))
        decomp_path, decomp_cost, labels, centroids = solve_tsp_decomposition(solver, coords, k=k_clusters)
        decomp_time = time.time() - start_time
        print(f"Decomposition Path: {' -> '.join(map(str, decomp_path))}")
        print(f"Decomposition Cost: {decomp_cost:.2f}")
    else:
        print(f"Skipping Decomposition for {name} (No coords or N too small).")

    mc_cost = None
    mc_time = None
    mc_path = None
    # Limit Monte Carlo to n < 50 due to computational constraints
    if solver.n < 50:
        print("Calculating approximate solution (Monte Carlo MLX)...")
        start_time = time.time()
        mc_path, mc_cost = solver.montecarlo_explore_mlx()
        mc_time = time.time() - start_time
        print(f"Monte Carlo Path: {' -> '.join(map(str, mc_path))}")
        print(f"Monte Carlo Cost: {mc_cost:.2f}")
    else:
        print(f"Skipping Monte Carlo for {solver.n} cities.")

    hk_cost = None
    hk_time = None
    hk_path = None
    # Limit Held-Karp to n < 30 due to computational constraints
    # RuntimeError: [metal::malloc] Attempting to allocate 128849018880 bytes which is greater than the maximum allowed buffer size of 17179869184 bytes.
    # This occurs around n=30 due to 2^n memory requirements.
    if solver.n < 30:
        print("Calculating exact solution (Held-Karp MLX)...")
        start_time = time.time()
        hk_cost, hk_path = solve_tsp_held_karp_mlx(solver.B)
        hk_time = time.time() - start_time
        print(f"Held-Karp Path: {' -> '.join(map(str, hk_path))}")
        print(f"Held-Karp Cost: {hk_cost:.2f}")
    else:
        print(f"Skipping Held-Karp for {solver.n} cities.")

    hk_cpu_cost = None
    hk_cpu_time = None
    hk_cpu_path = None
    if solver.n <= 16:
        print("Calculating exact solution (Held-Karp CPU)...")
        start_time = time.time()
        hk_cpu_cost, hk_cpu_path = solve_tsp_held_karp_cpu(solver.B)
        hk_cpu_time = time.time() - start_time
        print(f"Held-Karp CPU Path: {' -> '.join(map(str, hk_cpu_path))}")
        print(f"Held-Karp CPU Cost: {hk_cpu_cost:.2f}")
    else:
        print(f"Skipping Held-Karp CPU for {solver.n} cities.")

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
    
    results_summary.append({
        "Case": name,
        "Matrix": matrix,
        "Hybrid": (best_cost, hybrid_time, best_path),
        "Refined": (refined_cost, refined_time, refined_path),
        "Refined+2Opt": (final_cost, refined_time + two_opt_time, optimized_path),
        "Decomposition": (decomp_cost, decomp_time, decomp_path),
        "MonteCarlo": (mc_cost, mc_time, mc_path),
        "HeldKarp": (hk_cost, hk_time, hk_path),
        "HeldKarpCPU": (hk_cpu_cost, hk_cpu_time, hk_cpu_path),
        "BruteForce": (bf_cost, bf_time, bf_path)
    })

print("\n" + "="*215)
print(f"{'TEST CASE':<20} | {'HYBRID (Cost/Time)':<20} | {'REFINED (Cost/Time)':<20} | {'REF+2OPT (Cost/Time)':<20} | {'DECOMP (Cost/Time)':<20} | {'MONTE CARLO':<20} | {'HELD KARP':<20} | {'HK CPU':<20} | {'BRUTE FORCE':<15}")
print("-" * 215)
for res in results_summary:
    h_c, h_t, _ = res["Hybrid"]
    r_c, r_t, _ = res["Refined"]
    r2_c, r2_t, _ = res["Refined+2Opt"]
    dc_c, dc_t, _ = res["Decomposition"]
    mc_c, mc_t, _ = res["MonteCarlo"]
    hk_c, hk_t, _ = res["HeldKarp"]
    hk_cpu_c, hk_cpu_t, _ = res["HeldKarpCPU"]
    bf_c, bf_t, _ = res["BruteForce"]
    
    if h_c is not None:
        h_str = f"{h_c:.2f} / {h_t:.2f}s"
    else:
        h_str = "N/A"
    r_str = f"{r_c:.2f} / {r_t:.2f}s"
    r2_str = f"{r2_c:.2f} / {r2_t:.2f}s"
    
    if dc_c is not None:
        dc_str = f"{dc_c:.2f} / {dc_t:.2f}s"
    else:
        dc_str = "N/A"

    if mc_c is not None:
        mc_str = f"{mc_c:.2f} / {mc_t:.2f}s"
    else:
        mc_str = "N/A"

    if hk_c is not None:
        hk_str = f"{hk_c:.2f} / {hk_t:.2f}s"
    else:
        hk_str = "N/A"

    if hk_cpu_c is not None:
        hk_cpu_str = f"{hk_cpu_c:.2f} / {hk_cpu_t:.2f}s"
    else:
        hk_cpu_str = "N/A"

    if bf_c is not None:
        bf_str = f"{bf_c:.2f} / {bf_t:.4f}s"
    else:
        bf_str = "N/A"
    
    print(f"{res['Case']:<20} | {h_str:<20} | {r_str:<20} | {r2_str:<20} | {dc_str:<20} | {mc_str:<20} | {hk_str:<20} | {hk_cpu_str:<20} | {bf_str:<15}")
print("="*215 + "\n")

# --- Export Results to File ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(output_dir, f"tsp_results_{timestamp}.txt")

with open(filename, "w") as f:
    f.write(f"TSP Optimization Results - {timestamp}\n")
    f.write("="*215 + "\n")
    f.write(f"{'TEST CASE':<20} | {'HYBRID (Cost/Time)':<20} | {'REFINED (Cost/Time)':<20} | {'REF+2OPT (Cost/Time)':<20} | {'DECOMP (Cost/Time)':<20} | {'MONTE CARLO':<20} | {'HELD KARP':<20} | {'HK CPU':<20} | {'BRUTE FORCE':<15}\n")
    f.write("-" * 215 + "\n")
    
    for res in results_summary:
        h_c, h_t, _ = res["Hybrid"]
        r_c, r_t, _ = res["Refined"]
        r2_c, r2_t, _ = res["Refined+2Opt"]
        dc_c, dc_t, _ = res["Decomposition"]
        mc_c, mc_t, _ = res["MonteCarlo"]
        hk_c, hk_t, _ = res["HeldKarp"]
        hk_cpu_c, hk_cpu_t, _ = res["HeldKarpCPU"]
        bf_c, bf_t, _ = res["BruteForce"]
        
        if h_c is not None:
            h_str = f"{h_c:.2f} / {h_t:.2f}s"
        else:
            h_str = "N/A"
        r_str = f"{r_c:.2f} / {r_t:.2f}s"
        r2_str = f"{r2_c:.2f} / {r2_t:.2f}s"
        
        if dc_c is not None:
            dc_str = f"{dc_c:.2f} / {dc_t:.2f}s"
        else:
            dc_str = "N/A"

        if mc_c is not None:
            mc_str = f"{mc_c:.2f} / {mc_t:.2f}s"
        else:
            mc_str = "N/A"

        if hk_c is not None:
            hk_str = f"{hk_c:.2f} / {hk_t:.2f}s"
        else:
            hk_str = "N/A"

        if hk_cpu_c is not None:
            hk_cpu_str = f"{hk_cpu_c:.2f} / {hk_cpu_t:.2f}s"
        else:
            hk_cpu_str = "N/A"

        if bf_c is not None:
            bf_str = f"{bf_c:.2f} / {bf_t:.4f}s"
        else:
            bf_str = "N/A"
        
        f.write(f"{res['Case']:<20} | {h_str:<20} | {r_str:<20} | {r2_str:<20} | {dc_str:<20} | {mc_str:<20} | {hk_str:<20} | {hk_cpu_str:<20} | {bf_str:<15}\n")
    
    f.write("="*215 + "\n\n")
    f.write("DETAILED RESULTS\n")
    f.write("="*215 + "\n")

    for res in results_summary:
        f.write(f"\n--- {res['Case']} ---\n")
        f.write("Distance Matrix:\n")
        f.write(np.array2string(np.array(res['Matrix']), precision=2, separator=', ', threshold=sys.maxsize) + "\n\n")
        
        if res['Hybrid'][0] is not None:
            f.write(f"Hybrid SPSA:     Cost={res['Hybrid'][0]:.2f}, Time={res['Hybrid'][1]:.4f}s, Path={res['Hybrid'][2]}\n")
        else:
            f.write("Hybrid SPSA:     Skipped\n")
        f.write(f"Refined SPSA:    Cost={res['Refined'][0]:.2f}, Time={res['Refined'][1]:.4f}s, Path={res['Refined'][2]}\n")
        f.write(f"Refined + 2-Opt: Cost={res['Refined+2Opt'][0]:.2f}, Time={res['Refined+2Opt'][1]:.4f}s, Path={res['Refined+2Opt'][2]}\n")
        if res['Decomposition'][0] is not None:
            f.write(f"Decomposition:   Cost={res['Decomposition'][0]:.2f}, Time={res['Decomposition'][1]:.4f}s, Path={res['Decomposition'][2]}\n")
        else:
            f.write("Decomposition:   Skipped\n")
        if res['MonteCarlo'][0] is not None:
            f.write(f"Monte Carlo:     Cost={res['MonteCarlo'][0]:.2f}, Time={res['MonteCarlo'][1]:.4f}s, Path={res['MonteCarlo'][2]}\n")
        else:
            f.write("Monte Carlo:     Skipped\n")
        if res['HeldKarp'][0] is not None:
            f.write(f"Held-Karp:       Cost={res['HeldKarp'][0]:.2f}, Time={res['HeldKarp'][1]:.4f}s, Path={res['HeldKarp'][2]}\n")
        else:
            f.write("Held-Karp:       Skipped\n")
        if res['HeldKarpCPU'][0] is not None:
            f.write(f"Held-Karp CPU:   Cost={res['HeldKarpCPU'][0]:.2f}, Time={res['HeldKarpCPU'][1]:.4f}s, Path={res['HeldKarpCPU'][2]}\n")
        else:
            f.write("Held-Karp CPU:   Skipped\n")
        if res['BruteForce'][0] is not None:
            f.write(f"Brute Force:     Cost={res['BruteForce'][0]:.2f}, Time={res['BruteForce'][1]:.4f}s, Path={res['BruteForce'][2]}\n")
        else:
            f.write("Brute Force:     Skipped\n")
        f.write("-" * 50 + "\n")

print(f"Results exported to {filename}")

# --- Generate Plots ---
print("Generating plots...")
n_vals = []
hybrid_costs, hybrid_times = [], []
refined_costs, refined_times = [], []
r2opt_costs, r2opt_times = [], []
decomp_costs, decomp_times = [], []
mc_costs, mc_times = [], []
hk_costs, hk_times = [], []
hk_cpu_costs, hk_cpu_times = [], []
bf_costs, bf_times = [], []

for res in results_summary:
    n = len(res['Matrix'])
    n_vals.append(n)
    
    # Hybrid
    hc, ht, _ = res['Hybrid']
    hybrid_costs.append(hc if hc is not None else np.nan)
    hybrid_times.append(ht if ht is not None else np.nan)
    
    # Refined
    rc, rt, _ = res['Refined']
    refined_costs.append(rc)
    refined_times.append(rt)
    
    # Refined + 2Opt
    r2c, r2t, _ = res['Refined+2Opt']
    r2opt_costs.append(r2c)
    r2opt_times.append(r2t)
    
    # Decomposition
    dc, dt, _ = res['Decomposition']
    decomp_costs.append(dc if dc is not None else np.nan)
    decomp_times.append(dt if dt is not None else np.nan)

    # Monte Carlo
    mcc, mct, _ = res['MonteCarlo']
    mc_costs.append(mcc if mcc is not None else np.nan)
    mc_times.append(mct if mct is not None else np.nan)

    # Held-Karp
    hkc, hkt, _ = res['HeldKarp']
    hk_costs.append(hkc if hkc is not None else np.nan)
    hk_times.append(hkt if hkt is not None else np.nan)

    # Held-Karp CPU
    hkcc, hkct, _ = res['HeldKarpCPU']
    hk_cpu_costs.append(hkcc if hkcc is not None else np.nan)
    hk_cpu_times.append(hkct if hkct is not None else np.nan)

    # Brute Force
    bfc, bft, _ = res['BruteForce']
    bf_costs.append(bfc if bfc is not None else np.nan)
    bf_times.append(bft if bft is not None else np.nan)

# Sort data by N
perm = np.argsort(n_vals)
n_vals = np.array(n_vals)[perm]
hybrid_costs = np.array(hybrid_costs)[perm]
hybrid_times = np.array(hybrid_times)[perm]
refined_costs = np.array(refined_costs)[perm]
refined_times = np.array(refined_times)[perm]
r2opt_costs = np.array(r2opt_costs)[perm]
r2opt_times = np.array(r2opt_times)[perm]
decomp_costs = np.array(decomp_costs)[perm]
decomp_times = np.array(decomp_times)[perm]
mc_costs = np.array(mc_costs)[perm]
mc_times = np.array(mc_times)[perm]
hk_costs = np.array(hk_costs)[perm]
hk_times = np.array(hk_times)[perm]
hk_cpu_costs = np.array(hk_cpu_costs)[perm]
hk_cpu_times = np.array(hk_cpu_times)[perm]
bf_costs = np.array(bf_costs)[perm]
bf_times = np.array(bf_times)[perm]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Cost Plot
ax1.plot(n_vals, hybrid_costs, 'o-', label='Hybrid SPSA')
ax1.plot(n_vals, refined_costs, 's-', label='Refined SPSA')
ax1.plot(n_vals, r2opt_costs, '^-', label='Refined + 2-Opt')
ax1.plot(n_vals, decomp_costs, 'p-', label='Decomposition', color='green')
ax1.plot(n_vals, mc_costs, 'd--', label='Monte Carlo', color='purple', alpha=0.7)
ax1.plot(n_vals, hk_costs, '*--', label='Held-Karp', color='orange', alpha=0.8)
ax1.plot(n_vals, hk_cpu_costs, '+--', label='Held-Karp CPU', color='brown', alpha=0.8)
ax1.plot(n_vals, bf_costs, 'x--', label='Brute Force', color='k', alpha=0.6)
ax1.set_title('Cost vs Number of Cities')
ax1.set_xlabel('Number of Cities')
ax1.set_ylabel('Cost')
ax1.legend()
ax1.grid(True)

# Time Plot
ax2.plot(n_vals, hybrid_times, 'o-', label='Hybrid SPSA')
ax2.plot(n_vals, refined_times, 's-', label='Refined SPSA')
ax2.plot(n_vals, r2opt_times, '^-', label='Refined + 2-Opt')
ax2.plot(n_vals, decomp_times, 'p-', label='Decomposition', color='green')
ax2.plot(n_vals, mc_times, 'd--', label='Monte Carlo', color='purple', alpha=0.7)
ax2.plot(n_vals, hk_times, '*--', label='Held-Karp', color='orange', alpha=0.8)
ax2.plot(n_vals, hk_cpu_times, '+--', label='Held-Karp CPU', color='brown', alpha=0.8)
ax2.plot(n_vals, bf_times, 'x--', label='Brute Force', color='k', alpha=0.6)
ax2.set_title('Time vs Number of Cities')
ax2.set_xlabel('Number of Cities')
ax2.set_ylabel('Time (s)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True)

plot_filename = os.path.join(output_dir, f"tsp_results_{timestamp}.png")
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")

# --- Generate Decomposition Plot for the last applicable case ---
if decomp_path is not None and coords is not None:
    print("Generating Decomposition visualization...")
    decomp_plot_filename = os.path.join(output_dir, f"tsp_decomposition_{timestamp}.png")
    plot_decomposition_result(coords, labels, centroids, decomp_path, decomp_plot_filename)
    print(f"Decomposition plot saved to {decomp_plot_filename}")