"""
Advanced benchmark script comparing all TSP solvers across problem sizes.

Runs Hybrid SPSA, Refined SPSA, 2-Opt, Decomposition, Monte Carlo,
Held-Karp (MLX + CPU), and Brute Force, then exports results to CSV and plots.

Run from the project root:
    python experiments/benchmark_advanced.py
"""
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

from utils.helpers import print_progress, generate_random_tsp, plot_decomposition_result
from algorithms.spsa import SingleQubitTSP, two_opt_refinement
from algorithms.decomposition import solve_tsp_decomposition
from algorithms.held_karp import solve_tsp_held_karp_mlx, solve_tsp_held_karp_cpu


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
    if solver.n < 30:
        print("Calculating exact solution (Held-Karp MLX)...")
        start_time = time.time()
        hk_path, hk_cost = solve_tsp_held_karp_mlx(solver.B)  # returns (path, cost)
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
        hk_cpu_path, hk_cpu_cost = solve_tsp_held_karp_cpu(solver.B)  # returns (path, cost)
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
        print(f"Skipping Brute Force for {solver.n} cities.")

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
        "BruteForce": (bf_cost, bf_time, None)
    })


# ── Results Table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 215)
print(f"{'TEST CASE':<20} | {'HYBRID (Cost/Time)':<20} | {'REFINED (Cost/Time)':<20} | "
      f"{'REF+2OPT (Cost/Time)':<20} | {'DECOMP (Cost/Time)':<20} | "
      f"{'MONTE CARLO':<20} | {'HELD KARP':<20} | {'HK CPU':<20} | {'BRUTE FORCE':<15}")
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

    h_str = f"{h_c:.2f} / {h_t:.2f}s" if h_c is not None else "N/A"
    r_str = f"{r_c:.2f} / {r_t:.2f}s"
    r2_str = f"{r2_c:.2f} / {r2_t:.2f}s"
    dc_str = f"{dc_c:.2f} / {dc_t:.2f}s" if dc_c is not None else "N/A"
    mc_str = f"{mc_c:.2f} / {mc_t:.2f}s" if mc_c is not None else "N/A"
    hk_str = f"{hk_c:.2f} / {hk_t:.2f}s" if hk_c is not None else "N/A"
    hk_cpu_str = f"{hk_cpu_c:.2f} / {hk_cpu_t:.2f}s" if hk_cpu_c is not None else "N/A"
    bf_str = f"{bf_c:.2f} / {bf_t:.4f}s" if bf_c is not None else "N/A"

    print(f"{res['Case']:<20} | {h_str:<20} | {r_str:<20} | {r2_str:<20} | "
          f"{dc_str:<20} | {mc_str:<20} | {hk_str:<20} | {hk_cpu_str:<20} | {bf_str:<15}")

print("=" * 215 + "\n")


# ── Export Results ────────────────────────────────────────────────────────────

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(output_dir, f"tsp_results_{timestamp}.txt")

with open(filename, "w") as f:
    f.write(f"TSP Optimization Results - {timestamp}\n")
    for res in results_summary:
        f.write(f"\n--- {res['Case']} ---\n")
        f.write(np.array2string(np.array(res['Matrix']), precision=2, separator=', ', threshold=sys.maxsize) + "\n\n")
        for key in ["Hybrid", "Refined", "Refined+2Opt", "Decomposition", "MonteCarlo", "HeldKarp", "HeldKarpCPU", "BruteForce"]:
            c, t, _ = res[key]
            f.write(f"{key:<16}: Cost={c:.2f}, Time={t:.4f}s\n" if c is not None else f"{key:<16}: Skipped\n")
        f.write("-" * 50 + "\n")

print(f"Results exported to {filename}")


# ── Plots ─────────────────────────────────────────────────────────────────────

print("Generating plots...")
n_vals = [len(res['Matrix']) for res in results_summary]


def extract(key):
    return [res[key][0] if res[key][0] is not None else np.nan for res in results_summary]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(n_vals, extract("Hybrid"), 'o-', label='Hybrid SPSA')
ax1.plot(n_vals, [res["Refined"][0] for res in results_summary], 's-', label='Refined SPSA')
ax1.plot(n_vals, [res["Refined+2Opt"][0] for res in results_summary], '^-', label='Refined + 2-Opt')
ax1.plot(n_vals, extract("Decomposition"), 'p-', label='Decomposition', color='green')
ax1.plot(n_vals, extract("MonteCarlo"), 'd--', label='Monte Carlo', color='purple', alpha=0.7)
ax1.plot(n_vals, extract("HeldKarp"), '*--', label='Held-Karp', color='orange', alpha=0.8)
ax1.set_title('Cost vs Number of Cities')
ax1.set_xlabel('Number of Cities')
ax1.set_ylabel('Cost')
ax1.legend()
ax1.grid(True)

ax2.plot(n_vals, [res["Hybrid"][1] or np.nan for res in results_summary], 'o-', label='Hybrid SPSA')
ax2.plot(n_vals, [res["Refined"][1] for res in results_summary], 's-', label='Refined SPSA')
ax2.set_title('Time vs Number of Cities')
ax2.set_xlabel('Number of Cities')
ax2.set_ylabel('Time (s)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True)

plot_filename = os.path.join(output_dir, f"tsp_results_{timestamp}.png")
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")
