import tsp_loader
from tsp_decomposition import solve_tsp_decomposition
from tsp_spsa import SingleQubitTSP, two_opt_refinement
from tsp_ga import GeneticTSPSolver
from tsp_ortools import ORToolsTSPSolver
from tsp_cplex import CPLEXTSPSolver
import time
import numpy as np

def calculate_gap(cost, optimal):
    if optimal and optimal > 0:
        return (cost - optimal) / optimal * 100
    return float('nan')

def run_benchmark():
    # 1. Load the master solutions file once
    solutions_map = tsp_loader.load_solutions_file("tsplib/solutions")

    # 2. Define test set
    tsp_files = [
        "tsplib/burma14.tsp",
        "tsplib/bayg29.tsp",
        "tsplib/att48.tsp",
        "tsplib/berlin52.tsp",
        "tsplib/brazil58.tsp",
        "tsplib/dantzig42.tsp",
        "tsplib/ali535.tsp", 
        "tsplib/dsj1000.tsp",
        "tsplib/brg180.tsp",
    ]

    results_summary = []

    for filepath in tsp_files:
        print(f"\n{'='*60}")
        print(f"Loading {filepath}...")
        try:
            instance = tsp_loader.load_tsp(filepath, solutions_map)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            continue

        print(f"Instance: {instance.name} (N={instance.dimension})")
        print(f"Target Optimal: {instance.optimal_cost}")
        
        solver = SingleQubitTSP(instance.distance_matrix)
        res = {
            "name": instance.name,
            "n": instance.dimension,
            "optimal": instance.optimal_cost
        }

        # --- Refined SPSA ---
        if instance.dimension < 100:
            print("  > Running Refined SPSA...")
            start = time.time()
            refined_path, refined_cost = solver.solve_refined(trials=3, iterations=1000)
            res["spsa_time"] = time.time() - start
            res["spsa_cost"] = refined_cost
            res["spsa_gap"] = calculate_gap(refined_cost, instance.optimal_cost)

            # --- 2-Opt ---
            print("  > Running 2-Opt Refinement...")
            start = time.time()
            opt_path, opt_cost = two_opt_refinement(refined_path, solver.B)
            res["opt_time"] = (time.time() - start) + res["spsa_time"] # Cumulative time
            res["opt_cost"] = opt_cost
            res["opt_gap"] = calculate_gap(opt_cost, instance.optimal_cost)
        else:
            print(f"  > Skipping SPSA & 2-Opt (N={instance.dimension} >= 100)")
            res["spsa_cost"] = None
            res["spsa_time"] = None
            res["opt_cost"] = None
            res["opt_time"] = None
            res["opt_gap"] = None

        # --- Decomposition ---
        res["decomp_cost"] = None
        if instance.coords is not None and len(instance.coords) > 0 and solver.n >= 10 and instance.dimension < 100:
            print("  > Running Decomposition...")
            start = time.time()
            k_clusters = max(2, int(solver.n / 8))
            _, d_cost, _, _ = solve_tsp_decomposition(solver, instance.coords, k=k_clusters)
            res["decomp_time"] = time.time() - start
            res["decomp_cost"] = d_cost
            res["decomp_gap"] = calculate_gap(d_cost, instance.optimal_cost)
        
        # --- GA ---
        if instance.dimension < 100:
            print("  > Running Genetic Algorithm...")
            start = time.time()
            ga_solver = GeneticTSPSolver(instance.distance_matrix, pop_size=150, mutation_rate=0.1)
            _, ga_cost = ga_solver.solve(generations=1000)
            res["ga_time"] = time.time() - start
            res["ga_cost"] = ga_cost
            res["ga_gap"] = calculate_gap(ga_cost, instance.optimal_cost)
        else:
            print(f"  > Skipping Genetic Algorithm (N={instance.dimension} >= 100)")
            res["ga_cost"] = None
            res["ga_time"] = None
            res["ga_gap"] = None

        # --- OR-Tools ---
        print("  > Running OR-Tools...")
        start = time.time()
        ort_solver = ORToolsTSPSolver(instance.distance_matrix)
        or_time_limit = max(5, int(instance.dimension / 5))
        _, ort_cost = ort_solver.solve(time_limit_seconds=or_time_limit)
        res["ort_time"] = time.time() - start
        res["ort_cost"] = ort_cost
        res["ort_gap"] = calculate_gap(ort_cost, instance.optimal_cost)

        # --- CPLEX ---
        print("  > Running CPLEX...")
        start = time.time()
        try:
            cplex_solver = CPLEXTSPSolver(instance.distance_matrix)
            _, cplex_cost = cplex_solver.solve(log_output=False)
            res["cplex_time"] = time.time() - start
            res["cplex_cost"] = cplex_cost
            res["cplex_gap"] = calculate_gap(cplex_cost, instance.optimal_cost) if cplex_cost is not None else float('nan')
        except Exception as e:
            print(f"    > CPLEX failed: {e}")
            res["cplex_cost"] = None

        results_summary.append(res)

    # --- Print Summary Table ---
    print("\n\n" + "="*110)
    print(f"{'INSTANCE':<12} | {'METHOD':<15} | {'COST':<10} | {'TIME (s)':<10} | {'GAP (%)':<10} | {'OPTIMAL':<10}")
    print("-" * 110)

    for res in results_summary:
        opt_str = str(res['optimal']) if res['optimal'] else "N/A"

        def p_row(name, method, cost, time_v, gap, opt):
            c_s = f"{cost:<10.1f}" if cost is not None else "Skip      "
            t_s = f"{time_v:<10.2f}" if time_v is not None else "Skip      "
            g_s = f"{gap:<10.2f}" if gap is not None else "Skip      "
            print(f"{name:<12} | {method:<15} | {c_s} | {t_s} | {g_s} | {opt:<10}")
        
        # SPSA+2Opt
        p_row(res['name'], 'SPSA+2Opt', res.get('opt_cost'), res.get('opt_time'), res.get('opt_gap'), opt_str)
        
        # GA
        p_row('', 'Genetic Alg', res.get('ga_cost'), res.get('ga_time'), res.get('ga_gap'), opt_str)
        
        # OR-Tools
        p_row('', 'OR-Tools', res.get('ort_cost'), res.get('ort_time'), res.get('ort_gap'), opt_str)

        # CPLEX
        if res.get('cplex_cost') is not None:
            p_row('', 'CPLEX', res['cplex_cost'], res['cplex_time'], res['cplex_gap'], opt_str)

        # Decomp
        if res['decomp_cost'] is not None:
            p_row('', 'Decomposition', res['decomp_cost'], res['decomp_time'], res['decomp_gap'], opt_str)
        
        print("-" * 110)

if __name__ == "__main__":
    run_benchmark()