import time
import numpy as np

import utils.loader as tsp_loader
from algorithms.decomposition import solve_tsp_decomposition
from algorithms.spsa import SingleQubitTSP, two_opt_refinement
from algorithms.genetic import GeneticTSPSolver
from algorithms.ortools import ORToolsTSPSolver
from algorithms.cplex import CPLEXTSPSolver

try:
    from algorithms.lkh import LKH3Solver
except ImportError:
    LKH3Solver = None

import algorithms.lkh as lkh_mod
lkh_mod.configure("/Users/chemasj/Downloads/LKH-3.0.13/LKH")   # se aplica a todas las instancias

def calculate_gap(cost, optimal):
    """Returns percentage gap between a solution cost and the known optimal."""
    if optimal and optimal > 0:
        return (cost - optimal) / optimal * 100
    return float("nan")


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
            "optimal": instance.optimal_cost,
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
            res["opt_time"] = (time.time() - start) + res["spsa_time"]  # Cumulative time
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
        if (
            instance.coords is not None
            and len(instance.coords) > 0
            and solver.n >= 10
            and instance.dimension < 100
        ):
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
            res["cplex_gap"] = (
                calculate_gap(cplex_cost, instance.optimal_cost)
                if cplex_cost is not None
                else float("nan")
            )
        except Exception as e:
            print(f"    > CPLEX failed: {e}")
            res["cplex_cost"] = None

        # --- LKH ---
        if LKH3Solver is None or not LKH3Solver.is_available():
            print("  > Skipping LKH (binary not found or module unavailable)")
            res["lkh_cost"] = None
            res["lkh_time"] = None
            res["lkh_gap"] = None
        else:
            print("  > Running LKH...")
            start = time.time()
            try:
                lkh_solver = LKH3Solver(instance.distance_matrix)
                _, lkh_cost = lkh_solver.solve()
                res["lkh_time"] = time.time() - start
                res["lkh_cost"] = lkh_cost
                res["lkh_gap"] = (
                    calculate_gap(lkh_cost, instance.optimal_cost)
                    if lkh_cost is not None
                    else float("nan")
                )
            except Exception as e:
                print(f"    > LKH failed: {e}")
                res["lkh_cost"] = None
                res["lkh_time"] = None
                res["lkh_gap"] = None

        results_summary.append(res)

        # --- Print Summary Table ---
        print("\n\n" + "=" * 110)
        print(
            f"{'INSTANCE':<12} | {'METHOD':<15} | {'COST':<10} | "
            f"{'TIME (s)':<10} | {'GAP (%)':<10} | {'OPTIMAL':<10}"
        )
        print("-" * 110)

        for r in results_summary:
            opt_str = str(r["optimal"]) if r["optimal"] else "N/A"

            def p_row(name, method, cost, time_v, gap, opt):
                c_s = f"{cost:<10.1f}" if cost is not None else "Skip      "
                t_s = f"{time_v:<10.2f}" if time_v is not None else "Skip      "
                g_s = f"{gap:<10.2f}" if gap is not None else "Skip      "
                print(f"{name:<12} | {method:<15} | {c_s} | {t_s} | {g_s} | {opt:<10}")

            p_row(r["name"], "SPSA+2Opt", r.get("opt_cost"), r.get("opt_time"), r.get("opt_gap"), opt_str)
            p_row("", "Genetic Alg", r.get("ga_cost"), r.get("ga_time"), r.get("ga_gap"), opt_str)
            p_row("", "OR-Tools", r.get("ort_cost"), r.get("ort_time"), r.get("ort_gap"), opt_str)

            if r.get("cplex_cost") is not None:
                p_row("", "CPLEX", r["cplex_cost"], r["cplex_time"], r["cplex_gap"], opt_str)

            if r.get("lkh_cost") is not None:
                p_row("", "LKH", r["lkh_cost"], r["lkh_time"], r["lkh_gap"], opt_str)

            if r["decomp_cost"] is not None:
                p_row("", "Decomposition", r["decomp_cost"], r["decomp_time"], r["decomp_gap"], opt_str)

            print("-" * 110)


if __name__ == "__main__":
    run_benchmark()
