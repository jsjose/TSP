"""TSP CLI — command-line interface for the TSP solver library.

Commands
--------
  solve      Run one solver on one .tsp file
  benchmark  Run all eligible solvers on one or more .tsp files
  results    Query stored results from the database

Examples
--------
  python cli.py solve tsplib/berlin52.tsp --solver ortools
  python cli.py solve tsplib/att48.tsp --solver spsa --store
  python cli.py benchmark --instances tsplib/burma14.tsp tsplib/att48.tsp
  python cli.py benchmark --instances tsplib/berlin52.tsp --solvers ortools genetic --store
  python cli.py results
  python cli.py results --instance berlin52 --solver ortools --last 5
"""

import argparse
import json
import sys
import time

from tsp.registry import get_solver, list_solvers, solvers_for_n, SOLVER_META
from tsp.store import ResultStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gap(cost, optimal):
    if cost is not None and optimal and optimal > 0:
        return (cost - optimal) / optimal * 100
    return None


def _run_solver(name, cost_matrix, n, **kwargs):
    """Instantiate and run a solver. Returns (path, cost, elapsed_s)."""
    from lib.tsp_spsa import two_opt_refinement, SingleQubitTSP
    cls = get_solver(name)
    t0 = time.time()

    if name == "spsa":
        solver = cls(cost_matrix)
        path, cost = solver.solve_refined(trials=3, iterations=1000)
        path, cost = two_opt_refinement(path, solver.B)
    else:
        solver = cls(cost_matrix)
        path, cost = solver.solve(**kwargs)

    return path, cost, time.time() - t0


def _ortools_time_limit(n):
    return max(5, int(n / 5))


def _print_table_header():
    print("=" * 110)
    print(f"{'INSTANCE':<12} | {'SOLVER':<15} | {'COST':<10} | {'TIME (s)':<10} | {'GAP (%)':<10} | {'OPTIMAL':<10}")
    print("-" * 110)


def _print_row(instance, solver_label, cost, elapsed, gap, optimal):
    c_s = f"{cost:<10.1f}" if cost is not None else f"{'—':<10}"
    t_s = f"{elapsed:<10.2f}" if elapsed is not None else f"{'—':<10}"
    g_s = f"{gap:<10.2f}" if gap is not None else f"{'—':<10}"
    o_s = str(optimal) if optimal is not None else "N/A"
    print(f"{instance:<12} | {solver_label:<15} | {c_s} | {t_s} | {g_s} | {o_s:<10}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_solve(args):
    import lib.tsp_loader as tsp_loader
    solutions_map = tsp_loader.load_solutions_file("tsplib/solutions")
    try:
        instance = tsp_loader.load_tsp(args.tsp_file, solutions_map)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    solver_name = args.solver
    n = instance.dimension
    meta = SOLVER_META.get(solver_name, {})
    max_n = meta.get("max_n")
    if max_n is not None and n > max_n:
        print(
            f"Warning: solver '{solver_name}' is not recommended for n={n} (max {max_n}). "
            "Proceeding anyway."
        )

    print(f"Instance : {instance.name}  (n={n})")
    print(f"Solver   : {solver_name}")
    print(f"Optimal  : {instance.optimal_cost}")
    print("Running…")

    kwargs = {}
    if solver_name == "ortools":
        kwargs["time_limit_seconds"] = _ortools_time_limit(n)

    try:
        path, cost, elapsed = _run_solver(solver_name, instance.distance_matrix, n, **kwargs)
    except Exception as e:
        print(f"Solver error: {e}", file=sys.stderr)
        sys.exit(1)

    gap = _gap(cost, instance.optimal_cost)

    _print_table_header()
    label = SOLVER_META.get(solver_name, {}).get("label", solver_name)
    _print_row(instance.name, label, cost, elapsed, gap, instance.optimal_cost)
    print("-" * 110)

    if args.store:
        store = ResultStore(args.db)
        row_id = store.save(
            instance=instance.name,
            solver=solver_name,
            n=n,
            cost=cost,
            time_s=elapsed,
            gap_pct=gap,
            optimal=instance.optimal_cost,
            path=path,
        )
        print(f"\nStored as result #{row_id} in {args.db}")


def cmd_benchmark(args):
    import lib.tsp_loader as tsp_loader
    solutions_map = tsp_loader.load_solutions_file("tsplib/solutions")
    store = ResultStore(args.db) if args.store else None

    _print_table_header()

    for filepath in args.instances:
        try:
            instance = tsp_loader.load_tsp(filepath, solutions_map)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}", file=sys.stderr)
            continue

        n = instance.dimension

        if args.solvers:
            solver_names = args.solvers
        else:
            solver_names = solvers_for_n(n)

        first = True
        for solver_name in solver_names:
            if solver_name not in list_solvers():
                print(f"  Skipping '{solver_name}': not available (missing dependency?)")
                continue

            meta = SOLVER_META.get(solver_name, {})
            max_n = meta.get("max_n")
            label = meta.get("label", solver_name)

            if max_n is not None and n > max_n:
                _print_row(
                    instance.name if first else "",
                    label, None, None, None, instance.optimal_cost
                )
                first = False
                continue

            kwargs = {}
            if solver_name == "ortools":
                kwargs["time_limit_seconds"] = _ortools_time_limit(n)

            try:
                path, cost, elapsed = _run_solver(
                    solver_name, instance.distance_matrix, n, **kwargs
                )
            except Exception as e:
                print(f"  {label} error: {e}", file=sys.stderr)
                path, cost, elapsed = None, None, None

            gap = _gap(cost, instance.optimal_cost)
            _print_row(
                instance.name if first else "",
                label, cost, elapsed, gap, instance.optimal_cost
            )
            first = False

            if store and cost is not None:
                store.save(
                    instance=instance.name,
                    solver=solver_name,
                    n=n,
                    cost=cost,
                    time_s=elapsed,
                    gap_pct=gap,
                    optimal=instance.optimal_cost,
                    path=path,
                )

        print("-" * 110)

    if store:
        print(f"\nResults stored in {args.db}")


def cmd_results(args):
    store = ResultStore(args.db)
    rows = store.query(instance=args.instance, solver=args.solver, last=args.last)

    if not rows:
        print("No results found.")
        return

    _print_table_header()
    for row in rows:
        _print_row(
            row["instance"],
            row["solver"],
            row["cost"],
            row["time_s"],
            row["gap_pct"],
            row["optimal"],
        )
    print("-" * 110)
    print(f"\n{len(rows)} result(s) from {args.db}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="TSP solver CLI — solve instances, run benchmarks, query results.",
    )
    parser.add_argument(
        "--db",
        default="results.db",
        metavar="FILE",
        help="SQLite database file for stored results (default: results.db)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- solve ----------------------------------------------------------------
    p_solve = sub.add_parser("solve", help="Run one solver on one .tsp file.")
    p_solve.add_argument("tsp_file", help="Path to a .tsp file")
    p_solve.add_argument(
        "--solver",
        default="ortools",
        choices=["spsa", "genetic", "ortools", "cplex", "lkh", "held_karp"],
        help="Solver to use (default: ortools)",
    )
    p_solve.add_argument(
        "--store", action="store_true", help="Persist result to the database"
    )

    # -- benchmark ------------------------------------------------------------
    p_bench = sub.add_parser(
        "benchmark", help="Run solvers on one or more .tsp files."
    )
    p_bench.add_argument(
        "--instances",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more .tsp file paths",
    )
    p_bench.add_argument(
        "--solvers",
        nargs="+",
        choices=["spsa", "genetic", "ortools", "cplex", "lkh", "held_karp"],
        default=None,
        metavar="SOLVER",
        help="Solvers to run (default: all eligible for instance size)",
    )
    p_bench.add_argument(
        "--store", action="store_true", help="Persist results to the database"
    )

    # -- results --------------------------------------------------------------
    p_res = sub.add_parser("results", help="Query stored results from the database.")
    p_res.add_argument("--instance", default=None, help="Filter by instance name")
    p_res.add_argument("--solver", default=None, help="Filter by solver name")
    p_res.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Show only the N most recent results",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "solve":
        cmd_solve(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "results":
        cmd_results(args)


if __name__ == "__main__":
    main()
