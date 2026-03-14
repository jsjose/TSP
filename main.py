"""
TSP Solver Project — Entry Point

Runs the full benchmark against standard TSPLIB instances.
Requires the tsplib/ directory with .tsp files and a solutions file.

Usage:
    python main.py

See test_tsplib.py for the benchmark implementation.
"""
from test_tsplib import run_benchmark

if __name__ == "__main__":
    run_benchmark()
