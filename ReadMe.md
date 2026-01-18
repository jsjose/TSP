# TSP Solvers

This project implements and compares different algorithms for solving the Traveling Salesperson Problem (TSP).

## Algorithms

The following algorithms are implemented:

*   **SingleQubitTSP**: A TSP solver based on a single qubit.
*   **two_opt_refinement**: A 2-opt refinement algorithm for improving existing solutions.
*   **solve_tsp_decomposition**: A TSP solver that uses k-means clustering to decompose the problem into smaller subproblems.
*   **GeneticTSPSolver**: A genetic algorithm for solving the TSP.

## Changes Made

The genetic algorithm in `tsp_ga.py` was giving a large gap from the optimal cost as the number of cities grew. The following changes were made to improve its performance:

*   **Improved Population Diversity**: The selection mechanism was modified to select from the entire population instead of the top 50 individuals. This reduces the selection pressure and promotes more diversity in the population, which helps to prevent premature convergence.
*   **Partially Mapped Crossover (PMX)**: The crossover operator was changed to the Partially Mapped Crossover (PMX), which is a more effective crossover operator for TSP.
*   **Scramble Mutation**: The mutation operator was changed to the scramble mutation, which is a more effective mutation operator for TSP.

## How to Run

To run the comparison script, execute the following command:

```bash
python test_tsplib.py
```

This will run the different TSP solvers on a sample problem and print a performance summary.