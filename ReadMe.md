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

==============================================================================================================
INSTANCE     | METHOD          | COST       | TIME (s)   | GAP (%)    | OPTIMAL   
--------------------------------------------------------------------------------------------------------------
burma14      | SPSA+2Opt       | 3448.0     | 0.46       | 3.76       | 3323.0    
             | Genetic Alg     | 3475.0     | 0.77       | 4.57       | 3323.0    
             | OR-Tools        | 3323.0     | 5.01       | 0.00       | 3323.0    
             | CPLEX           | 3323.0     | 0.06       | 0.00       | 3323.0    
             | Decomposition   | 5150.0     | 0.08       | 54.98      | 3323.0    
--------------------------------------------------------------------------------------------------------------
bayg29       | SPSA+2Opt       | 1625.0     | 0.97       | 0.93       | 1610.0    
             | Genetic Alg     | 1868.0     | 1.53       | 16.02      | 1610.0    
             | OR-Tools        | 1610.0     | 5.00       | 0.00       | 1610.0    
             | CPLEX           | 1610.0     | 0.24       | 0.00       | 1610.0    
             | Decomposition   | 3066.0     | 0.11       | 90.43      | 1610.0    
--------------------------------------------------------------------------------------------------------------
att48        | SPSA+2Opt       | 11080.0    | 1.96       | 4.25       | 10628.0   
             | Genetic Alg     | 16810.0    | 2.91       | 58.17      | 10628.0   
             | OR-Tools        | 10628.0    | 9.00       | 0.00       | 10628.0   
             | Decomposition   | 17548.0    | 0.18       | 65.11      | 10628.0   
--------------------------------------------------------------------------------------------------------------
berlin52     | SPSA+2Opt       | 8377.0     | 2.27       | 11.07      | 7542.0    
             | Genetic Alg     | 12196.0    | 3.26       | 61.71      | 7542.0    
             | OR-Tools        | 7542.0     | 10.00      | 0.00       | 7542.0    
             | Decomposition   | 13561.0    | 0.19       | 79.81      | 7542.0    
--------------------------------------------------------------------------------------------------------------
brazil58     | SPSA+2Opt       | 25704.0    | 3.29       | 1.22       | 25395.0   
             | Genetic Alg     | 45591.0    | 3.82       | 79.53      | 25395.0   
             | OR-Tools        | 25395.0    | 11.00      | 0.00       | 25395.0   
--------------------------------------------------------------------------------------------------------------
dantzig42    | SPSA+2Opt       | 747.0      | 1.53       | 6.87       | 699.0     
             | Genetic Alg     | 1050.0     | 2.47       | 50.21      | 699.0     
             | OR-Tools        | 699.0      | 8.00       | 0.00       | 699.0     
             | Decomposition   | 1242.0     | 0.16       | 77.68      | 699.0     
--------------------------------------------------------------------------------------------------------------
ali535       | SPSA+2Opt       | Skip       | Skip       | Skip       | 202339.0  
             | Genetic Alg     | Skip       | Skip       | Skip       | 202339.0  
             | OR-Tools        | 213364.0   | 107.00     | 5.45       | 202339.0  
--------------------------------------------------------------------------------------------------------------
dsj1000      | SPSA+2Opt       | Skip       | Skip       | Skip       | 18660188.0
             | Genetic Alg     | Skip       | Skip       | Skip       | 18660188.0
             | OR-Tools        | 19459947.0 | 200.01     | 4.29       | 18660188.0
--------------------------------------------------------------------------------------------------------------
brg180       | SPSA+2Opt       | Skip       | Skip       | Skip       | 1950.0    
             | Genetic Alg     | Skip       | Skip       | Skip       | 1950.0    
             | OR-Tools        | 1950.0     | 36.00      | 0.00       | 1950.0    
--------------------------------------------------------------------------------------------------------------