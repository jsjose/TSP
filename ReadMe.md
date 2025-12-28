# Solving The Travelling Salesman Problem Using A Single Qubit

This paper, ["Solving The Travelling Salesman Problem Using A Single Qubit" (arXiv:2407.17207)](https://arxiv.org/pdf/2407.17207), presents a fascinating "quantum-inspired" approach. It maps the TSP into a Discrete Brachistochrone problem and solves it using a single qubit by encoding the cities and their distances as states on the Bloch Sphere.

The core idea is to use the qubit's ability to exist in a superposition to "traverse" multiple paths simultaneously. We then use an optimizer (SPSA) to tune the rotation parameters so the system collapses into the state representing the shortest path.

## Implementation Plan

1.  **Data Encoding:**
    *   **Cities:** Represent each city $c_i$ as a state $|P_{ii}\rangle$ on the equator of the Bloch sphere.
    *   **Distances:** Encode the distance $s_{ij}$ between city $i$ and $j$ as an angle $\theta_{ij}$ on the geodesic connecting $|P_{ii}\rangle$ to the pole $|0\rangle$.

2.  **State Evolution:**
    *   Construct a "Routing Chart" (layers of states).
    *   Apply rotation operators ($U^u$ for "upward" to an intermediate distance-state, $U^d$ for "downward" to the next city-state).

3.  **Optimization (SPSA):**
    *   Define a cost function based on the expected distance of the paths.
    *   Use SPSA (Simultaneous Perturbation Stochastic Approximation) to iteratively update rotation angles to minimize the cost.

4.  **Measurement & Decoding:**
    *   Perform "tomography" (state measurement) to find the path with the highest probability.

## Algorithms Implemented

In addition to the paper's method, this repository includes several other algorithms for benchmarking and comparison, leveraging Apple Silicon (MLX) for GPU acceleration where possible.

### 1. Hybrid SPSA (Paper Implementation)
This is the core algorithm from the referenced paper. It operates in three phases:
*   **Phase 1 (1SPSA):** First-order Simultaneous Perturbation Stochastic Approximation for global exploration.
*   **Phase 2 (2SPSA):** Second-order SPSA (using Hessian approximation) for curvature-based fine-tuning near the minimum.
*   **Phase 3 (2-Opt):** A classical local search algorithm to untangle any remaining crossing edges in the path.

### 2. Refined SPSA
A variation of the SPSA optimizer that incorporates **Momentum**. This helps the optimizer navigate the cost landscape more effectively, avoiding shallow local minima and converging faster in some scenarios.

### 3. Monte Carlo (MLX Accelerated)
A probabilistic approach that samples millions of random permutations to approximate the optimal solution.
*   **Implementation:** Uses `mlx.core` to generate and evaluate batches of permutations in parallel on the GPU.
*   **Scale:** Can evaluate hundreds of millions of paths in seconds, providing a strong baseline for larger N where exact methods fail.

### 4. Held-Karp (MLX Accelerated)
An exact algorithm for TSP based on Dynamic Programming.
*   **Complexity:** $O(n^2 2^n)$.
*   **Implementation:** Vectorized using `mlx.core` to perform state updates in parallel on the GPU.
*   **Limitation:** Due to exponential memory usage, it is feasible only up to $N \approx 18-20$.

### 5. Brute Force
The naive exact approach that evaluates all $(N-1)!$ possible permutations.
*   **Implementation:** Uses Python's `itertools`.
*   **Limitation:** Feasible only for very small $N$ ($N < 13$).

## Project Structure

*   **`2spsa.py`**: The main execution script. It runs a comprehensive benchmark suite across various problem sizes (from 4 to 100 cities), executing the algorithms described above and exporting results.
*   **`bruteforce_mlx.py`**: A standalone script demonstrating the MLX-based vectorized logic for batch distance calculations used in the Monte Carlo solver.
*   **`results/`**: Directory where the benchmark outputs are stored.
    *   `tsp_results_YYYYMMDD_HHMMSS.txt`: Detailed text report of the run.
    *   `tsp_results_YYYYMMDD_HHMMSS.png`: Plots comparing Cost and Time vs. Number of Cities.

## Results

The benchmark compares the "Quantum-Inspired" Hybrid approach against classical exact and approximate methods.

### Example Output Table

```text
=====================================================================================================================================================================
TEST CASE            | HYBRID (Cost/Time)   | REFINED (Cost/Time)  | REF+2OPT (Cost/Time) | MONTE CARLO          | HELD KARP            | BRUTE FORCE    
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
4-City Symmetric     | 80.00 / 0.14s        | 80.00 / 0.19s        | 80.00 / 0.19s        | 80.00 / 0.08s        | 80.00 / 0.16s        | 80.00 / 0.0000s
5-City Asymmetric    | 34.00 / 0.17s        | 36.00 / 0.22s        | 34.00 / 0.22s        | 34.00 / 0.01s        | 34.00 / 0.02s        | 34.00 / 0.0000s
8-City Symmetric     | 105.00 / 0.25s       | 107.00 / 0.33s       | 105.00 / 0.33s       | 99.00 / 0.01s        | 99.00 / 0.03s        | 99.00 / 0.0077s
12-City Random       | 297.25 / 0.41s       | 429.91 / 0.47s       | 297.25 / 0.47s       | 297.25 / 0.05s       | 297.25 / 0.12s       | 297.25 / 88.5304s
20-City Random       | 386.43 / 1.39s       | 692.19 / 0.77s       | 386.43 / 0.78s       | 386.43 / 0.85s       | Skipped              | Skipped
50-City Random       | 588.54 / 106.97s     | 2024.38 / 1.91s      | 573.93 / 2.22s       | 590.12 / 5.40s       | Skipped              | Skipped
=====================================================================================================================================================================
```

### Performance Visualization

The script generates a plot comparing the scalability of the algorithms:

!TSP Results Plot
*(Note: This image is generated in the `results/` directory after running `2spsa.py`)*
