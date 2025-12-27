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

## Python Implementation

This implementation uses `numpy` for the quantum state math and a custom SPSA loop to solve a 4-city TSP.

===================================================================================================================
TEST CASE            | HYBRID (Cost/Time)        | REFINED (Cost/Time)       | REF+2OPT (Cost/Time)      | BRUTE FORCE    
-------------------------------------------------------------------------------------------------------------------
4-City Symmetric     | 80.00 / 0.14s             | 80.00 / 0.19s             | 80.00 / 0.19s             | 80.00 / 0.0000s
5-City Asymmetric    | 34.00 / 0.17s             | 36.00 / 0.22s             | 34.00 / 0.22s             | 34.00 / 0.0000s
8-City Symmetric     | 105.00 / 0.25s            | 107.00 / 0.33s            | 105.00 / 0.33s            | 99.00 / 0.0077s
6-City Random        | 241.07 / 0.19s            | 241.07 / 0.27s            | 241.07 / 0.27s            | 241.07 / 0.0003s
7-City Random        | 276.22 / 0.22s            | 310.31 / 0.29s            | 276.22 / 0.29s            | 276.22 / 0.0010s
10-City Random       | 290.31 / 0.34s            | 420.37 / 0.40s            | 290.31 / 0.40s            | 290.31 / 0.7098s
11-City Random       | 296.25 / 0.38s            | 320.69 / 0.43s            | 296.25 / 0.44s            | 296.25 / 7.7026s
12-City Random       | 297.25 / 0.41s            | 429.91 / 0.47s            | 297.25 / 0.47s            | 297.25 / 88.5304s
13-City Random       | 298.93 / 0.52s            | 454.28 / 0.51s            | 298.93 / 0.51s            | N/A            
17-City Random       | 322.07 / 0.97s            | 561.84 / 0.68s            | 322.07 / 0.68s            | N/A            
20-City Random       | 386.43 / 1.39s            | 692.19 / 0.77s            | 386.43 / 0.78s            | N/A            
50-City Random       | 588.54 / 106.97s          | 2024.38 / 1.91s           | 573.93 / 2.22s            | N/A            
100-City Random      | 791.56 / 6340.06s         | 4631.91 / 3.93s           | 806.63 / 11.19s           | N/A            
===================================================================================================================
