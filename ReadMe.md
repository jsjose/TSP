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
