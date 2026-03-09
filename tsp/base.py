"""Base class for all TSP solvers."""
from abc import ABC, abstractmethod
import numpy as np


class BaseSolver(ABC):
    """Abstract base class that all TSP solver classes must implement."""

    def __init__(self, cost_matrix):
        self.B = np.array(cost_matrix)
        self.n = len(cost_matrix)

    @abstractmethod
    def solve(self) -> tuple[list[int], float]:
        """Solve the TSP instance.

        Returns:
            (path, cost): path is a list of city indices starting and ending
            at 0, e.g. [0, 3, 1, 4, 2, 0]. cost is the total tour distance.
        """
