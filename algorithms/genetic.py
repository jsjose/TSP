import numpy as np
import random

from utils.helpers import calculate_total_distance


class GeneticTSPSolver:
    """Genetic Algorithm solver for TSP using PMX crossover and scramble mutation.

    Args:
        dist_matrix: NxN distance/cost matrix.
        pop_size: Number of individuals in the population.
        mutation_rate: Probability of mutating each individual.
        elitism_rate: Fraction of top individuals kept unchanged each generation.
    """

    def __init__(
        self,
        dist_matrix,
        pop_size: int = 100,
        mutation_rate: float = 0.05,
        elitism_rate: float = 0.1,
    ) -> None:
        self.B = np.array(dist_matrix)
        self.n = len(dist_matrix)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate

    def _fitness(self, path: list[int]) -> float:
        """Tour cost used as fitness (lower is better)."""
        return calculate_total_distance(path, self.B)

    def _create_individual(self) -> list[int]:
        """Creates a random valid tour starting and ending at city 0."""
        cities = list(range(1, self.n))
        random.shuffle(cities)
        return [0] + cities + [0]

    def _crossover(self, parent1: list[int], parent2: list[int]) -> list[int]:
        """Partially Mapped Crossover (PMX) to maintain valid Hamiltonian cycles.

        Args:
            parent1: First parent tour.
            parent2: Second parent tour.

        Returns:
            Child tour combining segments from both parents.
        """
        size = self.n
        p1, p2 = parent1[1:size], parent2[1:size]
        child: list[int | None] = [None] * (size - 1)

        # Copy a random segment from parent1
        start, end = sorted(random.sample(range(size - 1), 2))
        child[start:end] = p1[start:end]

        # Fill remaining positions using parent2's mapping
        for i in range(size - 1):
            if child[i] is None:
                candidate = p2[i]
                while candidate in child:
                    candidate = p2[p1.index(candidate)]
                child[i] = candidate

        return [0] + child + [0]

    def _mutate(self, individual: list[int]) -> list[int]:
        """Scramble mutation: randomly shuffles a segment of the tour.

        Args:
            individual: Tour to potentially mutate.

        Returns:
            Mutated (or unchanged) tour.
        """
        if random.random() < self.mutation_rate:
            cities = individual[1:-1]
            start, end = sorted(random.sample(range(len(cities)), 2))
            scrambled_segment = cities[start:end]
            random.shuffle(scrambled_segment)
            cities[start:end] = scrambled_segment
            individual[1:-1] = cities
        return individual

    def solve(self, generations: int = 500) -> tuple[list[int], float]:
        """Runs the evolutionary loop.

        Args:
            generations: Number of generational iterations.

        Returns:
            (path, cost): Best tour found and its total cost.
        """
        population = [self._create_individual() for _ in range(self.pop_size)]

        for _ in range(generations):
            population = sorted(population, key=self._fitness)

            elite_size = int(self.pop_size * self.elitism_rate)
            new_population = population[:elite_size]

            while len(new_population) < self.pop_size:
                p1, p2 = random.sample(population, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        best_path = sorted(population, key=self._fitness)[0]
        return best_path, self._fitness(best_path)
