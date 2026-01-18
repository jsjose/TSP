import numpy as np
import random


class GeneticTSPSolver:
    def __init__(self, dist_matrix, pop_size=100, mutation_rate=0.05, elitism_rate=0.1):
        self.B = np.array(dist_matrix)
        self.n = len(dist_matrix)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate

    def _calculate_dist(self, path):
        """Standard TSP distance calculation."""
        return sum(self.B[path[i]][path[i + 1]] for i in range(len(path) - 1))

    def _create_individual(self):
        """Creates a random valid tour starting and ending at 0."""
        cities = list(range(1, self.n))
        random.shuffle(cities)
        return [0] + cities + [0]

    def _crossover(self, parent1, parent2):
        """Partially Mapped Crossover (PMX) to maintain valid Hamiltonian cycles."""
        size = self.n
        p1, p2 = parent1[1:size], parent2[1:size]
        child = [None] * (size-1)
        
        # Randomly select a segment
        start, end = sorted(random.sample(range(size-1), 2))
        
        # Copy the segment from parent1 to the child
        child[start:end] = p1[start:end]
        
        # Map the remaining values from parent2
        for i in range(size-1):
            if child[i] is None:
                candidate = p2[i]
                while candidate in child:
                    candidate = p2[p1.index(candidate)]
                child[i] = candidate
        
        return [0] + child + [0]

    def _mutate(self, individual):
        """Scramble mutation to introduce genetic diversity."""
        if random.random() < self.mutation_rate:
            cities = individual[1:-1]
            start, end = sorted(random.sample(range(len(cities)), 2))
            scrambled_segment = cities[start:end]
            random.shuffle(scrambled_segment)
            cities[start:end] = scrambled_segment
            individual[1:-1] = cities
        return individual

    def solve(self, generations=500):
        """Evolutionary loop for TSP optimization."""
        # Initialize Population
        population = [self._create_individual() for _ in range(self.pop_size)]

        for _ in range(generations):
            # Sort by fitness (distance)
            population = sorted(population, key=self._calculate_dist)

            # Elitism
            elite_size = int(self.pop_size * self.elitism_rate)
            new_population = population[:elite_size]

            # Crossover and mutation
            while len(new_population) < self.pop_size:
                p1, p2 = random.sample(population, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        best_path = sorted(population, key=self._calculate_dist)[0]
        return best_path, self._calculate_dist(best_path)
