"""Tests GA crossover compatibility with single-child and multi-child operators."""

from unittest import TestCase

import numpy as np

from src import population
from src.ga import algorithm, cross_over, mutate


def _zero_fitness(target: np.ndarray, generated: np.ndarray) -> float:
    """Returns a constant fitness value for operator-path tests."""

    return 0.0


def _make_target() -> np.ndarray:
    """Creates a tiny RGB target for deterministic GA tests."""

    return np.zeros((3, 3, 3), dtype=np.uint8)


class GeneticAlgorithmCrossoverCompatibilityTests(TestCase):
    """Covers crossover operators that emit one or many children."""

    def test_run_accepts_two_child_crossover_functions(self) -> None:
        """GA runs successfully when crossover returns two children."""

        np.random.seed(5)
        ga = algorithm.GeneticAlgorithm(
            target=_make_target(),
            fitness_function=_zero_fitness,
            population_size=4,
            generations=2,
            n_triangles=3,
            crossover_function=cross_over.two_point_crossover_two_children,
            crossover_rate=1.0,
            mutation_function=mutate.random_triangle_mutation,
            mutation_rate=1.0,
        )

        best_fitness, history = ga.run()

        self.assertEqual(0.0, best_fitness)
        self.assertEqual([0.0, 0.0], history)
        self.assertEqual(4, len(ga.population))
        for individual in ga.population:
            self.assertEqual(3, len(individual))
            for triangle in individual:
                self.assertIsInstance(triangle, population.Triangle)

    def test_crossover_without_operator_still_returns_one_child(self) -> None:
        """Fallback crossover keeps the normalized child collection shape."""

        np.random.seed(7)
        ga = algorithm.GeneticAlgorithm(
            target=_make_target(),
            fitness_function=_zero_fitness,
            population_size=2,
            generations=1,
            n_triangles=3,
        )
        parent1 = population.create_random_individual(
            n_triangles=3,
            image_width=3,
            image_height=3,
        )
        parent2 = population.create_random_individual(
            n_triangles=3,
            image_width=3,
            image_height=3,
        )

        children = ga.crossover(parent1, parent2)

        self.assertEqual(1, len(children))
        self.assertEqual(3, len(children[0]))
