"""Tests the RMSE+structure fitness factory."""

import pickle
from unittest import TestCase

import numpy as np

from src.ga import algorithm, fitness


class RMSEStructureFitnessTests(TestCase):
    """Covers factory output and process-backend compatibility."""

    def setUp(self) -> None:
        """Creates a small RGB target image for fitness evaluation."""

        self.target = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )

    def test_factory_returns_picklable_callable(self) -> None:
        """The configured fitness callable can be serialized for workers."""

        fitness_function = fitness.make_rmse_structure_fitness(1.0, 0.35)

        serialized = pickle.dumps(fitness_function)

        self.assertIsInstance(serialized, bytes)

    def test_process_backend_accepts_factory_output(self) -> None:
        """The GA accepts the factory output when using process evaluation."""

        ga = algorithm.GeneticAlgorithm(
            target=self.target,
            fitness_function=fitness.make_rmse_structure_fitness(1.0, 0.35),
            population_size=1,
            generations=1,
            n_triangles=1,
            evaluation_backend="process",
            n_jobs=1,
        )

        ga.initialize()
        fitness_values = ga.evaluate()

        self.assertEqual(1, len(fitness_values))
        self.assertTrue(np.isfinite(fitness_values[0]))
