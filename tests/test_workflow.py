"""Tests staged workflow parameter forwarding and validation."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src import population
from src.ga import cross_over, fitness, mutate, workflow


def _zero_fitness(target: np.ndarray, generated: np.ndarray) -> float:
    """Returns a constant fitness value for workflow tests."""

    return 0.0


class StagedWorkflowTests(TestCase):
    """Covers staged workflow configuration behavior."""

    def setUp(self) -> None:
        """Creates a tiny target image for staged workflow tests."""

        self.target = np.zeros((2, 2, 3), dtype=np.uint8)

    def test_stage_stagnation_window_is_forwarded_to_ga(self) -> None:
        """Each stage forwards its stagnation window to GeneticAlgorithm."""

        best_individual = [
            population.Triangle(
                x1=0,
                y1=0,
                x2=1,
                y2=0,
                x3=0,
                y3=1,
                r=10,
                g=20,
                b=30,
                a=40,
            )
        ]
        stage = workflow.StageConfig(
            n_triangles=1,
            generations=2,
            mutation_rate=0.1,
            adaptive_mutation=True,
            stagnation_window=35,
        )

        with patch("src.ga.workflow.GeneticAlgorithm") as mock_ga_class:
            mock_ga = mock_ga_class.return_value
            mock_ga.run.return_value = (0.25, [0.4, 0.25])
            mock_ga.best_individual = best_individual

            result = workflow.run_staged_triangle_optimization(
                target=self.target,
                fitness_function=_zero_fitness,
                population_size=3,
                stages=[stage],
            )

        constructor_kwargs = mock_ga_class.call_args.kwargs
        self.assertEqual(35, constructor_kwargs["stagnation_window"])
        self.assertEqual(0.25, result.best_fitness)
        self.assertEqual(best_individual, result.best_individual)

    def test_stage_stagnation_window_must_be_positive(self) -> None:
        """Invalid stage stagnation windows raise a clear ValueError."""

        stage = workflow.StageConfig(
            n_triangles=1,
            generations=1,
            stagnation_window=0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Each stage stagnation_window must be greater than zero.",
        ):
            workflow.run_staged_triangle_optimization(
                target=self.target,
                fitness_function=_zero_fitness,
                population_size=2,
                stages=[stage],
            )

    def test_staged_process_backend_runs_with_seeded_two_child_crossover(self) -> None:
        """Staged process evaluation keeps worker initializer args in sync."""

        np.random.seed(19)
        stage = workflow.StageConfig(
            n_triangles=3,
            generations=1,
            mutation_rate=0.01,
            crossover_rate=0.85,
            adaptive_mutation=True,
            mutation_rate_bounds=(0.001, 0.02),
            stagnation_window=2,
            random_immigrants=0,
        )

        result = workflow.run_staged_triangle_optimization(
            target=self.target,
            fitness_function=fitness.compute_rmse,
            population_size=3,
            stages=[stage],
            elitism=1,
            crossover_function=cross_over.two_point_crossover_two_children,
            mutation_function=mutate.random_triangle_mutation,
            evaluation_backend="process",
            n_jobs=1,
            chunksize=1,
            seeded=True,
            seed_mutation_rate=0.05,
        )

        self.assertTrue(np.isfinite(result.best_fitness))
        self.assertEqual(1, len(result.history))
        self.assertEqual(3, len(result.best_individual))
