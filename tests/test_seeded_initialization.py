"""Tests seeded RGB and alpha-range triangle initialization."""

from unittest import TestCase
from unittest.mock import call, patch

import numpy as np

from src import population
from src.ga import algorithm, workflow


def _zero_fitness(target: np.ndarray, generated: np.ndarray) -> float:
    """Returns a constant fitness value for initialization-only tests."""

    return 0.0


def _triangle_with_color(rgb: tuple[int, int, int]) -> population.Triangle:
    """Creates one valid triangle with a known color."""

    return population.Triangle(
        x1=0,
        y1=0,
        x2=1,
        y2=0,
        x3=0,
        y3=1,
        r=rgb[0],
        g=rgb[1],
        b=rgb[2],
        a=255,
    )


class SeededPopulationFactoryTests(TestCase):
    """Covers target-biased color sampling in population factories."""

    def setUp(self) -> None:
        """Sets deterministic random state and a small RGB target."""

        np.random.seed(7)
        self.target = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        self.target_colors = {
            tuple(int(value) for value in pixel)
            for row in self.target
            for pixel in row
        }

    def test_seeded_population_colors_come_from_target_pixels(self) -> None:
        """Seeded factories copy exact target-pixel RGB triplets."""

        created_population = population.create_population(
            population_size=4,
            n_triangles=5,
            image_width=2,
            image_height=2,
            triangle_alpha_range=(33, 33),
            target=self.target,
            seeded=True,
        )

        self.assertEqual(4, len(created_population))
        for individual in created_population:
            self.assertEqual(5, len(individual))
            for triangle in individual:
                self.assertIn((triangle.r, triangle.g, triangle.b), self.target_colors)
                self.assertTrue(0 <= triangle.x1 < 2)
                self.assertTrue(0 <= triangle.y1 < 2)
                self.assertTrue(0 <= triangle.x2 < 2)
                self.assertTrue(0 <= triangle.y2 < 2)
                self.assertTrue(0 <= triangle.x3 < 2)
                self.assertTrue(0 <= triangle.y3 < 2)
                self.assertEqual(33, triangle.a)

    def test_unseeded_triangle_uses_uniform_rgb_sampling(self) -> None:
        """Unseeded RGB values still come from independent 0-255 samples."""

        with patch(
            "src.population.np.random.randint",
            side_effect=[0, 0, 1, 1, 0, 1, 11, 12, 13, 44],
        ) as randint:
            triangle = population.create_random_triangle(
                image_width=2,
                image_height=2,
                triangle_alpha_range=(44, 44),
                target=self.target,
                seeded=False,
            )

        self.assertEqual((11, 12, 13), (triangle.r, triangle.g, triangle.b))
        self.assertEqual(44, triangle.a)
        self.assertEqual(
            [call(0, 256), call(0, 256), call(0, 256)],
            randint.mock_calls[6:9],
        )

    def test_seeded_triangle_rejects_invalid_targets(self) -> None:
        """Seeded creation requires a matching RGB NumPy target array."""

        invalid_targets = [
            None,
            [[[10, 20, 30]]],
            np.array([[10, 20], [30, 40]], dtype=np.uint8),
            np.zeros((3, 2, 3), dtype=np.uint8),
        ]

        for invalid_target in invalid_targets:
            with self.subTest(target=invalid_target):
                with self.assertRaises(ValueError):
                    population.create_random_triangle(
                        image_width=2,
                        image_height=2,
                        target=invalid_target,
                        seeded=True,
                    )


class GeneticAlgorithmSeededInitializationTests(TestCase):
    """Covers GA initialization paths that create fresh triangles."""

    def setUp(self) -> None:
        """Sets deterministic random state and a small RGB target."""

        np.random.seed(11)
        self.target = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        self.target_colors = {
            tuple(int(value) for value in pixel)
            for row in self.target
            for pixel in row
        }

    def test_ga_seeded_initializes_full_generation_zero(self) -> None:
        """A fresh generation-0 population uses target RGB when seeded."""

        ga = algorithm.GeneticAlgorithm(
            target=self.target,
            fitness_function=_zero_fitness,
            population_size=3,
            generations=1,
            n_triangles=4,
            triangle_alpha_range=(77, 77),
            seeded=True,
        )

        ga.initialize()

        self.assertEqual(3, len(ga.population))
        for individual in ga.population:
            self.assertEqual(4, len(individual))
            for triangle in individual:
                self.assertIn((triangle.r, triangle.g, triangle.b), self.target_colors)
                self.assertEqual(77, triangle.a)

    def test_ga_seeded_backfills_short_initial_population(self) -> None:
        """Short explicit generation-0 populations are target-seeded on backfill."""

        initial_triangle = _triangle_with_color((250, 251, 252))
        ga = algorithm.GeneticAlgorithm(
            target=self.target,
            fitness_function=_zero_fitness,
            population_size=3,
            generations=1,
            n_triangles=1,
            initial_population=[[initial_triangle]],
            triangle_alpha_range=(88, 88),
            seeded=True,
        )

        ga.initialize()

        self.assertEqual((250, 251, 252), _triangle_color(ga.population[0][0]))
        for individual in ga.population[1:]:
            self.assertIn(_triangle_color(individual[0]), self.target_colors)
            self.assertEqual(88, individual[0].a)

    def test_random_immigrants_remain_unseeded_when_ga_is_seeded(self) -> None:
        """Diversity injection does not receive target-backed seeded arguments."""

        ga = algorithm.GeneticAlgorithm(
            target=self.target,
            fitness_function=_zero_fitness,
            population_size=4,
            generations=1,
            n_triangles=1,
            random_immigrants=1,
            seeded=True,
        )
        immigrant_population = [[_triangle_with_color((200, 201, 202))]]

        with patch.object(
            algorithm.population,
            "create_population",
            return_value=immigrant_population,
        ) as create_population:
            next_population: list[algorithm.Individual] = []
            immigrant_count = ga._inject_random_immigrants(next_population)

        self.assertEqual(1, immigrant_count)
        self.assertEqual(immigrant_population, next_population)
        call_kwargs = create_population.call_args.kwargs
        self.assertNotIn("target", call_kwargs)
        self.assertNotIn("seeded", call_kwargs)

    def test_workflow_expansion_seeds_only_new_triangles(self) -> None:
        """Staged expansion preserves existing triangles and seeds additions."""

        base_triangle = _triangle_with_color((250, 251, 252))

        expanded = workflow.expand_individual_to_triangle_count(
            individual=[base_triangle],
            n_triangles=3,
            image_width=2,
            image_height=2,
            triangle_alpha_range=(99, 99),
            target=self.target,
            seeded=True,
        )

        self.assertEqual((250, 251, 252), _triangle_color(expanded[0]))
        for triangle in expanded[1:]:
            self.assertIn(_triangle_color(triangle), self.target_colors)
            self.assertEqual(99, triangle.a)


def _triangle_color(triangle: population.Triangle) -> tuple[int, int, int]:
    """Returns a triangle's RGB tuple."""

    return triangle.r, triangle.g, triangle.b
