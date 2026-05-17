"""Tests grid-search trial helpers and early stopping behavior."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ga import grid_search


class _ScriptedEarlyStoppingGA(grid_search.FitnessSharingRestrictedMatingGA):
    """Runs a fixed global-best sequence through the early-stopping loop."""

    def __init__(self, scripted_fitness: list[float]) -> None:
        """Stores the deterministic fitness sequence for the test run."""

        self.scripted_fitness = scripted_fitness
        self.generations = len(scripted_fitness)
        self.evaluation_backend = "sequential"
        self.n_jobs = None
        self.chunksize = None
        self.target = np.zeros((1, 1, 3), dtype=np.uint8)
        self.fitness_function = None
        self.image_width = 1
        self.image_height = 1
        self.population_size = 1
        self.elitism = 1
        self.logs = False
        self.progress = False
        self._emitted_logs = []

    def initialize(self) -> None:
        """Resets the minimal state used by run_with_early_stopping."""

        self._cursor = 0
        self.population = [["only-individual"]]
        self.best_individual = None
        self.best_fitness = float("inf")
        self.history = []
        self.run_logs = {}
        self._current_mutation_rate = 0.1
        self._last_improvement_generation = 0

    def _evaluate_population(self, executor=None) -> list[float]:
        """Feeds the next scripted fitness value into global-best tracking."""

        fitness_value = self.scripted_fitness[self._cursor]
        self._cursor += 1
        if fitness_value < self.best_fitness:
            self.best_fitness = fitness_value
            self.best_individual = ["best-individual"]
        return [fitness_value]

    def _update_mutation_rate(self, generation: int) -> None:
        """Keeps adaptive mutation out of the deterministic test path."""

    def _inject_random_immigrants(self, next_population: list) -> int:
        """Disables immigrants for the deterministic test path."""

        return 0

    def _emit_progress(self, generation_log) -> None:
        """Records progress logs without printing."""

        self._emitted_logs.append(generation_log)


class GridSearchTests(TestCase):
    """Covers grid-search result shape, summaries, and early stopping."""

    def setUp(self) -> None:
        """Creates shared minimal grid-search inputs."""

        self.setup = {
            "setup_id": 1,
            "crossover_type": "two_point_one_child",
            "selection_type": "tournament",
            "restricted_mating": "best_partial_match",
        }
        self.fixed_params = {
            "elitism": 1,
            "population_size": 2,
            "generations": 5,
            "triangle_alpha_range": (255, 255),
            "mutation_rate": 0.1,
            "crossover_rate": 0.9,
            "sigma_share": 0.3,
            "n_bins": 8,
            "candidate_pool": 5,
        }
        self.target = np.zeros((1, 1, 3), dtype=np.uint8)

    def test_run_one_trial_records_full_budget_without_early_stopping(self) -> None:
        """A normal trial records full generation count and does not stop early."""

        with patch("src.ga.grid_search.FitnessSharingRestrictedMatingGA") as ga_class:
            mock_ga = ga_class.return_value
            mock_ga.run.return_value = (0.2, [0.5, 0.3, 0.2, 0.2, 0.2])

            result = grid_search.run_one_trial(
                setup=self.setup,
                trial_id=1,
                target_array=self.target,
                fixed_params=self.fixed_params,
            )

        mock_ga.run.assert_called_once_with()
        mock_ga.run_with_early_stopping.assert_not_called()
        constructor_kwargs = ga_class.call_args.kwargs
        self.assertEqual("sequential", constructor_kwargs["evaluation_backend"])
        self.assertIsNone(constructor_kwargs["n_jobs"])
        self.assertIsNone(constructor_kwargs["chunksize"])
        self.assertEqual(0.1, constructor_kwargs["mutation_rate"])
        self.assertEqual(0.9, constructor_kwargs["crossover_rate"])
        self.assertEqual(174, result["seed"])
        self.assertEqual("sequential", result["evaluation_backend"])
        self.assertIsNone(result["n_jobs"])
        self.assertIsNone(result["chunksize"])
        self.assertEqual(0.1, result["mutation_rate"])
        self.assertEqual(0.9, result["crossover_rate"])
        self.assertEqual(5, result["generations_run"])
        self.assertFalse(result["stopped_early"])
        self.assertEqual(3, result["best_generation"])

    def test_run_one_trial_forwards_configured_evaluation_backend(self) -> None:
        """Configured backend settings are passed through to the GA."""

        fixed_params = {
            **self.fixed_params,
            "evaluation_backend": "process",
            "n_jobs": 4,
            "chunksize": 2,
        }
        with patch("src.ga.grid_search.FitnessSharingRestrictedMatingGA") as ga_class:
            mock_ga = ga_class.return_value
            mock_ga.run.return_value = (0.2, [0.5, 0.2, 0.2, 0.2, 0.2])

            result = grid_search.run_one_trial(
                setup=self.setup,
                trial_id=1,
                target_array=self.target,
                fixed_params=fixed_params,
            )

        constructor_kwargs = ga_class.call_args.kwargs
        self.assertEqual("process", constructor_kwargs["evaluation_backend"])
        self.assertEqual(4, constructor_kwargs["n_jobs"])
        self.assertEqual(2, constructor_kwargs["chunksize"])
        self.assertEqual("process", result["evaluation_backend"])
        self.assertEqual(4, result["n_jobs"])
        self.assertEqual(2, result["chunksize"])

    def test_run_one_trial_falls_back_to_legacy_setup_rates(self) -> None:
        """Legacy setup-provided rates still work when fixed params omit them."""

        setup = {
            **self.setup,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
        }
        fixed_params = {
            key: value
            for key, value in self.fixed_params.items()
            if key not in {"mutation_rate", "crossover_rate"}
        }
        with patch("src.ga.grid_search.FitnessSharingRestrictedMatingGA") as ga_class:
            mock_ga = ga_class.return_value
            mock_ga.run.return_value = (0.2, [0.5, 0.2, 0.2, 0.2, 0.2])

            result = grid_search.run_one_trial(
                setup=setup,
                trial_id=1,
                target_array=self.target,
                fixed_params=fixed_params,
            )

        constructor_kwargs = ga_class.call_args.kwargs
        self.assertEqual(0.2, constructor_kwargs["mutation_rate"])
        self.assertEqual(0.8, constructor_kwargs["crossover_rate"])
        self.assertEqual(0.2, result["mutation_rate"])
        self.assertEqual(0.8, result["crossover_rate"])

    def test_run_one_trial_uses_early_stopping_when_patience_is_set(self) -> None:
        """An early-stopped trial records short generation count and flag."""

        with patch("src.ga.grid_search.FitnessSharingRestrictedMatingGA") as ga_class:
            mock_ga = ga_class.return_value
            mock_ga.run_with_early_stopping.return_value = (0.2, [0.5, 0.2, 0.2])

            result = grid_search.run_one_trial(
                setup=self.setup,
                trial_id=1,
                target_array=self.target,
                fixed_params=self.fixed_params,
                early_stopping_patience=5,
            )

        mock_ga.run.assert_not_called()
        mock_ga.run_with_early_stopping.assert_called_once_with(5)
        self.assertEqual(3, result["generations_run"])
        self.assertTrue(result["stopped_early"])

    def test_early_stopping_patience_must_be_positive(self) -> None:
        """Invalid patience values fail before constructing a GA."""

        with self.assertRaisesRegex(
            ValueError,
            "early_stopping_patience must be greater than zero.",
        ):
            grid_search.run_one_trial(
                setup=self.setup,
                trial_id=1,
                target_array=self.target,
                fixed_params=self.fixed_params,
                early_stopping_patience=0,
            )

    def test_run_one_trial_requires_rates_from_fixed_params_or_setup(self) -> None:
        """Missing rates from both supported sources fail clearly."""

        fixed_params = {
            key: value
            for key, value in self.fixed_params.items()
            if key not in {"mutation_rate", "crossover_rate"}
        }

        with self.assertRaisesRegex(
            ValueError,
            "mutation_rate must be provided in fixed_params or the grid setup.",
        ):
            grid_search.run_one_trial(
                setup=self.setup,
                trial_id=1,
                target_array=self.target,
                fixed_params=fixed_params,
            )

    def test_run_with_early_stopping_uses_strict_improvement_patience(self) -> None:
        """Patience resets only on strictly lower global-best fitness."""

        ga = _ScriptedEarlyStoppingGA([5.0, 5.0, 4.5, 4.5, 4.5, 4.0])

        best_fitness, history = ga.run_with_early_stopping(patience=2)

        self.assertEqual(4.5, best_fitness)
        self.assertEqual([5.0, 5.0, 4.5, 4.5, 4.5], history)

    def test_build_summary_includes_early_stop_metrics(self) -> None:
        """Summary aggregation reports mean generations and stopped trials."""

        raw_results = pd.DataFrame(
            [
                {
                    "setup_id": 1,
                    "trial_id": 1,
                    "seed": 174,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.9,
                    "crossover_type": "two_point_one_child",
                    "selection_type": "tournament",
                    "restricted_mating": "best_partial_match",
                    "final_best_fitness": 0.3,
                    "best_generation": 2,
                    "generations_run": 3,
                    "stopped_early": True,
                    "runtime_seconds": 1.0,
                },
                {
                    "setup_id": 1,
                    "trial_id": 2,
                    "seed": 175,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.9,
                    "crossover_type": "two_point_one_child",
                    "selection_type": "tournament",
                    "restricted_mating": "best_partial_match",
                    "final_best_fitness": 0.2,
                    "best_generation": 5,
                    "generations_run": 5,
                    "stopped_early": False,
                    "runtime_seconds": 2.0,
                },
            ]
        )

        summary = grid_search.build_summary(raw_results)

        self.assertEqual("sequential", summary.loc[0, "evaluation_backend"])
        self.assertEqual(4.0, summary.loc[0, "mean_generations_run"])
        self.assertEqual(1, summary.loc[0, "stopped_early_trials"])
        self.assertEqual(2, summary.loc[0, "completed_trials"])

    def test_build_summary_groups_backend_settings_separately(self) -> None:
        """Summary rows keep backend configurations distinct."""

        raw_results = pd.DataFrame(
            [
                {
                    "setup_id": 1,
                    "trial_id": 1,
                    "seed": 174,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.9,
                    "crossover_type": "two_point_one_child",
                    "selection_type": "tournament",
                    "restricted_mating": "best_partial_match",
                    "evaluation_backend": "sequential",
                    "n_jobs": None,
                    "chunksize": None,
                    "final_best_fitness": 0.3,
                    "best_generation": 2,
                    "generations_run": 3,
                    "stopped_early": True,
                    "runtime_seconds": 1.0,
                },
                {
                    "setup_id": 1,
                    "trial_id": 2,
                    "seed": 175,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.9,
                    "crossover_type": "two_point_one_child",
                    "selection_type": "tournament",
                    "restricted_mating": "best_partial_match",
                    "evaluation_backend": "process",
                    "n_jobs": 4,
                    "chunksize": 2,
                    "final_best_fitness": 0.2,
                    "best_generation": 5,
                    "generations_run": 5,
                    "stopped_early": False,
                    "runtime_seconds": 2.0,
                },
            ]
        )

        summary = grid_search.build_summary(raw_results).sort_values(
            "evaluation_backend"
        )

        self.assertEqual(
            ["process", "sequential"],
            summary["evaluation_backend"].tolist(),
        )
        self.assertEqual(4.0, summary.iloc[0]["n_jobs"])
        self.assertEqual(2.0, summary.iloc[0]["chunksize"])
        self.assertTrue(pd.isna(summary.iloc[1]["n_jobs"]))
        self.assertTrue(pd.isna(summary.iloc[1]["chunksize"]))
