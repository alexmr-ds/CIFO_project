"""Grid-search helpers for GA notebook experiments."""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import cross_over, diversity, evaluation, fitness, logs, mutate


class FitnessSharingRestrictedMatingGA(
    diversity.FitnessSharingGA,
    diversity.RestrictedMatingGA,
):
    """GA combining fitness sharing and restricted mating."""

    def run_with_early_stopping(self, patience: int) -> tuple[float, list[float]]:
        """Run the GA until the global best stalls for the configured patience."""

        if patience <= 0:
            raise ValueError("patience must be greater than zero.")

        self.initialize()

        executor: evaluation.Executor | None = None
        generation_logs: list[logs.GenerationLog] = []
        stale_generations = 0

        try:
            if self.evaluation_backend != "sequential":
                executor = evaluation.create_evaluation_executor(
                    self.evaluation_backend,
                    self.n_jobs,
                    self.target,
                    self.fitness_function,
                    self.image_width,
                    self.image_height,
                )

            for generation in range(self.generations):
                previous_best = float(self.best_fitness)

                evaluation_started = time.perf_counter()
                fitness_values = self._evaluate_population(executor)
                evaluation_duration_seconds = (
                    time.perf_counter() - evaluation_started
                )

                ranked_indices = np.argsort(fitness_values)
                generation_best_fitness = float(fitness_values[int(ranked_indices[0])])
                global_best_fitness = float(self.best_fitness)
                improved = global_best_fitness < previous_best

                if improved:
                    self._last_improvement_generation = generation
                    stale_generations = 0
                else:
                    stale_generations += 1

                self._update_mutation_rate(generation)
                self.history.append(global_best_fitness)

                offspring_created = 0
                mutated_offspring = 0
                mutated_triangles = 0
                immigrant_count = 0
                should_stop = stale_generations >= patience

                if generation != self.generations - 1 and not should_stop:
                    next_population = [
                        copy.deepcopy(self.population[int(index)])
                        for index in ranked_indices[: self.elitism]
                    ]

                    immigrant_count = self._inject_random_immigrants(next_population)

                    while len(next_population) < self.population_size:
                        parent1, parent2 = self.select_parents(fitness_values)
                        children = self.crossover(parent1, parent2)

                        for child in children:
                            if len(next_population) >= self.population_size:
                                break
                            child, changed_triangles = self.mutate(child)
                            offspring_created += 1
                            mutated_triangles += changed_triangles
                            if changed_triangles > 0:
                                mutated_offspring += 1
                            next_population.append(child)

                    self.population = next_population[: self.population_size]

                generation_log = logs.create_generation_log(
                    generation=generation,
                    generation_best_fitness=generation_best_fitness,
                    generation_mean_fitness=float(np.mean(fitness_values)),
                    global_best_fitness=global_best_fitness,
                    evaluation_backend=self.evaluation_backend,
                    n_jobs=self.n_jobs,
                    chunksize=self.chunksize,
                    evaluation_duration_seconds=evaluation_duration_seconds,
                    mutation_rate_used=float(self._current_mutation_rate),
                    offspring_created=offspring_created,
                    mutated_offspring=mutated_offspring,
                    mutated_triangles=mutated_triangles,
                    immigrant_count=immigrant_count,
                )

                if self.logs:
                    generation_logs.append(generation_log)

                self._emit_progress(generation_log)

                if should_stop:
                    break

        finally:
            if executor is not None:
                executor.shutdown()

        if self.best_individual is None:
            raise RuntimeError(
                "The genetic algorithm did not produce a best individual."
            )

        if self.logs:
            self.run_logs = logs.create_run_logs(
                generation_logs,
                float(self.best_fitness),
                self.best_individual,
            )

        return float(self.best_fitness), list(self.history)


RESULT_COLUMNS = [
    "setup_id",
    "trial_id",
    "seed",
    "mutation_rate",
    "crossover_rate",
    "crossover_type",
    "selection_type",
    "restricted_mating",
    "final_best_fitness",
    "best_generation",
    "generations_run",
    "stopped_early",
    "runtime_seconds",
]

DEFAULT_CROSSOVER_FUNCTIONS = {
    "two_point_one_child": cross_over.two_point_crossover,
    "two_point_two_children": cross_over.two_point_crossover_two_children,
    "pmx": cross_over.pmx_crossover,
}


def build_full_search_space(search_space: Mapping[str, list[Any]]) -> pd.DataFrame:
    """Build the Cartesian product of a hyperparameter search space."""

    keys = list(search_space)
    values = [search_space[key] for key in keys]
    rows = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return pd.DataFrame(rows)


def build_grid_setups(search_space: Mapping[str, list[Any]]) -> pd.DataFrame:
    """Build all grid-search setups with 1-based setup IDs."""

    grid_setups = build_full_search_space(search_space).reset_index(drop=True)
    grid_setups.insert(0, "setup_id", np.arange(1, len(grid_setups) + 1))
    return grid_setups


def first_best_generation(history: list[float], final_best_fitness: float) -> int:
    """Return the 1-based first generation where the final best appears."""

    values = np.asarray(history, dtype=float)
    matches = np.where(np.isclose(values, final_best_fitness, rtol=0.0, atol=1e-12))[0]
    if len(matches) > 0:
        return int(matches[0] + 1)
    return int(np.argmin(values) + 1)


def load_raw_results(raw_results_path: Path | str) -> pd.DataFrame:
    """Load raw grid-search results, returning an empty table if missing."""

    raw_results_path = Path(raw_results_path)
    if not raw_results_path.exists() or raw_results_path.stat().st_size == 0:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    df = pd.read_csv(raw_results_path)
    df = df.reindex(columns=RESULT_COLUMNS)
    df = df.drop_duplicates(subset=["setup_id", "trial_id"], keep="last")
    return df.sort_values(["setup_id", "trial_id"]).reset_index(drop=True)


def save_raw_results(df: pd.DataFrame, raw_results_path: Path | str) -> None:
    """Persist raw grid-search results in deterministic order."""

    raw_results_path = Path(raw_results_path)
    raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = df.reindex(columns=RESULT_COLUMNS)
    ordered = ordered.sort_values(["setup_id", "trial_id"]).reset_index(drop=True)
    ordered.to_csv(raw_results_path, index=False)


def build_summary(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate grid-search trial results by setup."""

    summary_columns = [
        "setup_id",
        "mutation_rate",
        "crossover_rate",
        "crossover_type",
        "selection_type",
        "restricted_mating",
        "mean_final_best_fitness",
        "std_final_best_fitness",
        "min_final_best_fitness",
        "max_final_best_fitness",
        "mean_generations_run",
        "stopped_early_trials",
        "mean_runtime_seconds",
        "completed_trials",
    ]
    if raw_results.empty:
        return pd.DataFrame(columns=summary_columns)

    raw_results = raw_results.reindex(columns=RESULT_COLUMNS)

    grouping_columns = [
        "setup_id",
        "mutation_rate",
        "crossover_rate",
        "crossover_type",
        "selection_type",
        "restricted_mating",
    ]
    summary = (
        raw_results.groupby(grouping_columns, as_index=False)
        .agg(
            mean_final_best_fitness=("final_best_fitness", "mean"),
            std_final_best_fitness=("final_best_fitness", "std"),
            min_final_best_fitness=("final_best_fitness", "min"),
            max_final_best_fitness=("final_best_fitness", "max"),
            mean_generations_run=("generations_run", "mean"),
            stopped_early_trials=("stopped_early", "sum"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            completed_trials=("trial_id", "count"),
        )
        .sort_values("mean_final_best_fitness", ascending=True)
        .reset_index(drop=True)
    )
    summary["std_final_best_fitness"] = summary["std_final_best_fitness"].fillna(0.0)
    summary["stopped_early_trials"] = (
        summary["stopped_early_trials"].fillna(0).astype(int)
    )
    return summary.reindex(columns=summary_columns)


def run_one_trial(
    setup: Mapping[str, Any] | pd.Series,
    trial_id: int,
    target_array: np.ndarray,
    fixed_params: Mapping[str, Any],
    base_seed: int = 73,
    fitness_function: Any | None = None,
    mutation_function: Any | None = None,
    crossover_functions: Mapping[str, Any] | None = None,
    early_stopping_patience: int | None = None,
) -> dict[str, Any]:
    """Run one deterministic grid-search GA trial."""

    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be greater than zero.")

    fitness_function = fitness_function or fitness.compute_rmse
    mutation_function = mutation_function or mutate.random_triangle_mutation
    crossover_functions = crossover_functions or DEFAULT_CROSSOVER_FUNCTIONS

    setup_id = int(setup["setup_id"])
    seed = base_seed + setup_id * 100 + int(trial_id)
    np.random.seed(seed)

    ga = FitnessSharingRestrictedMatingGA(
        target=target_array,
        fitness_function=fitness_function,
        population_size=fixed_params["population_size"],
        generations=fixed_params["generations"],
        crossover_function=crossover_functions[setup["crossover_type"]],
        crossover_rate=float(setup["crossover_rate"]),
        mutation_function=mutation_function,
        mutation_rate=float(setup["mutation_rate"]),
        elitism=fixed_params["elitism"],
        selection_type=setup["selection_type"],
        triangle_alpha_range=fixed_params["triangle_alpha_range"],
        sigma_share=fixed_params["sigma_share"],
        n_bins=fixed_params["n_bins"],
        mating_type=setup["restricted_mating"],
        candidate_pool=fixed_params["candidate_pool"],
        evaluation_backend="sequential",
        progress=False,
    )

    started = time.perf_counter()
    if early_stopping_patience is None:
        final_best_fitness, history = ga.run()
    else:
        final_best_fitness, history = ga.run_with_early_stopping(
            early_stopping_patience,
        )
    runtime_seconds = time.perf_counter() - started
    generations_run = len(history)

    return {
        "setup_id": setup_id,
        "trial_id": int(trial_id),
        "seed": seed,
        "mutation_rate": float(setup["mutation_rate"]),
        "crossover_rate": float(setup["crossover_rate"]),
        "crossover_type": setup["crossover_type"],
        "selection_type": setup["selection_type"],
        "restricted_mating": setup["restricted_mating"],
        "final_best_fitness": float(final_best_fitness),
        "best_generation": first_best_generation(history, final_best_fitness),
        "generations_run": generations_run,
        "stopped_early": generations_run < int(fixed_params["generations"]),
        "runtime_seconds": float(runtime_seconds),
    }
