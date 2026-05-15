"""Grid-search helpers for GA notebook experiments."""

from __future__ import annotations

import itertools
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import cross_over, diversity, fitness, mutate


class FitnessSharingRestrictedMatingGA(
    diversity.FitnessSharingGA,
    diversity.RestrictedMatingGA,
):
    """GA combining fitness sharing and restricted mating."""

    pass


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
        "mean_runtime_seconds",
        "completed_trials",
    ]
    if raw_results.empty:
        return pd.DataFrame(columns=summary_columns)

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
            mean_runtime_seconds=("runtime_seconds", "mean"),
            completed_trials=("trial_id", "count"),
        )
        .sort_values("mean_final_best_fitness", ascending=True)
        .reset_index(drop=True)
    )
    summary["std_final_best_fitness"] = summary["std_final_best_fitness"].fillna(0.0)
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
) -> dict[str, Any]:
    """Run one deterministic grid-search GA trial."""

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
    final_best_fitness, history = ga.run()
    runtime_seconds = time.perf_counter() - started

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
        "runtime_seconds": float(runtime_seconds),
    }
