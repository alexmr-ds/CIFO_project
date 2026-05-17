"""Grid-search and random-search helpers for GA notebook experiments."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import cross_over, diversity, fitness, mutate
from .diversity import FitnessSharingRestrictedMatingGA


DEFAULT_CROSSOVER_FUNCTIONS = {
    "two_point_one_child": cross_over.two_point_crossover,
    "two_point_two_children": cross_over.two_point_crossover_two_children,
    "pmx": cross_over.pmx_crossover,
}

# Columns returned by load_all_results() and consumed by build_summary().
RESULT_COLUMNS = [
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


# ---------------------------------------------------------------------------
# Search-space builders
# ---------------------------------------------------------------------------

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


def build_random_setups(
    search_space: Mapping[str, list[Any]],
    n_samples: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample n_samples random configurations from the search space.

    Each configuration is drawn independently and uniformly from the
    per-parameter value lists, so the same combination can appear more
    than once with a large search space or a small n_samples.
    """
    rng = np.random.default_rng(seed)
    rows = [
        {key: rng.choice(values) for key, values in search_space.items()}
        for _ in range(n_samples)
    ]
    df = pd.DataFrame(rows).reset_index(drop=True)
    df.insert(0, "setup_id", np.arange(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# JSON caching — one file per trial, keyed on params
# ---------------------------------------------------------------------------

def _pipeline_name(setup: Mapping[str, Any]) -> str:
    """Human-readable pipeline label derived from the varying params."""
    return (
        f"search"
        f"-{setup['crossover_type']}"
        f"-{setup['selection_type']}"
        f"-{setup['restricted_mating']}"
        f"-mut{setup['mutation_rate']}"
        f"-xo{setup['crossover_rate']}"
    )


def _trial_params(
    setup: Mapping[str, Any],
    fixed_params: Mapping[str, Any],
    trial_id: int,
) -> dict:
    """Full parameter dict embedded in every saved JSON (used for cache matching)."""
    return {
        "trial_id":         int(trial_id),
        "mutation_rate":    float(setup["mutation_rate"]),
        "crossover_rate":   float(setup["crossover_rate"]),
        "crossover_type":   str(setup["crossover_type"]),
        "selection_type":   str(setup["selection_type"]),
        "restricted_mating": str(setup["restricted_mating"]),
        "population_size":  int(fixed_params["population_size"]),
        "generations":      int(fixed_params["generations"]),
        "elitism":          int(fixed_params["elitism"]),
        "sigma_share":      float(fixed_params["sigma_share"]),
        "n_bins":           int(fixed_params["n_bins"]),
        "candidate_pool":   int(fixed_params["candidate_pool"]),
        "triangle_alpha_range": list(fixed_params["triangle_alpha_range"]),
    }


def _params_match(run: dict, params: dict) -> bool:
    """True if the JSON run's stored parameters match the given params dict."""
    p = run.get("parameters", {})
    return all(p.get(k) == v for k, v in params.items())


def _deterministic_seed(params: dict) -> int:
    """Deterministic seed derived from params — same config always gets same seed."""
    key = json.dumps(params, sort_keys=True)
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 31)


def _save_trial(
    results_dir: Path,
    pipeline: str,
    params: dict,
    result: dict,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    filename = results_dir / f"{pipeline}_{timestamp}.json"
    payload = {
        "pipeline":        pipeline,
        "timestamp":       datetime.now().isoformat(),
        "parameters":      params,
        "results":         result,
    }
    filename.write_text(json.dumps(payload, indent=2))


def _load_cached_trial(
    results_dir: Path,
    params: dict,
) -> dict | None:
    """Return cached result dict if a matching JSON exists, else None."""
    if not results_dir.exists():
        return None
    for path in sorted(results_dir.glob("*.json")):
        try:
            run = json.loads(path.read_text())
            if _params_match(run, params):
                return run["results"]
        except Exception:
            pass
    return None


def load_all_results(results_dir: Path | str) -> pd.DataFrame:
    """Load all cached trial results from a directory into a flat DataFrame."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            run = json.loads(path.read_text())
            p = run.get("parameters", {})
            r = run.get("results", {})
            rows.append({
                "trial_id":         p.get("trial_id"),
                "seed":             r.get("seed"),
                "mutation_rate":    p.get("mutation_rate"),
                "crossover_rate":   p.get("crossover_rate"),
                "crossover_type":   p.get("crossover_type"),
                "selection_type":   p.get("selection_type"),
                "restricted_mating": p.get("restricted_mating"),
                "final_best_fitness": r.get("final_best_fitness"),
                "best_generation":  r.get("best_generation"),
                "generations_run":  r.get("generations_run"),
                "stopped_early":    r.get("stopped_early"),
                "runtime_seconds":  r.get("runtime_seconds"),
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def first_best_generation(history: list[float], final_best_fitness: float) -> int:
    """Return the 1-based first generation where the final best appears."""
    values = np.asarray(history, dtype=float)
    matches = np.where(np.isclose(values, final_best_fitness, rtol=0.0, atol=1e-12))[0]
    if len(matches) > 0:
        return int(matches[0] + 1)
    return int(np.argmin(values) + 1)


def build_summary(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial results by unique parameter combination."""
    grouping_columns = [
        "mutation_rate",
        "crossover_rate",
        "crossover_type",
        "selection_type",
        "restricted_mating",
    ]
    summary_columns = grouping_columns + [
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


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_one_trial(
    setup: Mapping[str, Any] | pd.Series,
    trial_id: int,
    target_array: np.ndarray,
    fixed_params: Mapping[str, Any],
    results_dir: Path | str,
    fitness_function: Any | None = None,
    mutation_function: Any | None = None,
    crossover_functions: Mapping[str, Any] | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.01,
) -> dict[str, Any]:
    """Run one GA trial, loading from JSON cache if an identical run exists.

    Cache hit: a JSON file in results_dir whose stored parameters match
    setup + fixed_params + trial_id exactly. On a hit the GA is skipped and
    the saved result is returned immediately.
    """
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be greater than zero.")

    fitness_function    = fitness_function    or fitness.compute_rmse
    mutation_function   = mutation_function   or mutate.random_triangle_mutation
    crossover_functions = crossover_functions or DEFAULT_CROSSOVER_FUNCTIONS
    results_dir         = Path(results_dir)

    pipeline = _pipeline_name(setup)
    params   = _trial_params(setup, fixed_params, trial_id)

    cached = _load_cached_trial(results_dir, params)
    if cached is not None:
        print(f"  ↩ [{pipeline}] trial {trial_id} loaded from cache")
        return cached

    seed = _deterministic_seed(params)
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
    )

    started = time.perf_counter()
    final_best_fitness, history = ga.run(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
    )
    runtime_seconds = time.perf_counter() - started
    generations_run = len(history)

    result = {
        "seed":               seed,
        "final_best_fitness": float(final_best_fitness),
        "best_generation":    first_best_generation(history, final_best_fitness),
        "generations_run":    generations_run,
        "stopped_early":      generations_run < int(fixed_params["generations"]),
        "runtime_seconds":    float(runtime_seconds),
    }

    _save_trial(results_dir, pipeline, params, result)
    return result
