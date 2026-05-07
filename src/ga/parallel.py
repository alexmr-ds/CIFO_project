"""
Parallel GA trial runner — the main entry point for notebook experiments.

Why this module exists
----------------------
Running a single GA trial is straightforward, but for reliable results we
need to repeat each experiment multiple times (because GAs are stochastic)
and compare results across different parameter values (grid search).
Doing that naively in a notebook produces a lot of boilerplate and is slow.

This module solves both problems:
  1. ``run_trials``       — runs N independent trials of one config in parallel
                            and returns aggregated mean/std statistics.
  2. ``run_grid_search``  — runs N trials for each value in a parameter grid,
                            submitting all jobs to one pool so every CPU core
                            stays busy for the full duration.

Both functions check a file-based cache before running: if matching results
already exist on disk from a previous run, they are loaded instantly instead
of re-computing, saving significant time when iterating on analysis cells.

Parallelism design
------------------
macOS uses the "spawn" start method for multiprocessing, which means each
worker process is a fresh Python interpreter.  Functions passed to workers
must be picklable — module-level functions are always picklable, but lambdas
and functions defined in notebook cells are not.

To avoid this:
  - ``GAConfig`` is a dataclass with only serialisable fields.
  - ``run_single_ga`` is a module-level function.
  - Inside each worker, the GA uses ``evaluation_backend="sequential"``
    to prevent the worker from spawning its own child processes (nested pools
    cause deadlocks or resource exhaustion on macOS).

The outer ``ProcessPoolExecutor`` in ``run_trials`` / ``run_grid_search``
provides the parallelism, with one process per trial.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import population
from .results import save_run

Individual = list[population.Triangle]


# ---------------------------------------------------------------------------
# GAConfig — fully self-contained configuration for one GA trial
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """
    All parameters needed to run one GA trial, stored in a serialisable dataclass.

    Using a dataclass instead of keyword arguments makes it easy to override
    a single parameter for a grid search::

        config = GAConfig(**{**base_config.__dict__, "elitism": 3})

    All fields except ``trial`` and ``label`` directly correspond to
    GeneticAlgorithm constructor parameters.

    Attributes:
        target:               RGB target image array (H, W, 3).
        fitness_function:     Module-level fitness callable (must be picklable).
        population_size:      Number of individuals per generation.
        generations:          Number of generations to evolve.
        crossover_function:   Module-level crossover callable.
        crossover_rate:       Crossover probability per parent pair (0–1).
        mutation_function:    Module-level mutation callable.
        mutation_rate:        Per-triangle mutation probability (0–1).
        elitism:              Best N individuals copied unchanged each generation.
        selection_type:       "tournament", "ranking", or "roulette".
        triangle_alpha_range: Inclusive (min, max) alpha range for triangles.
        trial:                0-based trial index (set automatically by run_trials).
        label:                Pipeline name injected into log messages (set automatically).
    """

    target: np.ndarray
    fitness_function: Any
    population_size: int
    generations: int
    crossover_function: Any
    crossover_rate: float
    mutation_function: Any
    mutation_rate: float
    elitism: int
    selection_type: str
    triangle_alpha_range: tuple[int, int]
    trial: int = 0    # set automatically — do not set manually
    label: str = ""   # set automatically — do not set manually


# ---------------------------------------------------------------------------
# TrialSummary — aggregated statistics returned to notebook cells
# ---------------------------------------------------------------------------

@dataclass
class TrialSummary:
    """
    Aggregated statistics from N independent GA trials of the same config.

    Attributes:
        pipeline:        Pipeline label (e.g. "Baseline-GA", "Elitism-3").
        n_trials:        Number of trials that were averaged.
        mean_fitness:    Mean final RMSE across all trials.
        std_fitness:     Standard deviation of final RMSE across trials.
        min_fitness:     Best (lowest) RMSE seen across all trials.
        max_fitness:     Worst (highest) RMSE seen across all trials.
        mean_history:    Per-generation mean RMSE averaged across trials.
        std_history:     Per-generation std of RMSE across trials.
        best_individual: The triangle individual that achieved min_fitness.
        all_fitness:     Raw list of final RMSE values, one per trial.
        total_runtime:   Total wall-clock time for all trials combined.
    """

    pipeline: str
    n_trials: int
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    mean_history: list[float]
    std_history: np.ndarray
    best_individual: Individual
    all_fitness: list[float]
    total_runtime: float

    def __str__(self) -> str:
        return (
            f"{self.pipeline} — {self.n_trials} trials\n"
            f"  Mean RMSE : {self.mean_fitness:.6f} ± {self.std_fitness:.6f}\n"
            f"  Best      : {self.min_fitness:.6f}  |  Worst: {self.max_fitness:.6f}\n"
            f"  Runtime   : {self.total_runtime:.0f}s total"
        )


# ---------------------------------------------------------------------------
# Cache helpers — check disk before running expensive trials
# ---------------------------------------------------------------------------

def _params_match(config: GAConfig, run: dict) -> bool:
    """
    Returns True if a saved JSON run's parameters match the current GAConfig.

    Compares the key numeric and string parameters that define the experiment
    setup.  If all match, the saved run is considered a valid cache hit for
    this config.

    The fields compared are: population_size, generations, crossover_rate,
    mutation_rate, elitism, selection_type, triangle_alpha_range,
    crossover_function name, and mutation_function name.

    Args:
        config: The current GAConfig to check against.
        run:    A loaded run dict as returned by load_all_runs().

    Returns:
        True if all key parameters match, False otherwise.
    """

    p = run.get("parameters", {})
    return (
        p.get("population_size")    == config.population_size
        and p.get("generations")    == config.generations
        and p.get("crossover_rate") == config.crossover_rate
        and p.get("mutation_rate")  == config.mutation_rate
        and p.get("elitism")        == config.elitism
        and p.get("selection_type") == config.selection_type
        # alpha range is stored as a list in JSON, but as a tuple in GAConfig
        and p.get("triangle_alpha_range") == list(config.triangle_alpha_range)
        # compare by function name so the check works across Python sessions
        and p.get("crossover_function") == getattr(config.crossover_function, "__name__", "")
        and p.get("mutation_function")  == getattr(config.mutation_function,  "__name__", "")
    )


def _load_cached_results(
    pipeline: str,
    results_dir: Path | str,
    config: GAConfig,
) -> list[dict]:
    """
    Returns all saved runs that match the pipeline name and config parameters.

    Each returned dict uses the same keys as ``run_single_ga`` output so that
    cached and freshly-computed results can be combined uniformly:
    ``fitness``, ``history``, ``individual``, ``runtime``, ``trial``,
    ``label``, ``params``.

    Returns an empty list when the directory does not exist or no runs match.

    Args:
        pipeline:    Pipeline name to search for (e.g. "Baseline-GA").
        results_dir: Directory containing saved JSON files.
        config:      Current GAConfig to match against saved parameters.

    Returns:
        List of normalised run dicts, sorted oldest-first.
    """

    from .results import load_all_runs

    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    loaded = []
    for r in load_all_runs(results_dir):
        if r.get("pipeline") == pipeline and _params_match(config, r):
            loaded.append({
                "fitness":    r["results"]["best_fitness"],
                "history":    r["results"]["history"],
                "individual": r["individual"],
                "runtime":    r.get("runtime_seconds", 0),
                "trial":      len(loaded),
                "label":      pipeline,
                "params":     r.get("parameters", {}),
            })
    return loaded


def _build_trial_summary(
    pipeline: str,
    results: list[dict],
    total_runtime: float,
) -> TrialSummary:
    """
    Builds a TrialSummary from a list of run dicts (cached or fresh).

    Args:
        pipeline:      Pipeline label for the summary.
        results:       List of dicts with keys fitness, history, individual.
        total_runtime: Combined wall-clock seconds for all runs.

    Returns:
        Aggregated TrialSummary across all supplied results.
    """

    fitnesses   = [r["fitness"]    for r in results]
    histories   = [r["history"]    for r in results]
    individuals = [r["individual"] for r in results]

    return TrialSummary(
        pipeline=pipeline,
        n_trials=len(results),
        mean_fitness=float(np.mean(fitnesses)),
        std_fitness=float(np.std(fitnesses)),
        min_fitness=float(np.min(fitnesses)),
        max_fitness=float(np.max(fitnesses)),
        mean_history=np.mean(histories, axis=0).tolist(),
        std_history=np.std(histories, axis=0),
        best_individual=individuals[int(np.argmin(fitnesses))],
        all_fitness=fitnesses,
        total_runtime=total_runtime,
    )


# ---------------------------------------------------------------------------
# Worker function — must be at module level to be picklable by spawn workers
# ---------------------------------------------------------------------------

def run_single_ga(config: GAConfig) -> dict:
    """
    Runs one complete GA trial and returns results as a plain serialisable dict.

    This function is the unit of work submitted to ``ProcessPoolExecutor``.
    It must be defined at module level (not inside a class or another function)
    so that Python's multiprocessing can pickle and send it to worker processes
    — this is a requirement of the macOS "spawn" start method.

    The GA inside this worker uses ``evaluation_backend="sequential"`` to avoid
    nested process pools: the outer pool (in run_trials / run_grid_search) already
    provides CPU parallelism across trials, so each individual trial evaluates
    its population sequentially on its assigned core.

    Args:
        config: Fully populated GAConfig for this specific trial.

    Returns:
        Dict with keys: label, trial, elitism, fitness, history,
        individual, runtime, params.
    """

    # Deferred import inside the worker to avoid import-time side effects
    # when the module is first loaded in the main process
    from .algorithm import GeneticAlgorithm

    print(f"  → [{config.label}] Trial {config.trial + 1} starting  "
          f"(pop={config.population_size}, gen={config.generations})")

    ga = GeneticAlgorithm(
        target=config.target,
        fitness_function=config.fitness_function,
        population_size=config.population_size,
        generations=config.generations,
        crossover_function=config.crossover_function,
        crossover_rate=config.crossover_rate,
        mutation_function=config.mutation_function,
        mutation_rate=config.mutation_rate,
        elitism=config.elitism,
        selection_type=config.selection_type,
        triangle_alpha_range=config.triangle_alpha_range,
        evaluation_backend="sequential",  # no nested pools
        progress=False,                   # no per-generation noise from workers
    )

    t0 = time.perf_counter()
    best_fitness, history = ga.run()
    runtime = time.perf_counter() - t0

    print(f"  ✓ [{config.label}] Trial {config.trial + 1} done     "
          f"RMSE={best_fitness:.6f}  ({runtime:.1f}s)")

    return {
        "label":      config.label,
        "trial":      config.trial,
        "elitism":    config.elitism,
        "fitness":    best_fitness,
        "history":    history,
        "individual": ga.best_individual,
        "runtime":    runtime,
        "params":     ga.params_dict(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_trials(
    config: GAConfig,
    n_trials: int,
    pipeline: str,
    results_dir: Path | str,
    notes: str = "",
    max_workers: int | None = None,
) -> TrialSummary:
    """
    Runs N independent trials of one GA configuration and returns aggregated statistics.

    Trials run in parallel using a ProcessPoolExecutor.  Before running, the
    function checks the disk cache: only the trials that are missing are run;
    already-saved results are loaded and merged with the new ones.

    Args:
        config:      Base GA configuration.  ``trial`` and ``label`` are
                     set automatically for each worker.
        n_trials:    Total number of independent trials desired.
        pipeline:    Label written into each saved JSON and used as the
                     cache key (e.g. ``"Baseline-GA"``).
        results_dir: Directory where individual trial JSONs are saved.
        notes:       Free-text note appended to every saved run.
        max_workers: Number of parallel processes.  Defaults to the number
                     of missing trials (one process per missing trial).

    Returns:
        TrialSummary with mean/std RMSE, convergence history, and best individual.
    """

    results_dir = Path(results_dir)

    # Load however many matching trials are already on disk
    existing = _load_cached_results(pipeline, results_dir, config)
    n_existing = len(existing)

    if n_existing >= n_trials:
        # Full cache hit — take the most recent n_trials runs
        runs = existing[-n_trials:]
        summary = _build_trial_summary(pipeline, runs,
                                       sum(r["runtime"] for r in runs))
        print(f"✓ '{pipeline}' — loaded {n_trials} cached trials  "
              f"(mean RMSE={summary.mean_fitness:.6f} ± {summary.std_fitness:.6f})")
        return summary

    # Partial or cold cache — only run the missing trials
    n_missing = n_trials - n_existing
    workers = max_workers or n_missing

    if n_existing:
        print(f"~ '{pipeline}' — {n_existing} cached + {n_missing} missing  "
              f"pop={config.population_size}  gen={config.generations}  workers={workers}")
    else:
        print(f"┌─ '{pipeline}'  {n_trials} trials  "
              f"pop={config.population_size}  gen={config.generations}  "
              f"workers={workers}")

    # Trial indices continue from where the cached runs left off
    configs = [
        GAConfig(**{**config.__dict__, "trial": n_existing + i, "label": pipeline})
        for i in range(n_missing)
    ]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_ga, c): c for c in configs}
        new_results = []
        for future in as_completed(futures):
            new_results.append(future.result())
    new_runtime = time.perf_counter() - t0

    new_results.sort(key=lambda r: r["trial"])

    for r in new_results:
        save_run(
            pipeline=pipeline,
            parameters=r["params"],
            best_fitness=r["fitness"],
            history=r["history"],
            best_individual=r["individual"],
            runtime_seconds=r["runtime"],
            notes=f"{notes} [trial {r['trial'] + 1}/{n_trials}]",
            results_dir=results_dir,
        )

    all_results = (existing + new_results)[-n_trials:]
    total_runtime = sum(r["runtime"] for r in existing) + new_runtime
    summary = _build_trial_summary(pipeline, all_results, total_runtime)

    print(f"└─ {summary.pipeline} — mean RMSE: {summary.mean_fitness:.6f} ± {summary.std_fitness:.6f}  "
          f"best: {summary.min_fitness:.6f}  ({new_runtime:.0f}s for new trials)")
    return summary


def run_grid_search(
    base_config: GAConfig,
    grid: dict[str, list[Any]],
    n_trials: int,
    pipeline_prefix: str,
    results_dir: Path | str,
    notes: str = "",
    max_workers: int | None = None,
) -> dict[Any, TrialSummary]:
    """
    Runs a single-parameter grid search with all jobs dispatched to one pool.

    The key optimisation over a naive loop is that all ``(param_value × trial)``
    combinations are submitted to a *single* ``ProcessPoolExecutor`` rather than
    running one ``run_trials`` call per parameter value.  With 4 values × 5
    trials = 20 jobs and 12 CPU cores, all 20 jobs can overlap, using all cores
    for the full duration instead of wasting 7 cores while 5 run at a time.

    Per-value cache checking is still performed: any values that already have
    n_trials saved results are loaded from disk and excluded from the job list,
    so only the missing values are computed.

    Only one parameter key may be varied per call.

    Args:
        base_config:     Shared GA configuration.  The varied key is overridden
                         per job via ``**{**base_config.__dict__, param_name: value}``.
        grid:            Exactly one entry: ``{"param_name": [val1, val2, ...]}``.
        n_trials:        Number of independent trials per parameter value.
        pipeline_prefix: Prefix for pipeline labels.  E.g. ``"PopSize"`` produces
                         ``"PopSize-100"``, ``"PopSize-200"``, …
        results_dir:     Directory where individual trial JSONs are saved.
        notes:           Free-text note appended to every saved run.
        max_workers:     Parallel processes.  Defaults to ``os.cpu_count()``
                         (uses all logical CPU cores).

    Returns:
        Dict mapping each parameter value to its TrialSummary.

    Raises:
        ValueError: If grid does not contain exactly one key.
    """

    if len(grid) != 1:
        raise ValueError("run_grid_search expects exactly one key in `grid`.")

    param_name, param_values = next(iter(grid.items()))
    workers = max_workers or os.cpu_count() or 1
    results_dir = Path(results_dir)

    # --- Per-value cache check: separate fully-cached from needing work ---
    summaries: dict[Any, TrialSummary] = {}
    cached_per_value: dict[Any, list[dict]] = {}

    for value in param_values:
        pipeline = f"{pipeline_prefix}-{value}"
        cfg = GAConfig(**{**base_config.__dict__, param_name: value})
        existing = _load_cached_results(pipeline, results_dir, cfg)

        if len(existing) >= n_trials:
            runs = existing[-n_trials:]
            summary = _build_trial_summary(pipeline, runs,
                                           sum(r["runtime"] for r in runs))
            summaries[value] = summary
            print(f"✓ '{pipeline}' — loaded {n_trials} cached trials  "
                  f"(mean RMSE={summary.mean_fitness:.6f} ± {summary.std_fitness:.6f})")
        else:
            cached_per_value[value] = existing

    if not cached_per_value:
        best_value = min(summaries, key=lambda v: summaries[v].mean_fitness)
        print(f"All {len(param_values)} values loaded from cache.  "
              f"Best {param_name}: {best_value}  "
              f"(mean RMSE={summaries[best_value].mean_fitness:.6f})")
        return summaries

    # --- Build flat job list: only the missing trials for each value ---
    all_configs: list[GAConfig] = []
    for value, existing in cached_per_value.items():
        pipeline = f"{pipeline_prefix}-{value}"
        n_missing = n_trials - len(existing)
        start_trial = len(existing)
        for i in range(n_missing):
            all_configs.append(
                GAConfig(**{**base_config.__dict__, param_name: value,
                            "trial": start_trial + i, "label": pipeline})
            )

    total_jobs = len(all_configs)
    print(f"┌─ Grid search  {param_name}={list(cached_per_value.keys())}")
    for value, existing in cached_per_value.items():
        n_missing = n_trials - len(existing)
        if existing:
            print(f"│  {pipeline_prefix}-{value}: {len(existing)} cached + {n_missing} new")
        else:
            print(f"│  {pipeline_prefix}-{value}: {n_trials} new trials")
    print(f"│  {total_jobs} jobs total  max_workers={workers}")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_ga, c): c for c in all_configs}
        all_new_unordered = []
        completed = 0
        for future in as_completed(futures):
            all_new_unordered.append(future.result())
            completed += 1
            print(f"│  {completed}/{total_jobs} jobs done")
    total_runtime = time.perf_counter() - t0

    print(f"│  All jobs done in {total_runtime:.0f}s")

    # Group new results by value label
    new_by_value: dict[Any, list[dict]] = {v: [] for v in cached_per_value}
    order = {(c.label, c.trial): i for i, c in enumerate(all_configs)}
    for r in sorted(all_new_unordered, key=lambda r: order[(r["label"], r["trial"])]):
        for value in cached_per_value:
            if r["label"] == f"{pipeline_prefix}-{value}":
                new_by_value[value].append(r)
                break

    # --- Save new results and build summaries by merging cached + new ---
    for value in cached_per_value:
        pipeline = f"{pipeline_prefix}-{value}"
        existing    = cached_per_value[value]
        new_results = new_by_value[value]

        for r in new_results:
            save_run(
                pipeline=pipeline,
                parameters=r["params"],
                best_fitness=r["fitness"],
                history=r["history"],
                best_individual=r["individual"],
                runtime_seconds=r["runtime"],
                notes=f"{notes} [trial {r['trial'] + 1}/{n_trials}]",
                results_dir=results_dir,
            )

        all_results = (existing + new_results)[-n_trials:]
        runtime = sum(r["runtime"] for r in existing) + (total_runtime / len(cached_per_value))
        summary = _build_trial_summary(pipeline, all_results, runtime)
        summaries[value] = summary
        print(f"│  {summary.pipeline:<20} mean RMSE={summary.mean_fitness:.6f} ± {summary.std_fitness:.6f}  "
              f"best={summary.min_fitness:.6f}")

    # Final summary line showing the winning parameter value
    best_value = min(param_values, key=lambda v: summaries[v].mean_fitness)
    print(f"└─ Best {param_name}: {best_value}  "
          f"(mean RMSE={summaries[best_value].mean_fitness:.6f})")
    return summaries
