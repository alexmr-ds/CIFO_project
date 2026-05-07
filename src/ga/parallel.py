"""Parallel GA trial runner.

Provides GAConfig, TrialSummary, and run_trials so notebook cells only
need to declare what to run, not how to run it in parallel.

All callables stored in GAConfig are module-level functions, which Python's
spawn-based multiprocessing can pickle and send to worker processes without
the issues that arise with lambdas or notebook-defined functions.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import population
from .results import save_run

Individual = list[population.Triangle]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """Fully self-contained configuration for one GA trial."""

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
    trial: int = 0
    label: str = ""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrialSummary:
    """Aggregated statistics across N repeated GA trials."""

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
# Worker (must stay at module level to be picklable)
# ---------------------------------------------------------------------------

def run_single_ga(config: GAConfig) -> dict:
    """Runs one GA trial and returns results as a plain dict.

    Uses sequential evaluation internally so this function is safe to call
    from multiple ProcessPoolExecutor workers simultaneously without spawning
    nested process pools.
    """
    from .algorithm import GeneticAlgorithm

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
        evaluation_backend="sequential",
        progress=False,
    )

    t0 = time.perf_counter()
    best_fitness, history = ga.run()
    runtime = time.perf_counter() - t0

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
    """Runs N independent trials of a GA config in parallel and returns aggregated results.

    Args:
        config:      Base GA configuration (trial index is set automatically).
        n_trials:    Number of independent runs.
        pipeline:    Label written into each saved JSON (e.g. ``"Baseline-GA"``).
        results_dir: Directory where individual trial JSONs are saved.
        notes:       Free-text note appended to every saved run.
        max_workers: Number of parallel workers. Defaults to ``n_trials``.

    Returns:
        TrialSummary with mean/std fitness, mean/std history, and best individual.
    """
    workers = max_workers or n_trials
    configs = [
        GAConfig(**{**config.__dict__, "trial": i, "label": pipeline})
        for i in range(n_trials)
    ]

    print(f"Running {n_trials} trials for '{pipeline}' (max_workers={workers})...")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(run_single_ga, configs))
    total_runtime = time.perf_counter() - t0

    for r in results:
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

    fitnesses   = [r["fitness"]    for r in results]
    histories   = [r["history"]    for r in results]
    individuals = [r["individual"] for r in results]

    summary = TrialSummary(
        pipeline=pipeline,
        n_trials=n_trials,
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

    print(summary)
    return summary
