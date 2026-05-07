"""Picklable runner for parallel GA trials via ProcessPoolExecutor.

All callables stored in GAConfig are module-level functions, which Python's
spawn-based multiprocessing can pickle and send to worker processes without
the issues that arise with lambdas or notebook-defined functions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import population

Individual = list[population.Triangle]


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
