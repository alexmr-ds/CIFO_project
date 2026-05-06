"""Provides higher-level staged workflows for triangle-based GA runs."""

import copy
from dataclasses import dataclass

import numpy as np

from .. import population
from .algorithm import (
    CrossoverFunction,
    EvaluationBackend,
    FitnessFunction,
    GeneticAlgorithm,
    MutationFunction,
    RateBounds,
)
from . import mutate

Individual = list[population.Triangle]


@dataclass(frozen=True)
class StageConfig:
    """Configures one stage in a staged triangle-count optimization run."""

    n_triangles: int
    generations: int
    mutation_rate: float | None = None
    crossover_rate: float | None = None
    adaptive_mutation: bool = False
    mutation_rate_bounds: RateBounds | None = None
    stagnation_window: int = 8
    random_immigrants: int = 0


@dataclass(frozen=True)
class StageResult:
    """Stores summary metrics for a completed optimization stage."""

    stage_index: int
    n_triangles: int
    generations: int
    best_fitness: float
    history: list[float]


@dataclass(frozen=True)
class StagedRunResult:
    """Captures the final output of a staged optimization run."""

    best_fitness: float
    history: list[float]
    best_individual: Individual
    stage_results: list[StageResult]


def expand_individual_to_triangle_count(
    individual: Individual,
    n_triangles: int,
    image_width: int,
    image_height: int,
    triangle_alpha_range: population.AlphaRange = (
        population.TRIANGLE_ALPHA_RANGE
    ),
    target: np.ndarray | None = None,
    seeded: bool = False,
) -> Individual:
    """Expands or truncates an individual to match the requested triangle count."""

    if n_triangles <= 0:
        raise ValueError("n_triangles must be greater than zero.")

    expanded = copy.deepcopy(individual[:n_triangles])
    while len(expanded) < n_triangles:
        expanded.append(
            population.create_random_triangle(
                image_width=image_width,
                image_height=image_height,
                triangle_alpha_range=triangle_alpha_range,
                target=target,
                seeded=seeded,
            )
        )

    return expanded


def create_seed_population_from_best(
    best_individual: Individual,
    population_size: int,
    n_triangles: int,
    image_width: int,
    image_height: int,
    seed_mutation_rate: float,
    triangle_alpha_range: population.AlphaRange = (
        population.TRIANGLE_ALPHA_RANGE
    ),
    target: np.ndarray | None = None,
    seeded: bool = False,
) -> list[Individual]:
    """Creates a seeded initial population around a known best individual."""

    if population_size <= 0:
        raise ValueError("population_size must be greater than zero.")
    if not 0.0 <= seed_mutation_rate <= 1.0:
        raise ValueError("seed_mutation_rate must be between 0 and 1.")

    base_individual = expand_individual_to_triangle_count(
        individual=best_individual,
        n_triangles=n_triangles,
        image_width=image_width,
        image_height=image_height,
        triangle_alpha_range=triangle_alpha_range,
        target=target,
        seeded=seeded,
    )
    seeded_population = [copy.deepcopy(base_individual)]

    for _ in range(population_size - 1):
        candidate = copy.deepcopy(base_individual)
        mutate.random_triangle_mutation(
            individual=candidate,
            mutation_rate=seed_mutation_rate,
            image_width=image_width,
            image_height=image_height,
            triangle_alpha_range=triangle_alpha_range,
        )
        seeded_population.append(candidate)

    return seeded_population


def run_staged_triangle_optimization(
    target: np.ndarray,
    fitness_function: FitnessFunction,
    population_size: int,
    stages: list[StageConfig],
    elitism: int = 0,
    selection_type: str = "tournament",
    logs: bool = False,
    crossover_function: CrossoverFunction | None = None,
    mutation_function: MutationFunction | None = None,
    evaluation_backend: EvaluationBackend = "sequential",
    n_jobs: int | None = None,
    chunksize: int | None = None,
    triangle_alpha_range: population.AlphaRange = (
        population.TRIANGLE_ALPHA_RANGE
    ),
    seeded: bool = False,
    seed_mutation_rate: float = 0.08,
) -> StagedRunResult:
    """Runs multiple GA stages with increasing or customized triangle counts."""

    if not stages:
        raise ValueError("stages must contain at least one StageConfig.")

    image_height, image_width = target.shape[:2]
    combined_history: list[float] = []
    stage_results: list[StageResult] = []
    best_individual: Individual | None = None
    best_fitness = float("inf")

    for stage_index, stage in enumerate(stages):
        if stage.n_triangles <= 0:
            raise ValueError("Each stage n_triangles must be greater than zero.")
        if stage.generations <= 0:
            raise ValueError("Each stage generations value must be greater than zero.")
        if stage.stagnation_window <= 0:
            raise ValueError(
                "Each stage stagnation_window must be greater than zero."
            )

        initial_population: list[Individual] | None = None
        if best_individual is not None:
            initial_population = create_seed_population_from_best(
                best_individual=best_individual,
                population_size=population_size,
                n_triangles=stage.n_triangles,
                image_width=image_width,
                image_height=image_height,
                seed_mutation_rate=seed_mutation_rate,
                triangle_alpha_range=triangle_alpha_range,
                target=target,
                seeded=seeded,
            )

        ga = GeneticAlgorithm(
            target=target,
            fitness_function=fitness_function,
            population_size=population_size,
            generations=stage.generations,
            crossover_rate=stage.crossover_rate,
            mutation_rate=stage.mutation_rate,
            elitism=elitism,
            selection_type=selection_type,
            logs=logs,
            crossover_function=crossover_function,
            mutation_function=mutation_function,
            evaluation_backend=evaluation_backend,
            n_jobs=n_jobs,
            chunksize=chunksize,
            triangle_alpha_range=triangle_alpha_range,
            n_triangles=stage.n_triangles,
            adaptive_mutation=stage.adaptive_mutation,
            mutation_rate_bounds=stage.mutation_rate_bounds,
            stagnation_window=stage.stagnation_window,
            random_immigrants=stage.random_immigrants,
            initial_population=initial_population,
            seeded=seeded,
        )

        stage_best_fitness, stage_history = ga.run()
        if ga.best_individual is None:
            raise RuntimeError("Stage completed without a best individual.")

        if stage_best_fitness < best_fitness:
            best_fitness = stage_best_fitness
            best_individual = copy.deepcopy(ga.best_individual)

        combined_history.extend(stage_history)
        stage_results.append(
            StageResult(
                stage_index=stage_index,
                n_triangles=stage.n_triangles,
                generations=stage.generations,
                best_fitness=stage_best_fitness,
                history=list(stage_history),
            )
        )

    if best_individual is None:
        raise RuntimeError("The staged workflow did not produce a best individual.")

    return StagedRunResult(
        best_fitness=float(best_fitness),
        history=list(combined_history),
        best_individual=best_individual,
        stage_results=stage_results,
    )
