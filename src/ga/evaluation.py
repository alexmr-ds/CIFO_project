"""Computes GA fitness values with sequential and parallel backends."""

import concurrent.futures
import pickle
from collections.abc import Callable
from itertools import repeat
from typing import Literal

import numpy as np

from .. import population, rendering

Individual = list[population.Triangle]
FitnessFunction = Callable[[np.ndarray, np.ndarray], float]
EvaluationBackend = Literal["sequential", "thread", "process"]
Executor = concurrent.futures.Executor

_PROCESS_TARGET: np.ndarray | None = None
_PROCESS_FITNESS_FUNCTION: FitnessFunction | None = None
_PROCESS_IMAGE_WIDTH: int | None = None
_PROCESS_IMAGE_HEIGHT: int | None = None


def normalize_evaluation_backend(evaluation_backend: str) -> EvaluationBackend:
    """Normalizes and validates the evaluation backend name."""

    normalized_backend = evaluation_backend.strip().lower()

    if normalized_backend == "sequential":
        return "sequential"
    if normalized_backend == "thread":
        return "thread"
    if normalized_backend == "process":
        return "process"

    raise ValueError(
        "evaluation_backend must be 'sequential', 'thread', or 'process'."
    )


def validate_optional_positive_int(value: int | None, name: str) -> None:
    """Validates optional positive integer executor settings."""

    if value is None:
        return

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be None or a positive integer.")


def validate_process_fitness_function(fitness_function: FitnessFunction) -> None:
    """Fails early when a fitness function cannot be sent to workers."""

    try:
        pickle.dumps(fitness_function)
    except Exception as exc:
        raise ValueError(
            "process evaluation requires a picklable fitness_function. "
            "Use a module-level function such as src.fitness.compute_rmse."
        ) from exc


def compute_individual_fitness(
    individual: Individual,
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
) -> float:
    """Computes the fitness for one individual."""

    generated = rendering.image_to_array(
        individual,
        image_width=image_width,
        image_height=image_height,
    )

    return float(fitness_function(target, generated))


def compute_population_fitness_sequential(
    population_data: list[Individual],
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
) -> list[float]:
    """Computes ordered population fitness values on the main process."""

    return [
        compute_individual_fitness(
            individual,
            target,
            fitness_function,
            image_width,
            image_height,
        )
        for individual in population_data
    ]


def compute_population_fitness_with_executor(
    executor: Executor,
    evaluation_backend: EvaluationBackend,
    population_data: list[Individual],
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
    chunksize: int | None = None,
) -> list[float]:
    """Computes ordered population fitness values with an executor."""

    if evaluation_backend == "thread":
        return list(
            executor.map(
                compute_individual_fitness,
                population_data,
                repeat(target),
                repeat(fitness_function),
                repeat(image_width),
                repeat(image_height),
            )
    )

    if chunksize is None:
        return list(
            executor.map(_compute_individual_fitness_in_process, population_data)
        )

    return list(
        executor.map(
            _compute_individual_fitness_in_process,
            population_data,
            chunksize=chunksize,
        )
    )


def create_evaluation_executor(
    evaluation_backend: EvaluationBackend,
    n_jobs: int | None,
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
) -> Executor:
    """Creates the configured executor for non-sequential evaluation."""

    if evaluation_backend == "thread":
        return concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs)

    return concurrent.futures.ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_initialize_process_worker,
        initargs=(
            target,
            fitness_function,
            image_width,
            image_height,
        ),
    )


def _initialize_process_worker(
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
) -> None:
    """Stores shared evaluation inputs once per worker process."""

    global _PROCESS_TARGET
    global _PROCESS_FITNESS_FUNCTION
    global _PROCESS_IMAGE_WIDTH
    global _PROCESS_IMAGE_HEIGHT

    _PROCESS_TARGET = target
    _PROCESS_FITNESS_FUNCTION = fitness_function
    _PROCESS_IMAGE_WIDTH = image_width
    _PROCESS_IMAGE_HEIGHT = image_height


def _compute_individual_fitness_in_process(individual: Individual) -> float:
    """Computes one fitness value using process-local evaluation state."""

    if (
        _PROCESS_TARGET is None
        or _PROCESS_FITNESS_FUNCTION is None
        or _PROCESS_IMAGE_WIDTH is None
        or _PROCESS_IMAGE_HEIGHT is None
    ):
        raise RuntimeError("Process worker evaluation state was not initialized.")

    return compute_individual_fitness(
        individual,
        _PROCESS_TARGET,
        _PROCESS_FITNESS_FUNCTION,
        _PROCESS_IMAGE_WIDTH,
        _PROCESS_IMAGE_HEIGHT,
    )
