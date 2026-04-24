"""Builds JSON-friendly genetic algorithm run logs."""

from dataclasses import asdict
from typing import TypedDict, cast

from .. import population
from . import evaluation

Individual = list[population.Triangle]


class SerializedTriangle(TypedDict):
    """JSON-friendly representation of one triangle."""

    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    r: int
    g: int
    b: int
    a: int


class GenerationLog(TypedDict):
    """Per-generation GA metrics captured when logging is enabled."""

    generation: int
    generation_best_fitness: float
    generation_mean_fitness: float
    global_best_fitness: float
    evaluation_backend: evaluation.EvaluationBackend
    n_jobs: int | None
    chunksize: int | None
    evaluation_duration_seconds: float


class RunLogs(TypedDict, total=False):
    """Run-level GA log payload populated when logging is enabled."""

    generations: list[GenerationLog]
    best_fitness: float
    best_individual_configuration: list[SerializedTriangle]


def create_generation_log(
    generation: int,
    generation_best_fitness: float,
    generation_mean_fitness: float,
    global_best_fitness: float,
    evaluation_backend: evaluation.EvaluationBackend,
    n_jobs: int | None,
    chunksize: int | None,
    evaluation_duration_seconds: float,
) -> GenerationLog:
    """Creates one per-generation log entry."""

    return {
        "generation": generation,
        "generation_best_fitness": generation_best_fitness,
        "generation_mean_fitness": generation_mean_fitness,
        "global_best_fitness": global_best_fitness,
        "evaluation_backend": evaluation_backend,
        "n_jobs": n_jobs,
        "chunksize": chunksize,
        "evaluation_duration_seconds": evaluation_duration_seconds,
    }


def create_run_logs(
    generation_logs: list[GenerationLog],
    best_fitness: float,
    best_individual: Individual,
) -> RunLogs:
    """Creates the run-level log payload."""

    return {
        "generations": list(generation_logs),
        "best_fitness": float(best_fitness),
        "best_individual_configuration": serialize_individual(best_individual),
    }


def serialize_individual(individual: Individual) -> list[SerializedTriangle]:
    """Converts triangle dataclasses to JSON-friendly dictionaries."""

    return [
        cast(
            SerializedTriangle,
            {field_name: int(value) for field_name, value in asdict(triangle).items()},
        )
        for triangle in individual
    ]
