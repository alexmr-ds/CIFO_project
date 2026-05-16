"""
Structured logging types and builders for GA run telemetry.

When ``logs=True`` is passed to GeneticAlgorithm, the algorithm captures
detailed per-generation metrics and stores them in ``ga.run_logs`` after
``ga.run()`` completes.

GenerationLog captures: fitness values, evaluation duration, mutation rate,
and offspring / mutation counts per generation.
"""

from dataclasses import asdict
from typing import TypedDict, cast

from .. import population
from . import evaluation

Individual = list[population.Triangle]


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class SerializedTriangle(TypedDict):
    """JSON-serialisable representation of one triangle (all fields are plain ints)."""

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
    """Per-generation telemetry captured during a GA run."""

    generation: int                        # 0-indexed generation number
    generation_best_fitness: float         # best fitness seen in this generation
    generation_mean_fitness: float         # mean fitness across all individuals
    global_best_fitness: float             # best fitness ever seen so far
    evaluation_backend: evaluation.EvaluationBackend  # which backend was used
    n_jobs: int | None                     # number of parallel workers (if any)
    chunksize: int | None                  # process-pool chunk size (if any)
    evaluation_duration_seconds: float     # wall-clock time for fitness evaluation
    mutation_rate_used: float              # effective mutation rate this generation
    offspring_created: int                 # total new individuals created
    mutated_offspring: int                 # individuals that had at least one change
    mutated_triangles: int                 # total triangle attributes changed


class RunLogs(TypedDict, total=False):
    """Top-level log payload for a complete GA run (all keys optional, set by run())."""

    generations: list[GenerationLog]                        # all generation entries
    best_fitness: float                                      # final global best RMSE
    best_individual_configuration: list[SerializedTriangle] # best individual as JSON


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_generation_log(
    generation: int,
    generation_best_fitness: float,
    generation_mean_fitness: float,
    global_best_fitness: float,
    evaluation_backend: evaluation.EvaluationBackend,
    n_jobs: int | None,
    chunksize: int | None,
    evaluation_duration_seconds: float,
    mutation_rate_used: float,
    offspring_created: int,
    mutated_offspring: int,
    mutated_triangles: int,
) -> GenerationLog:
    """Build and return one GenerationLog entry (called once per generation)."""
    return {
        "generation":                   generation,
        "generation_best_fitness":      generation_best_fitness,
        "generation_mean_fitness":      generation_mean_fitness,
        "global_best_fitness":          global_best_fitness,
        "evaluation_backend":           evaluation_backend,
        "n_jobs":                       n_jobs,
        "chunksize":                    chunksize,
        "evaluation_duration_seconds":  evaluation_duration_seconds,
        "mutation_rate_used":           mutation_rate_used,
        "offspring_created":            offspring_created,
        "mutated_offspring":            mutated_offspring,
        "mutated_triangles":            mutated_triangles,
    }


def create_run_logs(
    generation_logs: list[GenerationLog],
    best_fitness: float,
    best_individual: Individual,
) -> RunLogs:
    """Assemble the top-level RunLogs payload from a completed run."""

    return {
        "generations":                    list(generation_logs),
        "best_fitness":                   float(best_fitness),
        "best_individual_configuration":  serialize_individual(best_individual),
    }


def serialize_individual(individual: Individual) -> list[SerializedTriangle]:
    """Convert a list of Triangle dataclasses to JSON-serialisable dicts."""
    return [
        cast(
            SerializedTriangle,
            {field_name: int(value) for field_name, value in asdict(triangle).items()},
        )
        for triangle in individual
    ]
