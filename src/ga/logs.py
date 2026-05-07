"""
Structured logging types and builders for GA run telemetry.

When ``logs=True`` is passed to GeneticAlgorithm, the algorithm captures
detailed per-generation metrics and stores them in ``ga.run_logs`` after
``ga.run()`` completes.  This module defines the data structures and the
factory functions that populate them.

Log structure
-------------
RunLogs
  └── generations: list[GenerationLog]   — one entry per generation
  └── best_fitness: float                — global best RMSE achieved
  └── best_individual_configuration      — serialised triangle list

GenerationLog fields of interest:
  - generation_best_fitness  : best RMSE found in *this* generation
  - global_best_fitness      : best RMSE found *so far* (never increases)
  - evaluation_duration_seconds : time spent computing fitness this gen
  - mutation_rate_used       : effective rate after adaptive scheduling
  - mutated_offspring / mutated_triangles : how much mutation occurred
  - immigrant_count          : how many random individuals were injected

These logs are useful for diagnosing convergence issues: e.g. a flat
global_best_fitness over many generations with a high mutation_rate_used
suggests the search is stuck.
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
    """
    JSON-serialisable representation of one triangle.

    All integer fields are stored as plain Python ints so they can be
    written directly to JSON without a custom encoder.
    """

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
    """
    Per-generation telemetry captured during a GA run.

    One entry is created at the end of each generation loop iteration.
    Together these entries form a complete audit trail of the run.
    """

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
    immigrant_count: int                   # random individuals injected this gen


class RunLogs(TypedDict, total=False):
    """
    Top-level log payload for a complete GA run.

    ``total=False`` means all keys are optional — the dict starts empty
    and is populated at the end of ``GeneticAlgorithm.run()`` when logging
    is enabled.
    """

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
    immigrant_count: int,
) -> GenerationLog:
    """
    Builds and returns one GenerationLog entry.

    This is called once per generation inside the GA main loop.  All values
    are passed explicitly so the function has no side effects and is easy
    to test in isolation.

    Returns:
        A fully populated GenerationLog TypedDict.
    """

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
        "immigrant_count":              immigrant_count,
    }


def create_run_logs(
    generation_logs: list[GenerationLog],
    best_fitness: float,
    best_individual: Individual,
) -> RunLogs:
    """
    Assembles the top-level RunLogs payload from a completed run.

    Called once at the end of ``GeneticAlgorithm.run()`` when logging is
    enabled.  Serialises the best individual to plain dicts so the result
    can be written to JSON without a custom encoder.

    Args:
        generation_logs: All per-generation log entries collected during the run.
        best_fitness:    Final global best fitness value.
        best_individual: The best triangle individual found during the run.

    Returns:
        A RunLogs TypedDict ready for JSON serialisation or inspection.
    """

    return {
        "generations":                    list(generation_logs),
        "best_fitness":                   float(best_fitness),
        "best_individual_configuration":  serialize_individual(best_individual),
    }


def serialize_individual(individual: Individual) -> list[SerializedTriangle]:
    """
    Converts a list of Triangle dataclasses to JSON-serialisable dicts.

    Each Triangle is converted via ``dataclasses.asdict`` and then all
    values are cast to plain Python ints so they serialise correctly
    (numpy integer types would fail with the default JSON encoder).

    Args:
        individual: List of Triangle objects to serialise.

    Returns:
        List of SerializedTriangle dicts with integer-typed fields.
    """

    return [
        cast(
            SerializedTriangle,
            # asdict returns a dict of field_name → value
            # We cast each value to int to strip any numpy integer wrapping
            {field_name: int(value) for field_name, value in asdict(triangle).items()},
        )
        for triangle in individual
    ]
