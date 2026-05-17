"""Per-generation telemetry type and builder used by progress_callback."""

from typing import TypedDict

from . import evaluation


class GenerationLog(TypedDict):
    """Per-generation telemetry passed to progress_callback each generation."""

    generation: int
    generation_best_fitness: float
    generation_mean_fitness: float
    global_best_fitness: float
    evaluation_backend: evaluation.EvaluationBackend
    n_jobs: int | None
    chunksize: int | None
    evaluation_duration_seconds: float
    mutation_rate_used: float
    offspring_created: int
    mutated_offspring: int
    mutated_triangles: int


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
    """Build one GenerationLog entry (called once per generation)."""
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
