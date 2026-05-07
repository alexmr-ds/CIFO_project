"""
Higher-level staged workflow for multi-phase triangle optimization.

What is a staged run?
---------------------
A staged run splits the optimization into sequential phases, each with its
own triangle count and generation budget.  For example:

    Stage 0:  20 triangles,  100 generations  → coarse approximation
    Stage 1:  50 triangles,  150 generations  → medium detail
    Stage 2: 100 triangles,  200 generations  → fine detail

At the end of each stage the best individual found so far is used to seed
the next stage's population.  The individual is *expanded* to the new
triangle count by appending random triangles for the extra slots.  The
rest of the population is created by applying small mutations to the best
individual, so the search starts close to a known good solution instead
of from scratch.

Why use staged runs?
--------------------
Starting with fewer triangles keeps the search space small and allows the
GA to find a good coarse-grained approximation quickly.  Increasing the
triangle count in later stages then adds detail on top of a solid foundation,
rather than trying to place 100 triangles correctly from a random start.

Data classes
------------
StageConfig     — parameters for one stage
StageResult     — output metrics from one completed stage
StagedRunResult — aggregate output of the full staged run
"""

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


# ---------------------------------------------------------------------------
# Configuration and result data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageConfig:
    """
    Configuration for one stage in a multi-stage optimization run.

    Only ``n_triangles`` and ``generations`` are required.  All optional
    fields override the shared parameters passed to
    ``run_staged_triangle_optimization``.  If left as None, the shared
    values are used unchanged.

    Attributes:
        n_triangles:          Number of triangles for this stage.
        generations:          Number of generations to run in this stage.
        mutation_rate:        Per-triangle mutation probability, or None to
                              use the shared value.
        crossover_rate:       Crossover probability, or None to use shared.
        adaptive_mutation:    Whether to use adaptive mutation this stage.
        mutation_rate_bounds: (min, max) bounds for adaptive mutation.
        stagnation_window:    Generations without improvement before boost.
        random_immigrants:    Random individuals injected each generation.
    """

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
    """
    Summary metrics for one completed optimization stage.

    Attributes:
        stage_index:  0-based index of this stage in the pipeline.
        n_triangles:  Triangle count used in this stage.
        generations:  Number of generations this stage ran.
        best_fitness: Best RMSE achieved at the end of this stage.
        history:      Per-generation global best RMSE list for this stage.
    """

    stage_index: int
    n_triangles: int
    generations: int
    best_fitness: float
    history: list[float]


@dataclass(frozen=True)
class StagedRunResult:
    """
    Final output of a complete staged optimization run.

    Attributes:
        best_fitness:    Lowest RMSE achieved across all stages combined.
        history:         Concatenated convergence history across all stages.
        best_individual: Best triangle individual found (at any stage).
        stage_results:   Per-stage metrics for detailed analysis.
    """

    best_fitness: float
    history: list[float]
    best_individual: Individual
    stage_results: list[StageResult]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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
    """
    Resizes an individual to exactly ``n_triangles`` triangles.

    Two cases:
      - **More triangles needed**: append randomly initialised triangles
        until the target length is reached.
      - **Fewer triangles needed**: truncate the list (keep the first N).

    Truncation is rarely used in practice because staged runs typically
    increase the triangle count at each stage.

    Args:
        individual:           Source individual to expand or truncate.
        n_triangles:          Target number of triangles.
        image_width:          Canvas width for newly created triangles.
        image_height:         Canvas height for newly created triangles.
        triangle_alpha_range: Alpha range for newly appended triangles.
        target:               Target image used for seeded colour init.
        seeded:               Whether new triangles sample colours from target.

    Returns:
        A new Individual of exactly n_triangles triangles.

    Raises:
        ValueError: If n_triangles is not positive.
    """

    if n_triangles <= 0:
        raise ValueError("n_triangles must be greater than zero.")

    # Deep copy and truncate if the source is too long
    expanded = copy.deepcopy(individual[:n_triangles])

    # Append random triangles until we reach the desired count
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
    """
    Creates a population seeded around the best individual from a prior stage.

    The first individual in the population is an exact copy of the best
    individual (expanded to ``n_triangles`` if necessary).  The remaining
    ``population_size - 1`` individuals are mutated copies of that same
    base, creating a cluster of candidates near the known good solution.

    This warm-starting strategy is much more efficient than re-initialising
    from scratch because the search begins close to where the previous stage
    left off.

    Args:
        best_individual:      The best individual from the previous stage.
        population_size:      Number of individuals in the new population.
        n_triangles:          Triangle count for the new stage.
        image_width:          Canvas width.
        image_height:         Canvas height.
        seed_mutation_rate:   Mutation rate applied when creating the variants.
                              A small rate (e.g. 0.08) keeps variants close to
                              the best individual.
        triangle_alpha_range: Alpha range for any newly appended triangles.
        target:               Target image for seeded colour initialisation.
        seeded:               Whether new triangles sample colours from target.

    Returns:
        A list of population_size individuals, all close to best_individual.

    Raises:
        ValueError: If population_size or seed_mutation_rate is invalid.
    """

    if population_size <= 0:
        raise ValueError("population_size must be greater than zero.")
    if not 0.0 <= seed_mutation_rate <= 1.0:
        raise ValueError("seed_mutation_rate must be between 0 and 1.")

    # Expand to the new triangle count (adds random triangles if needed)
    base_individual = expand_individual_to_triangle_count(
        individual=best_individual,
        n_triangles=n_triangles,
        image_width=image_width,
        image_height=image_height,
        triangle_alpha_range=triangle_alpha_range,
        target=target,
        seeded=seeded,
    )

    # The first population member is an exact copy of the best individual
    seeded_population = [copy.deepcopy(base_individual)]

    # Create the rest by applying light mutations to the base individual.
    # Each variant is independently mutated so the population has diversity
    # while still clustering around the known good starting point.
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


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

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
    """
    Runs a multi-stage GA with increasing triangle counts.

    Each stage is a full GA run.  At the end of one stage, the best
    individual found so far is expanded and used to seed the next stage's
    population, so later stages refine the solution found by earlier ones.

    Parameters shared across all stages (elitism, selection_type, etc.) can
    be overridden per-stage via the StageConfig fields.

    Args:
        target:               RGB target image, shape (H, W, 3).
        fitness_function:     Callable: (target, generated) → scalar fitness.
        population_size:      Number of individuals per stage (shared).
        stages:               List of StageConfig objects, one per stage.
                              Must contain at least one entry.
        elitism:              Shared elitism value (copied to every stage).
        selection_type:       Shared selection strategy.
        logs:                 Whether to capture detailed per-generation logs.
        crossover_function:   Shared crossover operator.
        mutation_function:    Shared mutation operator.
        evaluation_backend:   "sequential", "thread", or "process".
        n_jobs:               Worker count for parallel backends.
        chunksize:            Batch size for process pool.
        triangle_alpha_range: Alpha range shared across all stages.
        seeded:               Whether to seed triangle colours from target pixels.
        seed_mutation_rate:   Mutation rate used when building the seeded
                              population at the start of each stage.

    Returns:
        StagedRunResult containing the best individual, combined convergence
        history, and per-stage breakdown.

    Raises:
        ValueError: If stages is empty or any stage has invalid parameters.
        RuntimeError: If a stage completes without producing a best individual.
    """

    if not stages:
        raise ValueError("stages must contain at least one StageConfig.")

    image_height, image_width = target.shape[:2]

    # State accumulates across stages
    combined_history: list[float] = []   # full convergence trace across all stages
    stage_results: list[StageResult] = []
    best_individual: Individual | None = None
    best_fitness = float("inf")

    for stage_index, stage in enumerate(stages):
        # Validate stage parameters early for clearer error messages
        if stage.n_triangles <= 0:
            raise ValueError("Each stage n_triangles must be greater than zero.")
        if stage.generations <= 0:
            raise ValueError("Each stage generations value must be greater than zero.")
        if stage.stagnation_window <= 0:
            raise ValueError(
                "Each stage stagnation_window must be greater than zero."
            )

        # Build the initial population for this stage:
        # - Stage 0: no prior best → GA will generate a random population
        # - Stage 1+: seed around the best individual found so far
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

        # Create and run the GA for this stage
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

        # Update the global best if this stage improved on prior stages
        if stage_best_fitness < best_fitness:
            best_fitness = stage_best_fitness
            # Deep copy to prevent the next stage's mutations from corrupting it
            best_individual = copy.deepcopy(ga.best_individual)

        # Append this stage's history so the full run can be plotted as one curve
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
