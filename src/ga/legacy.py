"""Implements a legacy-style GA pipeline with stronger convergence heuristics."""

import copy
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from .. import population
from . import cross_over, evaluation, fitness, mutate, selection

Individual = list[population.Triangle]


class LegacyGenerationLog(TypedDict):
    """Stores per-generation telemetry for the legacy pipeline."""

    generation: int
    generation_best_fitness: float
    global_best_fitness: float
    stagnation_generations: int
    position_sigma: float
    color_sigma: float
    diversity_injected: bool
    local_search_improvements: int


@dataclass(frozen=True)
class LegacyPipelineConfig:
    """Configures the legacy-style GA workflow."""

    population_size: int = 200
    generations: int = 400
    n_triangles: int = 100
    elitism_fraction: float = 0.2
    tournament_size: int = 3
    mutation_rate: float = 0.15
    position_sigma: float = 0.05
    color_sigma: float = 0.08
    stagnation_boost_window: int = 15
    sigma_boost_factor: float = 3.0
    position_sigma_cap: float = 0.30
    color_sigma_cap: float = 0.40
    sigma_decay: float = 0.95
    diversity_window: int = 30
    diversity_replace_fraction: float = 0.2
    guided_sigma: float = 0.25
    local_search_steps: int = 50
    local_search_position_sigma: float = 0.025
    local_search_color_sigma: float = 0.0375
    evaluation_backend: evaluation.EvaluationBackend = "sequential"
    n_jobs: int | None = None
    chunksize: int | None = None
    seed: int = 42
    progress: bool = True
    progress_interval: int = 10


@dataclass(frozen=True)
class LegacyRunResult:
    """Captures the output of one legacy pipeline run."""

    best_fitness: float
    history: list[float]
    best_individual: Individual
    generation_logs: list[LegacyGenerationLog]


def _validate_config(config: LegacyPipelineConfig) -> None:
    """Validates legacy pipeline configuration values."""

    if config.population_size <= 0:
        raise ValueError("population_size must be greater than zero.")
    if config.generations <= 0:
        raise ValueError("generations must be greater than zero.")
    if config.n_triangles <= 0:
        raise ValueError("n_triangles must be greater than zero.")
    if not 0.0 <= config.elitism_fraction <= 1.0:
        raise ValueError("elitism_fraction must be between 0 and 1.")
    if config.tournament_size <= 0:
        raise ValueError("tournament_size must be greater than zero.")
    if not 0.0 <= config.mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1.")
    if config.position_sigma < 0.0 or config.color_sigma < 0.0:
        raise ValueError("position_sigma and color_sigma must be non-negative.")
    if config.stagnation_boost_window <= 0 or config.diversity_window <= 0:
        raise ValueError("stagnation windows must be greater than zero.")
    if config.sigma_boost_factor <= 0.0 or config.sigma_decay <= 0.0:
        raise ValueError("sigma factors must be greater than zero.")
    if config.local_search_steps < 0:
        raise ValueError("local_search_steps must be non-negative.")
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be greater than zero.")
    if not 0.0 <= config.diversity_replace_fraction <= 1.0:
        raise ValueError("diversity_replace_fraction must be between 0 and 1.")

    evaluation.normalize_evaluation_backend(config.evaluation_backend)
    evaluation.validate_optional_positive_int(config.n_jobs, "n_jobs")
    evaluation.validate_optional_positive_int(config.chunksize, "chunksize")


def _create_seeded_triangle(
    target: np.ndarray,
    image_width: int,
    image_height: int,
) -> population.Triangle:
    """Creates one random triangle with color seeded from the target image."""

    x1 = int(np.random.randint(0, image_width))
    y1 = int(np.random.randint(0, image_height))
    x2 = int(np.random.randint(0, image_width))
    y2 = int(np.random.randint(0, image_height))
    x3 = int(np.random.randint(0, image_width))
    y3 = int(np.random.randint(0, image_height))
    sample_x = int(np.random.randint(0, image_width))
    sample_y = int(np.random.randint(0, image_height))
    sampled_rgb = target[sample_y, sample_x].astype(np.int32)

    return population.Triangle(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        x3=x3,
        y3=y3,
        r=int(sampled_rgb[0]),
        g=int(sampled_rgb[1]),
        b=int(sampled_rgb[2]),
        a=255,
    )


def _create_seeded_population(
    target: np.ndarray,
    population_size: int,
    n_triangles: int,
    image_width: int,
    image_height: int,
) -> list[Individual]:
    """Creates a population with target-seeded triangle colors."""

    return [
        [
            _create_seeded_triangle(
                target=target,
                image_width=image_width,
                image_height=image_height,
            )
            for _ in range(n_triangles)
        ]
        for _ in range(population_size)
    ]


def _mutate_with_sigmas(
    individual: Individual,
    mutation_rate: float,
    image_width: int,
    image_height: int,
    position_sigma: float,
    color_sigma: float,
) -> Individual:
    """Applies Gaussian mutation with separate position and color sigmas."""

    return mutate.gaussian_triangle_mutation(
        individual=individual,
        mutation_rate=mutation_rate,
        image_width=image_width,
        image_height=image_height,
        triangle_alpha_range=(255, 255),
        position_sigma=position_sigma,
        color_sigma=color_sigma,
        force_opaque=True,
    )


def _local_search_on_best(
    individual: Individual,
    fitness_value: float,
    target: np.ndarray,
    fitness_function: evaluation.FitnessFunction,
    image_width: int,
    image_height: int,
    steps: int,
    position_sigma: float,
    color_sigma: float,
) -> tuple[Individual, float, int]:
    """Performs hill-climbing on one individual and accepts only improvements."""

    if steps == 0:
        return individual, fitness_value, 0

    current = copy.deepcopy(individual)
    current_fitness = float(fitness_value)
    accepted_improvements = 0

    for _ in range(steps):
        candidate = copy.deepcopy(current)
        mutated_index = int(np.random.randint(0, len(candidate)))
        mutate.gaussian_triangle_mutation(
            individual=[candidate[mutated_index]],
            mutation_rate=1.0,
            image_width=image_width,
            image_height=image_height,
            triangle_alpha_range=(255, 255),
            position_sigma=position_sigma,
            color_sigma=color_sigma,
            force_opaque=True,
        )
        candidate_fitness = evaluation.compute_individual_fitness(
            individual=candidate,
            target=target,
            fitness_function=fitness_function,
            image_width=image_width,
            image_height=image_height,
        )
        if candidate_fitness < current_fitness:
            current = candidate
            current_fitness = float(candidate_fitness)
            accepted_improvements += 1

    return current, current_fitness, accepted_improvements


def _apply_diversity_injection(
    population_data: list[Individual],
    ranked_indices: np.ndarray,
    best_individual: Individual,
    replace_fraction: float,
    guided_sigma: float,
    target: np.ndarray,
    image_width: int,
    image_height: int,
) -> None:
    """Replaces worst individuals with guided and fresh candidates."""

    replace_count = int(len(population_data) * replace_fraction)
    if replace_count == 0:
        return

    worst_indices = [int(index) for index in ranked_indices[-replace_count:]]
    guided_count = replace_count // 2

    for local_index, population_index in enumerate(worst_indices):
        if local_index < guided_count:
            guided = copy.deepcopy(best_individual)
            _mutate_with_sigmas(
                individual=guided,
                mutation_rate=1.0,
                image_width=image_width,
                image_height=image_height,
                position_sigma=guided_sigma,
                color_sigma=guided_sigma,
            )
            population_data[population_index] = guided
            continue

        population_data[population_index] = [
            _create_seeded_triangle(
                target=target,
                image_width=image_width,
                image_height=image_height,
            )
            for _ in range(len(best_individual))
        ]


def run_legacy_pipeline(
    target: np.ndarray,
    config: LegacyPipelineConfig = LegacyPipelineConfig(),
    fitness_function: evaluation.FitnessFunction = fitness.compute_rmse,
) -> LegacyRunResult:
    """Runs the legacy-style optimization pipeline with stagnation controls."""

    _validate_config(config)
    if target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target must have shape (H, W, 3).")
    if config.evaluation_backend == "process":
        evaluation.validate_process_fitness_function(fitness_function)

    np.random.seed(config.seed)
    image_height, image_width = target.shape[:2]

    population_data = _create_seeded_population(
        target=target,
        population_size=config.population_size,
        n_triangles=config.n_triangles,
        image_width=image_width,
        image_height=image_height,
    )

    best_individual: Individual | None = None
    best_fitness = float("inf")
    history: list[float] = []
    generation_logs: list[LegacyGenerationLog] = []
    stagnation_generations = 0
    current_position_sigma = float(config.position_sigma)
    current_color_sigma = float(config.color_sigma)
    executor: evaluation.Executor | None = None

    try:
        if config.evaluation_backend != "sequential":
            executor = evaluation.create_evaluation_executor(
                evaluation_backend=config.evaluation_backend,
                n_jobs=config.n_jobs,
                target=target,
                fitness_function=fitness_function,
                image_width=image_width,
                image_height=image_height,
            )

        for generation in range(config.generations):
            if config.evaluation_backend == "sequential":
                fitness_values = evaluation.compute_population_fitness_sequential(
                    population_data=population_data,
                    target=target,
                    fitness_function=fitness_function,
                    image_width=image_width,
                    image_height=image_height,
                )
            else:
                if executor is None:
                    raise RuntimeError("Evaluation executor was not initialized.")
                fitness_values = evaluation.compute_population_fitness_with_executor(
                    executor=executor,
                    evaluation_backend=config.evaluation_backend,
                    population_data=population_data,
                    target=target,
                    fitness_function=fitness_function,
                    image_width=image_width,
                    image_height=image_height,
                    chunksize=config.chunksize,
                )

            ranked_indices = np.argsort(fitness_values)
            generation_best_index = int(ranked_indices[0])
            generation_best_fitness = float(fitness_values[generation_best_index])
            generation_best_individual = copy.deepcopy(population_data[generation_best_index])

            local_search_improvements = 0
            if config.local_search_steps > 0:
                (
                    generation_best_individual,
                    generation_best_fitness,
                    local_search_improvements,
                ) = _local_search_on_best(
                    individual=generation_best_individual,
                    fitness_value=generation_best_fitness,
                    target=target,
                    fitness_function=fitness_function,
                    image_width=image_width,
                    image_height=image_height,
                    steps=config.local_search_steps,
                    position_sigma=config.local_search_position_sigma,
                    color_sigma=config.local_search_color_sigma,
                )

            improved = generation_best_fitness < best_fitness
            if improved:
                best_fitness = generation_best_fitness
                best_individual = copy.deepcopy(generation_best_individual)
                stagnation_generations = 0
                current_position_sigma = max(
                    config.position_sigma,
                    current_position_sigma * config.sigma_decay,
                )
                current_color_sigma = max(
                    config.color_sigma,
                    current_color_sigma * config.sigma_decay,
                )
            else:
                stagnation_generations += 1
                if stagnation_generations == config.stagnation_boost_window:
                    current_position_sigma = min(
                        config.position_sigma_cap,
                        current_position_sigma * config.sigma_boost_factor,
                    )
                    current_color_sigma = min(
                        config.color_sigma_cap,
                        current_color_sigma * config.sigma_boost_factor,
                    )

            diversity_injected = False
            if (
                best_individual is not None
                and stagnation_generations >= config.diversity_window
            ):
                _apply_diversity_injection(
                    population_data=population_data,
                    ranked_indices=ranked_indices,
                    best_individual=best_individual,
                    replace_fraction=config.diversity_replace_fraction,
                    guided_sigma=config.guided_sigma,
                    target=target,
                    image_width=image_width,
                    image_height=image_height,
                )
                stagnation_generations = 0
                diversity_injected = True

            history.append(float(best_fitness))
            generation_log: LegacyGenerationLog = {
                "generation": generation,
                "generation_best_fitness": generation_best_fitness,
                "global_best_fitness": float(best_fitness),
                "stagnation_generations": stagnation_generations,
                "position_sigma": current_position_sigma,
                "color_sigma": current_color_sigma,
                "diversity_injected": diversity_injected,
                "local_search_improvements": local_search_improvements,
            }
            generation_logs.append(generation_log)

            if config.progress and generation % config.progress_interval == 0:
                print(
                    f"[LEGACY] gen {generation + 1}/{config.generations} | "
                    f"best={best_fitness:.6f} | "
                    f"gen_best={generation_best_fitness:.6f} | "
                    f"sigmas=({current_position_sigma:.4f}, {current_color_sigma:.4f}) | "
                    f"stagnation={stagnation_generations}"
                )

            if generation == config.generations - 1:
                break

            elitism_count = int(config.population_size * config.elitism_fraction)
            elitism_count = min(config.population_size, max(1, elitism_count))
            next_population = [
                copy.deepcopy(population_data[int(index)])
                for index in ranked_indices[:elitism_count]
            ]

            while len(next_population) < config.population_size:
                parent1 = selection.tournament_selection(
                    population_data,
                    fitness_values,
                    k=config.tournament_size,
                )
                parent2 = selection.tournament_selection(
                    population_data,
                    fitness_values,
                    k=config.tournament_size,
                )
                child = cross_over.whole_triangle_crossover(
                    parent1=parent1,
                    parent2=parent2,
                    crossover_rate=1.0,
                )
                next_population.append(
                    _mutate_with_sigmas(
                        individual=child,
                        mutation_rate=config.mutation_rate,
                        image_width=image_width,
                        image_height=image_height,
                        position_sigma=current_position_sigma,
                        color_sigma=current_color_sigma,
                    )
                )

            population_data = next_population
    finally:
        if executor is not None:
            executor.shutdown()

    if best_individual is None:
        raise RuntimeError("The legacy pipeline did not produce a best individual.")

    return LegacyRunResult(
        best_fitness=float(best_fitness),
        history=history,
        best_individual=best_individual,
        generation_logs=generation_logs,
    )
