"""Implements the genetic algorithm for triangle-based image approximation."""

import copy
import time
from collections.abc import Callable

import numpy as np

from .. import population
from . import evaluation, logs, selection

Individual = list[population.Triangle]
FitnessFunction = evaluation.FitnessFunction
CrossoverFunction = Callable[[Individual, Individual, float], Individual]
MutationFunction = Callable[
    [Individual, float, int, int, population.AlphaRange], Individual
]
OperatorFunction = CrossoverFunction | MutationFunction
EvaluationBackend = evaluation.EvaluationBackend
RateBounds = tuple[float, float]
ProgressCallback = Callable[[logs.GenerationLog], None]


class GeneticAlgorithm:
    """
    Evolves triangle-based images to minimize a target-image fitness function.

    The class owns population initialization, parent selection, optional
    crossover/mutation operators, sequential or parallel fitness evaluation,
    global-best tracking, and optional in-memory run logs.
    """

    @staticmethod
    def _resolve_operator_rate(
        rate_name: str,
        rate: float | None,
        operator: OperatorFunction | None,
    ) -> float:
        """Validates and resolves an operator rate to an internal float."""

        if rate is None:
            if operator is not None:
                raise ValueError(
                    f"{rate_name} must be provided when its function is set."
                )
            return 0.0

        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{rate_name} must be between 0 and 1.")

        return float(rate)

    @staticmethod
    def _validate_optional_rate_bounds(
        mutation_rate_bounds: RateBounds | None,
    ) -> RateBounds | None:
        """Validates optional mutation-rate bounds for adaptive mutation."""

        if mutation_rate_bounds is None:
            return None
        if (
            not isinstance(mutation_rate_bounds, tuple)
            or len(mutation_rate_bounds) != 2
        ):
            raise ValueError("mutation_rate_bounds must be a tuple of two floats.")

        lower, upper = mutation_rate_bounds
        if not 0.0 <= lower <= 1.0 or not 0.0 <= upper <= 1.0:
            raise ValueError("mutation_rate_bounds values must be between 0 and 1.")
        if lower > upper:
            raise ValueError(
                "mutation_rate_bounds lower bound must be <= upper bound."
            )

        return float(lower), float(upper)

    @staticmethod
    def _normalize_initial_population(
        initial_population: list[Individual] | None,
        n_triangles: int,
    ) -> list[Individual] | None:
        """Validates and deep-copies an optional initial population."""

        if initial_population is None:
            return None
        if not initial_population:
            raise ValueError("initial_population must not be empty when provided.")

        copied_population = copy.deepcopy(initial_population)
        for individual in copied_population:
            if len(individual) != n_triangles:
                raise ValueError(
                    "Each initial_population individual must have exactly "
                    "n_triangles triangles."
                )

        return copied_population

    @staticmethod
    def _count_changed_triangles(
        before_mutation: Individual,
        after_mutation: Individual,
    ) -> int:
        """Counts how many triangles changed after mutation."""

        if len(before_mutation) != len(after_mutation):
            return max(len(before_mutation), len(after_mutation))

        return sum(
            1
            for before_triangle, after_triangle in zip(
                before_mutation, after_mutation, strict=True
            )
            if before_triangle != after_triangle
        )

    def __init__(
        self,
        target: np.ndarray,
        fitness_function: FitnessFunction,
        population_size: int,
        generations: int,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
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
        n_triangles: int = population.N_TRIANGLES,
        adaptive_mutation: bool = False,
        mutation_rate_bounds: RateBounds | None = None,
        stagnation_window: int = 8,
        random_immigrants: int = 0,
        initial_population: list[Individual] | None = None,
        progress: bool = False,
        progress_interval: int = 1,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """
        Configures a genetic image approximation run.

        Args:
            target: RGB target image with shape ``(height, width, 3)``.
            fitness_function: Callable that evaluates ``(target, generated)`` arrays.
            population_size: Number of individuals evaluated each generation.
            generations: Number of generations to run.
            crossover_rate: Required when ``crossover_function`` is provided.
            mutation_rate: Required when ``mutation_function`` is provided.
            elitism: Number of best individuals copied unchanged per generation.
            selection_type: Parent selection strategy handled by ``selection``.
            logs: Whether to populate ``run_logs`` after ``run()``.
            crossover_function: Optional operator returning one child individual.
            mutation_function: Optional operator returning one mutated individual.
            evaluation_backend: ``sequential``, ``thread``, or ``process``.
            n_jobs: Optional worker count for thread or process evaluation.
            chunksize: Optional process-pool batch size.
            triangle_alpha_range: Inclusive triangle alpha range from 0 to 255.
            n_triangles: Number of triangles per individual.
            adaptive_mutation: Enables adaptive mutation-rate scheduling.
            mutation_rate_bounds: Optional adaptive mutation min/max rate.
            stagnation_window: Generations without improvement before mutation boost.
            random_immigrants: Number of random individuals injected each generation.
            initial_population: Optional seeded population for generation 0.
            progress: If ``True``, prints one-line generation summaries.
            progress_interval: Print frequency when ``progress`` is enabled.
            progress_callback: Optional callback receiving per-generation telemetry.

        Raises:
            ValueError: If dimensions, rates, alpha range, backend, worker
                settings, mutation controls, or process-mode pickling requirements
                are invalid.
        """

        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError("target must have shape (H, W, 3).")
        if population_size <= 0:
            raise ValueError("population_size must be greater than zero.")
        if generations <= 0:
            raise ValueError("generations must be greater than zero.")
        if not 0 <= elitism <= population_size:
            raise ValueError("elitism must be between 0 and population_size.")
        if n_triangles <= 0:
            raise ValueError("n_triangles must be greater than zero.")
        if random_immigrants < 0 or random_immigrants >= population_size:
            raise ValueError("random_immigrants must be in [0, population_size).")
        if stagnation_window <= 0:
            raise ValueError("stagnation_window must be greater than zero.")
        if progress_interval <= 0:
            raise ValueError("progress_interval must be greater than zero.")
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable when provided.")

        normalized_backend = evaluation.normalize_evaluation_backend(
            evaluation_backend
        )
        evaluation.validate_optional_positive_int(n_jobs, "n_jobs")
        evaluation.validate_optional_positive_int(chunksize, "chunksize")
        if normalized_backend == "process":
            evaluation.validate_process_fitness_function(fitness_function)

        self.target = target.astype(np.float32)
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.crossover_rate = self._resolve_operator_rate(
            "crossover_rate", crossover_rate, self.crossover_function
        )
        self.mutation_rate = self._resolve_operator_rate(
            "mutation_rate", mutation_rate, self.mutation_function
        )
        if adaptive_mutation and self.mutation_function is None:
            raise ValueError(
                "adaptive_mutation requires mutation_function to be provided."
            )
        self.adaptive_mutation = adaptive_mutation
        self.mutation_rate_bounds = self._validate_optional_rate_bounds(
            mutation_rate_bounds
        )
        if self.adaptive_mutation and self.mutation_rate_bounds is None:
            base = self.mutation_rate
            self.mutation_rate_bounds = (max(0.001, base * 0.25), min(1.0, base * 2.0))
        self.stagnation_window = stagnation_window
        self.random_immigrants = random_immigrants
        self.elitism = elitism
        self.selection_type = selection.normalize_selection_type(selection_type)
        self.logs = logs
        self.evaluation_backend = normalized_backend
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.triangle_alpha_range = population.validate_triangle_alpha_range(
            triangle_alpha_range
        )
        self.n_triangles = n_triangles
        self.initial_population = self._normalize_initial_population(
            initial_population,
            n_triangles=n_triangles,
        )
        self.progress = progress
        self.progress_interval = progress_interval
        self.progress_callback = progress_callback
        self.image_height, self.image_width = self.target.shape[:2]

        self.population: list[Individual] = []
        self.best_individual: Individual | None = None
        self.best_fitness = float("inf")
        self.history: list[float] = []
        self.run_logs: logs.RunLogs = {}
        self._current_mutation_rate = self.mutation_rate
        self._last_improvement_generation = 0

    def initialize(self) -> None:
        """Creates the initial population and resets run state."""

        if self.initial_population is None:
            self.population = population.create_population(
                population_size=self.population_size,
                n_triangles=self.n_triangles,
                image_width=self.image_width,
                image_height=self.image_height,
                triangle_alpha_range=self.triangle_alpha_range,
            )
        else:
            seeded = copy.deepcopy(self.initial_population[: self.population_size])
            if len(seeded) < self.population_size:
                seeded.extend(
                    population.create_population(
                        population_size=self.population_size - len(seeded),
                        n_triangles=self.n_triangles,
                        image_width=self.image_width,
                        image_height=self.image_height,
                        triangle_alpha_range=self.triangle_alpha_range,
                    )
                )
            self.population = seeded

        self.best_individual = None
        self.best_fitness = float("inf")
        self.history = []
        self.run_logs = {}
        self._current_mutation_rate = self.mutation_rate
        self._last_improvement_generation = 0

    def evaluate(self) -> list[float]:
        """Computes population fitness and updates the tracked best individual."""

        return self._evaluate_population()

    def _evaluate_population(
        self,
        executor: evaluation.Executor | None = None,
    ) -> list[float]:
        """Computes population fitness and updates global-best state."""

        if not self.population:
            raise ValueError(
                "Population is empty. Call initialize() before evaluate()."
            )

        fitness_values = self._compute_population_fitness(executor)

        for individual, fitness_value in zip(
            self.population, fitness_values, strict=True
        ):
            if fitness_value < self.best_fitness:
                self.best_fitness = fitness_value
                self.best_individual = copy.deepcopy(individual)

        return fitness_values

    def _compute_population_fitness(
        self,
        executor: evaluation.Executor | None = None,
    ) -> list[float]:
        """Dispatches population fitness computation to the configured backend."""

        if self.evaluation_backend == "sequential":
            return evaluation.compute_population_fitness_sequential(
                self.population,
                self.target,
                self.fitness_function,
                self.image_width,
                self.image_height,
            )

        if executor is None:
            with evaluation.create_evaluation_executor(
                self.evaluation_backend,
                self.n_jobs,
                self.target,
                self.fitness_function,
                self.image_width,
                self.image_height,
            ) as managed_executor:
                return evaluation.compute_population_fitness_with_executor(
                    managed_executor,
                    self.evaluation_backend,
                    self.population,
                    self.target,
                    self.fitness_function,
                    self.image_width,
                    self.image_height,
                    self.chunksize,
                )

        return evaluation.compute_population_fitness_with_executor(
            executor,
            self.evaluation_backend,
            self.population,
            self.target,
            self.fitness_function,
            self.image_width,
            self.image_height,
            self.chunksize,
        )

    def _update_mutation_rate(self, generation: int) -> None:
        """Updates the effective mutation rate for this generation."""

        if not self.adaptive_mutation or self.mutation_function is None:
            self._current_mutation_rate = self.mutation_rate
            return

        if self.mutation_rate_bounds is None:
            raise RuntimeError("mutation_rate_bounds must be set in adaptive mode.")

        lower_bound, upper_bound = self.mutation_rate_bounds
        progress = generation / max(1, self.generations - 1)
        scheduled_rate = upper_bound - ((upper_bound - lower_bound) * progress)

        if generation - self._last_improvement_generation >= self.stagnation_window:
            scheduled_rate = min(upper_bound, scheduled_rate * 1.35)

        self._current_mutation_rate = float(
            min(upper_bound, max(lower_bound, scheduled_rate))
        )

    def _inject_random_immigrants(self, next_population: list[Individual]) -> int:
        """Injects random individuals to preserve population diversity."""

        if self.random_immigrants == 0:
            return 0

        available_slots = self.population_size - len(next_population)
        if available_slots <= 0:
            return 0

        immigrant_count = min(self.random_immigrants, available_slots)
        next_population.extend(
            population.create_population(
                population_size=immigrant_count,
                n_triangles=self.n_triangles,
                image_width=self.image_width,
                image_height=self.image_height,
                triangle_alpha_range=self.triangle_alpha_range,
            )
        )
        return immigrant_count

    def select_parent(self, fitness_values: list[float]) -> Individual:
        """Selects one parent from the current population."""

        return selection.select_parent(
            self.population,
            fitness_values,
            selection_type=self.selection_type,
            tournament_size=3,
        )

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> Individual:
        """Applies the configured crossover function when one is provided."""

        if self.crossover_function is None:
            fallback_parent = parent1 if np.random.random() < 0.5 else parent2
            return copy.deepcopy(fallback_parent)

        child = self.crossover_function(parent1, parent2, self.crossover_rate)

        return copy.deepcopy(child)

    def mutate(self, individual: Individual) -> tuple[Individual, int]:
        """Applies the configured mutation function when one is provided."""

        if self.mutation_function is None:
            return individual, 0

        before_mutation = copy.deepcopy(individual)
        mutated_individual = self.mutation_function(
            individual,
            self._current_mutation_rate,
            self.image_width,
            self.image_height,
            self.triangle_alpha_range,
        )
        changed_triangles = self._count_changed_triangles(
            before_mutation=before_mutation,
            after_mutation=mutated_individual,
        )

        return mutated_individual, changed_triangles

    def _emit_progress(self, generation_log: logs.GenerationLog) -> None:
        """Emits optional live progress updates for notebook and CLI usage."""

        if self.progress_callback is not None:
            self.progress_callback(generation_log)

        if not self.progress:
            return
        if generation_log["generation"] % self.progress_interval != 0:
            return

        generation_index = generation_log["generation"] + 1
        print(
            f"[GA] gen {generation_index}/{self.generations} | "
            f"best={generation_log['global_best_fitness']:.6f} | "
            f"gen_best={generation_log['generation_best_fitness']:.6f} | "
            f"mut_rate={generation_log['mutation_rate_used']:.4f} | "
            f"mutated_offspring={generation_log['mutated_offspring']} | "
            f"mutated_triangles={generation_log['mutated_triangles']}"
        )

    def run(self) -> tuple[float, list[float]]:
        """
        Runs all generations and returns ``(best_fitness, global_best_history)``.

        When ``logs`` is enabled, ``run_logs`` is populated with generation
        metrics and the best individual configuration after the run finishes.
        """

        self.initialize()
        executor: evaluation.Executor | None = None
        generation_logs: list[logs.GenerationLog] = []

        try:
            if self.evaluation_backend != "sequential":
                executor = evaluation.create_evaluation_executor(
                    self.evaluation_backend,
                    self.n_jobs,
                    self.target,
                    self.fitness_function,
                    self.image_width,
                    self.image_height,
                )

            for generation in range(self.generations):
                previous_best = float(self.best_fitness)
                evaluation_started = time.perf_counter()
                fitness_values = self._evaluate_population(executor)
                evaluation_duration_seconds = (
                    time.perf_counter() - evaluation_started
                )
                ranked_indices = np.argsort(fitness_values)
                generation_best_fitness = float(fitness_values[int(ranked_indices[0])])
                global_best_fitness = float(self.best_fitness)

                if global_best_fitness < previous_best:
                    self._last_improvement_generation = generation

                self._update_mutation_rate(generation)
                self.history.append(global_best_fitness)
                offspring_created = 0
                mutated_offspring = 0
                mutated_triangles = 0
                immigrant_count = 0

                if generation != self.generations - 1:
                    next_population = [
                        copy.deepcopy(self.population[int(index)])
                        for index in ranked_indices[: self.elitism]
                    ]
                    immigrant_count = self._inject_random_immigrants(next_population)

                    while len(next_population) < self.population_size:
                        parent1 = self.select_parent(fitness_values)
                        parent2 = self.select_parent(fitness_values)
                        child = self.crossover(parent1, parent2)
                        child, changed_triangles = self.mutate(child)
                        offspring_created += 1
                        mutated_triangles += changed_triangles
                        if changed_triangles > 0:
                            mutated_offspring += 1
                        next_population.append(child)

                    self.population = next_population[: self.population_size]

                generation_log = logs.create_generation_log(
                    generation=generation,
                    generation_best_fitness=generation_best_fitness,
                    generation_mean_fitness=float(np.mean(fitness_values)),
                    global_best_fitness=global_best_fitness,
                    evaluation_backend=self.evaluation_backend,
                    n_jobs=self.n_jobs,
                    chunksize=self.chunksize,
                    evaluation_duration_seconds=(
                        evaluation_duration_seconds
                    ),
                    mutation_rate_used=float(self._current_mutation_rate),
                    offspring_created=offspring_created,
                    mutated_offspring=mutated_offspring,
                    mutated_triangles=mutated_triangles,
                    immigrant_count=immigrant_count,
                )

                if self.logs:
                    generation_logs.append(generation_log)

                self._emit_progress(generation_log)
        finally:
            if executor is not None:
                executor.shutdown()

        if self.best_individual is None:
            raise RuntimeError(
                "The genetic algorithm did not produce a best individual."
            )

        if self.logs:
            self.run_logs = logs.create_run_logs(
                generation_logs,
                float(self.best_fitness),
                self.best_individual,
            )

        return float(self.best_fitness), list(self.history)
