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
MutationFunction = Callable[[Individual, float, int, int], Individual]
OperatorFunction = CrossoverFunction | MutationFunction
EvaluationBackend = evaluation.EvaluationBackend


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

        Raises:
            ValueError: If dimensions, rates, backend, worker settings, or
                process-mode fitness pickling requirements are invalid.
        """

        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError("target must have shape (H, W, 3).")
        if population_size <= 0:
            raise ValueError("population_size must be greater than zero.")
        if generations <= 0:
            raise ValueError("generations must be greater than zero.")
        if not 0 <= elitism <= population_size:
            raise ValueError("elitism must be between 0 and population_size.")

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
        self.elitism = elitism
        self.selection_type = selection.normalize_selection_type(selection_type)
        self.logs = logs
        self.evaluation_backend = normalized_backend
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.image_height, self.image_width = self.target.shape[:2]

        self.population: list[Individual] = []
        self.best_individual: Individual | None = None
        self.best_fitness = float("inf")
        self.history: list[float] = []
        self.run_logs: logs.RunLogs = {}

    def initialize(self) -> None:
        """Creates the initial population and resets run state."""

        self.population = population.create_population(
            population_size=self.population_size,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        self.best_individual = None
        self.best_fitness = float("inf")
        self.history = []
        self.run_logs = {}

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

    def mutate(self, individual: Individual) -> Individual:
        """Applies the configured mutation function when one is provided."""

        if self.mutation_function is None:
            return individual

        return self.mutation_function(
            individual,
            self.mutation_rate,
            self.image_width,
            self.image_height,
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
                evaluation_started = time.perf_counter()
                fitness_values = self._evaluate_population(executor)
                evaluation_duration_seconds = (
                    time.perf_counter() - evaluation_started
                )
                ranked_indices = np.argsort(fitness_values)
                generation_best_fitness = float(fitness_values[int(ranked_indices[0])])
                global_best_fitness = float(self.best_fitness)

                self.history.append(global_best_fitness)

                if self.logs:
                    generation_logs.append(
                        logs.create_generation_log(
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
                        )
                    )

                if generation == self.generations - 1:
                    break

                next_population = [
                    copy.deepcopy(self.population[int(index)])
                    for index in ranked_indices[: self.elitism]
                ]

                while len(next_population) < self.population_size:
                    parent1 = self.select_parent(fitness_values)
                    parent2 = self.select_parent(fitness_values)
                    child = self.crossover(parent1, parent2)
                    next_population.append(self.mutate(child))

                self.population = next_population[: self.population_size]
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
