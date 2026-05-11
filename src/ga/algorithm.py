"""
Core GeneticAlgorithm class for triangle-based image approximation.

How the GA works
----------------
Each candidate solution (individual) is a list of Triangle objects.
When rendered, those triangles form a coloured image that the algorithm
tries to make look as close as possible to a target photo.

One generation of the GA:
  1. Evaluate   — render every individual and compute its RMSE vs. the target.
  2. Select     — pick pairs of parents from the population, biased toward
                  individuals with lower RMSE.
  3. Crossover  — combine each parent pair to produce one or more children.
  4. Mutate     — randomly perturb some children to maintain diversity.
  5. Replace    — the children become the next generation's population
                  (optionally keeping the best individuals unchanged via elitism).

Optional features
-----------------
  Elitism           : copy the N best individuals unchanged into each new
                      generation so the best solution never gets worse.
  Adaptive mutation : automatically raise the mutation rate when the search
                      stagnates, then lower it again as progress resumes.
  Random immigrants : inject a few fully random individuals each generation
                      to maintain diversity and help escape local optima.
  Seeded init       : initialise triangle colours from random target pixels
                      rather than fully random colours for a better start.
  Progress output   : print a one-line summary after each generation.
  Parallel eval     : use threads or processes to speed up fitness evaluation.
"""

import copy
import time
from collections.abc import Callable

import numpy as np

from .. import population
from . import evaluation, logs, selection

# ---------------------------------------------------------------------------
# Type aliases — give descriptive names to complex callable signatures
# ---------------------------------------------------------------------------

Individual = list[population.Triangle]

# A fitness function takes two RGB arrays (target, generated) → scalar
FitnessFunction = evaluation.FitnessFunction

# A crossover result can be one individual, a tuple, or a list of individuals
CrossoverResult = Individual | tuple[Individual, ...] | list[Individual]

# Crossover function signature: (parent1, parent2, rate) → child(ren)
CrossoverFunction = Callable[[Individual, Individual, float], CrossoverResult]

# Mutation function signature: (individual, rate, width, height, alpha_range) → individual
MutationFunction = Callable[
    [Individual, float, int, int, population.AlphaRange], Individual
]

# Union type for validation helpers
OperatorFunction = CrossoverFunction | MutationFunction

# Re-export for callers that import from this module
EvaluationBackend = evaluation.EvaluationBackend

# (lower_bound, upper_bound) for adaptive mutation rates
RateBounds = tuple[float, float]

# Progress callback receives the GenerationLog dict each generation
ProgressCallback = Callable[[logs.GenerationLog], None]


class GeneticAlgorithm:
    """
    Evolves a population of triangle-based image candidates toward lower RMSE.

    Each individual is a complete candidate image composed of ``n_triangles``
    triangles.  The algorithm maintains ``population_size`` individuals per
    generation, evaluates their rendered images against the target, and
    iteratively improves them through selection, crossover, and mutation.

    Usage example::

        ga = GeneticAlgorithm(
            target=target_array,
            fitness_function=fitness.compute_rmse,
            population_size=100,
            generations=200,
            crossover_function=cross_over.single_point_crossover,
            crossover_rate=0.9,
            mutation_function=mutate.random_triangle_mutation,
            mutation_rate=0.1,
            elitism=2,
            selection_type="tournament",
        )
        best_rmse, history = ga.run()
        best_image = ga.best_individual
    """

    # -----------------------------------------------------------------------
    # Private static helpers (validation / counting)
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_operator_rate(
        rate_name: str,
        rate: float | None,
        operator: OperatorFunction | None,
    ) -> float:
        """
        Resolves an operator rate to a float, validating consistency.

        Rules:
          - If the operator is None, the rate is not needed → return 0.0.
          - If the operator is set, the rate must also be provided.
          - The rate must be in [0.0, 1.0].

        Args:
            rate_name: Name of the rate for error messages (e.g. "crossover_rate").
            rate:      The raw rate value (may be None).
            operator:  The associated operator function (may be None).

        Returns:
            The validated float rate, or 0.0 if no operator is configured.

        Raises:
            ValueError: If the operator is set but the rate is missing, or if
                        the rate is outside [0, 1].
        """

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
        """
        Validates the optional (lower, upper) bounds for adaptive mutation.

        Both bounds must be in [0, 1] and lower ≤ upper.

        Args:
            mutation_rate_bounds: The (min, max) tuple or None.

        Returns:
            The validated tuple (as floats), or None if not provided.

        Raises:
            ValueError: If the bounds are malformed or out of range.
        """

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
        """
        Validates and deep-copies an externally supplied initial population.

        Deep-copying protects the caller's list from being mutated by the GA.
        Validates that every individual has exactly ``n_triangles`` triangles
        so mismatches are caught at configuration time.

        Args:
            initial_population: Caller-supplied population, or None.
            n_triangles:        Expected number of triangles per individual.

        Returns:
            A deep copy of the population, or None if not provided.

        Raises:
            ValueError: If the list is empty or any individual has wrong length.
        """

        if initial_population is None:
            return None
        if not initial_population:
            raise ValueError("initial_population must not be empty when provided.")

        # Deep copy so that mutations during the run do not affect the caller's data
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
        """
        Counts how many triangles actually changed after a mutation call.

        Used for logging purposes: tracks how many triangles were modified
        vs. how many were skipped by the mutation operator.

        Args:
            before_mutation: Deep copy of the individual before mutation.
            after_mutation:  The individual after mutation was applied.

        Returns:
            Number of triangles that differ between before and after.
        """

        if len(before_mutation) != len(after_mutation):
            # Length mismatch — treat all as changed
            return max(len(before_mutation), len(after_mutation))

        return sum(
            1
            for before_triangle, after_triangle in zip(
                before_mutation, after_mutation, strict=True
            )
            if before_triangle != after_triangle
        )

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

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
        seeded: bool = False,
        progress: bool = False,
        progress_interval: int = 1,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """
        Configures a genetic image approximation run.

        Args:
            target:               RGB target image with shape (height, width, 3).
            fitness_function:     Callable: (target_array, generated_array) → float.
            population_size:      Number of individuals evaluated each generation.
            generations:          Number of generations to run.
            crossover_rate:       Probability of crossover per pair (0–1).
                                  Required when crossover_function is provided.
            mutation_rate:        Per-triangle mutation probability (0–1).
                                  Required when mutation_function is provided.
            elitism:              Number of best individuals copied unchanged to
                                  the next generation (0 = no elitism).
            selection_type:       Parent selection strategy: "tournament",
                                  "ranking", or "roulette".
            logs:                 If True, populate run_logs after run() finishes.
            crossover_function:   Operator: (p1, p2, rate) → child or children.
            mutation_function:    Operator: (individual, rate, w, h, alpha) → individual.
            evaluation_backend:   "sequential", "thread", or "process".
            n_jobs:               Worker count for thread/process backends.
            chunksize:            Batch size for the process pool.
            triangle_alpha_range: Inclusive (min, max) alpha values for triangles.
            n_triangles:          Triangles per individual (default 100).
            adaptive_mutation:    Automatically adjust mutation rate based on
                                  whether the search is improving or stagnating.
            mutation_rate_bounds: (min, max) rates for adaptive scheduling.
                                  Auto-derived from mutation_rate if not given.
            stagnation_window:    Generations without improvement before the
                                  adaptive scheduler boosts the mutation rate.
            random_immigrants:    Random individuals injected each generation
                                  to maintain diversity (0 = disabled).
            initial_population:   Optional explicit generation-0 population.
                                  Useful for warm-starting from a prior run.
            seeded:               If True, initialise triangle colours by sampling
                                  random pixels from the target image.
            progress:             Print a one-line summary after each generation.
            progress_interval:    Print every N-th generation (default 1 = every gen).
            progress_callback:    Optional callable receiving each GenerationLog.

        Raises:
            ValueError: If any parameter fails validation.
        """

        # --- Shape / range validation ---
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

        # --- Backend validation ---
        normalized_backend = evaluation.normalize_evaluation_backend(
            evaluation_backend
        )
        evaluation.validate_optional_positive_int(n_jobs, "n_jobs")
        evaluation.validate_optional_positive_int(chunksize, "chunksize")

        # Process backend requires picklable fitness functions because jobs
        # are serialised and sent across process boundaries
        if normalized_backend == "process":
            evaluation.validate_process_fitness_function(fitness_function)

        # --- Store configuration ---
        # Convert target to float32 once here; all fitness computations use float32
        self.target = target.astype(np.float32)
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function

        # Resolve rates: returns 0.0 if the corresponding operator is None
        self.crossover_rate = self._resolve_operator_rate(
            "crossover_rate", crossover_rate, self.crossover_function
        )
        self.mutation_rate = self._resolve_operator_rate(
            "mutation_rate", mutation_rate, self.mutation_function
        )

        # Adaptive mutation requires a mutation function to be present
        if adaptive_mutation and self.mutation_function is None:
            raise ValueError(
                "adaptive_mutation requires mutation_function to be provided."
            )
        self.adaptive_mutation = adaptive_mutation
        self.mutation_rate_bounds = self._validate_optional_rate_bounds(
            mutation_rate_bounds
        )

        # Auto-derive bounds from the base rate when adaptive mode is on
        # but no explicit bounds were provided:
        #   lower = 25 % of base rate (minimum exploration floor)
        #   upper = 200 % of base rate (maximum boost during stagnation)
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
        self.seeded = seeded
        self.progress = progress
        self.progress_interval = progress_interval
        self.progress_callback = progress_callback

        # Derive canvas dimensions from the target shape
        self.image_height, self.image_width = self.target.shape[:2]

        # --- Mutable run state (reset by initialize()) ---
        self.population: list[Individual] = []
        self.best_individual: Individual | None = None
        self.best_fitness = float("inf")      # starts at infinity, only decreases
        self.history: list[float] = []        # global_best_fitness per generation
        self.run_logs: logs.RunLogs = {}      # populated after run() when logs=True
        self._current_mutation_rate = self.mutation_rate  # may change in adaptive mode
        self._last_improvement_generation = 0             # used by stagnation detection

    # -----------------------------------------------------------------------
    # Public lifecycle methods
    # -----------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Creates the generation-0 population and resets all run state.

        If an ``initial_population`` was supplied at construction time, it is
        used directly (padded with random individuals if it is smaller than
        ``population_size``).  Otherwise a fresh random population is created.

        Calling ``initialize()`` again resets the run completely — useful for
        running the GA multiple times with the same configuration.
        """

        if self.initial_population is None:
            # No warm-start — generate a fully random population
            self.population = population.create_population(
                population_size=self.population_size,
                n_triangles=self.n_triangles,
                image_width=self.image_width,
                image_height=self.image_height,
                triangle_alpha_range=self.triangle_alpha_range,
                target=self.target,
                seeded=self.seeded,
            )
        else:
            # Use the provided population as the starting point
            seeded = copy.deepcopy(self.initial_population[: self.population_size])

            # If the provided population is smaller, pad it with random individuals
            if len(seeded) < self.population_size:
                seeded.extend(
                    population.create_population(
                        population_size=self.population_size - len(seeded),
                        n_triangles=self.n_triangles,
                        image_width=self.image_width,
                        image_height=self.image_height,
                        triangle_alpha_range=self.triangle_alpha_range,
                        target=self.target,
                        seeded=self.seeded,
                    )
                )
            self.population = seeded

        # Reset all tracked state so a fresh run starts clean
        self.best_individual = None
        self.best_fitness = float("inf")
        self.history = []
        self.run_logs = {}
        self._current_mutation_rate = self.mutation_rate
        self._last_improvement_generation = 0

    def evaluate(self) -> list[float]:
        """
        Computes fitness for all individuals in the current population.

        Also updates ``best_individual`` and ``best_fitness`` if any individual
        in the current population beats the previously tracked best.

        Returns:
            List of fitness floats in population order (lower = better).
        """
        return self._evaluate_population()

    # -----------------------------------------------------------------------
    # Private helpers used inside run()
    # -----------------------------------------------------------------------

    def _evaluate_population(
        self,
        executor: evaluation.Executor | None = None,
    ) -> list[float]:
        """
        Evaluates the population and updates the tracked global best.

        Args:
            executor: An active executor for non-sequential backends, or None.

        Returns:
            List of fitness values in population order.
        """

        if not self.population:
            raise ValueError(
                "Population is empty. Call initialize() before evaluate()."
            )

        fitness_values = self._compute_population_fitness(executor)

        # Scan all individuals and update the global best if a new one is found
        for individual, fitness_value in zip(
            self.population, fitness_values, strict=True
        ):
            if fitness_value < self.best_fitness:
                self.best_fitness = fitness_value
                # Deep copy so later mutations don't corrupt the tracked best
                self.best_individual = copy.deepcopy(individual)

        return fitness_values

    def _compute_population_fitness(
        self,
        executor: evaluation.Executor | None = None,
    ) -> list[float]:
        """
        Dispatches fitness computation to the configured backend.

        When called from run(), the executor is passed in directly to avoid
        creating and destroying a new pool for every generation.  The public
        evaluate() method calls this without an executor, which creates a
        short-lived pool (or runs sequentially).

        Args:
            executor: Pre-created executor, or None to create/use inline.

        Returns:
            List of fitness values for the current population.
        """

        if self.evaluation_backend == "sequential":
            # No parallelism — evaluate directly on the calling thread
            return evaluation.compute_population_fitness_sequential(
                self.population,
                self.target,
                self.fitness_function,
                self.image_width,
                self.image_height,
            )

        if executor is None:
            # Create a short-lived pool just for this one evaluation call
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

        # Reuse the long-lived executor passed in from run()
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
        """
        Updates the effective mutation rate for this generation.

        In non-adaptive mode the rate stays constant at self.mutation_rate.

        In adaptive mode the rate is scheduled as a linear decay from
        upper_bound (generation 0) down to lower_bound (last generation).
        If the search has not improved for ``stagnation_window`` generations,
        the rate is boosted by 35 % to help escape the local optimum.
        The final rate is always clamped to [lower_bound, upper_bound].

        Args:
            generation: 0-indexed current generation number.
        """

        if not self.adaptive_mutation or self.mutation_function is None:
            # No adaptation — keep the rate fixed
            self._current_mutation_rate = self.mutation_rate
            return

        if self.mutation_rate_bounds is None:
            raise RuntimeError("mutation_rate_bounds must be set in adaptive mode.")

        lower_bound, upper_bound = self.mutation_rate_bounds

        # Linear decay from upper_bound at gen 0 to lower_bound at last gen.
        # progress ∈ [0, 1]: 0 = start of run, 1 = end of run.
        progress = generation / max(1, self.generations - 1)
        scheduled_rate = upper_bound - ((upper_bound - lower_bound) * progress)

        # Stagnation boost: if there has been no improvement for
        # stagnation_window generations, increase the rate by 35 %
        if generation - self._last_improvement_generation >= self.stagnation_window:
            scheduled_rate = min(upper_bound, scheduled_rate * 1.35)

        # Clamp to the configured bounds and store for use this generation
        self._current_mutation_rate = float(
            min(upper_bound, max(lower_bound, scheduled_rate))
        )

    def _inject_random_immigrants(self, next_population: list[Individual]) -> int:
        """
        Injects fully random individuals into the nascent next generation.

        Random immigrants prevent premature convergence by continuously
        introducing new genetic material.  They replace slots that would
        otherwise be filled by crossover/mutation offspring.

        Only injects up to the number of available empty slots so the
        next generation never exceeds ``population_size``.

        Args:
            next_population: The partially-built next generation list.
                             Modified in place.

        Returns:
            Number of immigrants actually injected.
        """

        if self.random_immigrants == 0:
            return 0

        # Don't overflow the population beyond its configured size
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

    def select_parents(
        self,
        fitness_values: list[float],
    ) -> tuple[Individual, Individual]:
        """
        Selects a pair of parents for crossover.

        The default implementation selects both parents independently using
        the configured selection strategy.  Subclasses can override this
        method to implement restricted mating — e.g. choosing parent2 based
        on its genetic distance to parent1.

        Args:
            fitness_values: Current generation fitness values (lower = better).

        Returns:
            Tuple of (parent1, parent2).
        """

        return self.select_parent(fitness_values), self.select_parent(fitness_values)

    def select_parent(self, fitness_values: list[float]) -> Individual:
        """
        Selects one parent from the current population using the configured strategy.

        Delegates to the selection module so the GA class itself does not
        need to know the details of each selection algorithm.

        Args:
            fitness_values: Current generation fitness values (lower = better).

        Returns:
            One individual chosen according to the selection strategy.
        """

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
    ) -> list[Individual]:
        """
        Applies the configured crossover operator and normalises its output.

        The crossover operator may return one child, a tuple of children, or
        a list of children.  This method normalises all three forms to a plain
        list so the main loop can iterate uniformly.

        If no crossover function was configured, one parent is cloned at random
        (pure reproduction — the child is an exact copy of a parent).

        Args:
            parent1: First selected parent individual.
            parent2: Second selected parent individual.

        Returns:
            A list of one or more child individuals (deep-copied).

        Raises:
            ValueError: If the crossover function returns no children.
        """

        if self.crossover_function is None:
            # No crossover configured — clone one parent at random
            fallback_parent = parent1 if np.random.random() < 0.5 else parent2
            return [copy.deepcopy(fallback_parent)]

        crossover_result = self.crossover_function(
            parent1,
            parent2,
            self.crossover_rate,
        )

        # Normalise the return value to a flat list of individuals:
        # - tuple → convert to list
        # - list of Triangles (one child) → wrap in outer list
        # - list of lists (multiple children) → use as-is
        if isinstance(crossover_result, tuple):
            children = list(crossover_result)
        elif crossover_result and isinstance(
            crossover_result[0],
            population.Triangle,
        ):
            # The operator returned a single individual (flat list of Triangles)
            children = [crossover_result]
        else:
            children = list(crossover_result)

        if not children:
            raise ValueError("crossover_function must return at least one child.")

        return copy.deepcopy(children)

    def mutate(self, individual: Individual) -> tuple[Individual, int]:
        """
        Applies the configured mutation function to one individual.

        Returns both the (potentially modified) individual and a count of
        how many triangles were actually changed, which is used for logging.

        If no mutation function was configured, the individual is returned
        unchanged with a change count of 0.

        Args:
            individual: The child individual to mutate.

        Returns:
            (mutated_individual, changed_triangle_count)
        """

        if self.mutation_function is None:
            return individual, 0

        # Snapshot before mutation to count how many triangles changed
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
        """
        Optionally prints a generation summary and/or calls the progress callback.

        Controls whether output is emitted based on ``progress``,
        ``progress_interval``, and ``progress_callback`` settings.

        Args:
            generation_log: The log entry for the just-completed generation.
        """

        # Always fire the callback if one was registered, regardless of interval
        if self.progress_callback is not None:
            self.progress_callback(generation_log)

        # Only print if progress output is enabled
        if not self.progress:
            return

        # Respect the interval — only print every N-th generation
        if generation_log["generation"] % self.progress_interval != 0:
            return

        # 1-indexed generation number for human-readable output
        generation_index = generation_log["generation"] + 1
        print(
            f"[GA] gen {generation_index}/{self.generations} | "
            f"best={generation_log['global_best_fitness']:.6f} | "
            f"gen_best={generation_log['generation_best_fitness']:.6f} | "
            f"mut_rate={generation_log['mutation_rate_used']:.4f} | "
            f"mutated_offspring={generation_log['mutated_offspring']} | "
            f"mutated_triangles={generation_log['mutated_triangles']}"
        )

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def run(self) -> tuple[float, list[float]]:
        """
        Runs the full GA and returns the best fitness and convergence history.

        Execution flow:
          1. initialize() — create generation-0 population, reset state.
          2. Optionally open a long-lived evaluation executor.
          3. For each generation:
             a. Evaluate the current population.
             b. Record and update the adaptive mutation rate.
             c. Apply elitism (copy best N individuals to next generation).
             d. Inject random immigrants if configured.
             e. Fill the rest of the next generation via selection, crossover,
                and mutation until population_size is reached.
             f. Log and emit progress.
          4. Shut down the executor.
          5. If logging is enabled, assemble run_logs.
          6. Return (best_fitness, history).

        After run() returns:
          - ``ga.best_individual`` holds the best triangle list found.
          - ``ga.history`` holds the global best RMSE after each generation.
          - ``ga.run_logs`` holds detailed per-generation telemetry (if logs=True).

        Returns:
            (best_fitness, history):
              best_fitness — lowest RMSE achieved across all generations.
              history      — list of global-best RMSE values, one per generation.

        Raises:
            RuntimeError: If the algorithm completes without a best individual
                          (should never happen under normal operation).
        """

        self.initialize()

        # Open the executor once at the start of the run so it is reused
        # across all generations (avoiding repeated pool startup overhead).
        # For the sequential backend, executor stays None.
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
                # Track the previous best so we can detect improvement
                previous_best = float(self.best_fitness)

                # --- Step 1: Evaluate current population ---
                evaluation_started = time.perf_counter()
                fitness_values = self._evaluate_population(executor)
                evaluation_duration_seconds = (
                    time.perf_counter() - evaluation_started
                )

                # Sort indices from best (lowest RMSE) to worst (highest RMSE)
                ranked_indices = np.argsort(fitness_values)
                generation_best_fitness = float(fitness_values[int(ranked_indices[0])])
                global_best_fitness = float(self.best_fitness)

                # Record the generation in which improvement last occurred
                # (used by the adaptive mutation scheduler)
                if global_best_fitness < previous_best:
                    self._last_improvement_generation = generation

                # --- Step 2: Update adaptive mutation rate ---
                self._update_mutation_rate(generation)

                # Append the current global best to the convergence history
                self.history.append(global_best_fitness)

                # Counters reset each generation for per-generation logging
                offspring_created = 0
                mutated_offspring = 0
                mutated_triangles = 0
                immigrant_count = 0

                # --- Step 3: Build the next generation ---
                # Skip for the final generation since we don't need a "next" pop
                if generation != self.generations - 1:

                    # Elitism: copy the best `elitism` individuals unchanged.
                    # They keep their position at the front of the next population.
                    next_population = [
                        copy.deepcopy(self.population[int(index)])
                        for index in ranked_indices[: self.elitism]
                    ]

                    # Inject random immigrants to maintain diversity
                    immigrant_count = self._inject_random_immigrants(next_population)

                    # Fill remaining slots with crossover + mutation offspring
                    while len(next_population) < self.population_size:
                        parent1, parent2 = self.select_parents(fitness_values)
                        children = self.crossover(parent1, parent2)

                        for child in children:
                            # Stop adding children once the population is full
                            if len(next_population) >= self.population_size:
                                break
                            child, changed_triangles = self.mutate(child)
                            offspring_created += 1
                            mutated_triangles += changed_triangles
                            if changed_triangles > 0:
                                mutated_offspring += 1
                            next_population.append(child)

                    # Truncate to exactly population_size in case crossover
                    # produced one extra child that pushed us over
                    self.population = next_population[: self.population_size]

                # --- Step 4: Log and emit progress ---
                generation_log = logs.create_generation_log(
                    generation=generation,
                    generation_best_fitness=generation_best_fitness,
                    generation_mean_fitness=float(np.mean(fitness_values)),
                    global_best_fitness=global_best_fitness,
                    evaluation_backend=self.evaluation_backend,
                    n_jobs=self.n_jobs,
                    chunksize=self.chunksize,
                    evaluation_duration_seconds=evaluation_duration_seconds,
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
            # Always shut down the executor, even if an exception occurred
            if executor is not None:
                executor.shutdown()

        if self.best_individual is None:
            raise RuntimeError(
                "The genetic algorithm did not produce a best individual."
            )

        # Assemble the detailed run logs if logging was requested
        if self.logs:
            self.run_logs = logs.create_run_logs(
                generation_logs,
                float(self.best_fitness),
                self.best_individual,
            )

        return float(self.best_fitness), list(self.history)

    # -----------------------------------------------------------------------
    # Introspection helpers
    # -----------------------------------------------------------------------

    def params_dict(self) -> dict:
        """
        Returns a JSON-serialisable dict of all configuration parameters.

        Used by the results module to record what settings produced a given
        run so results can be compared across experiments.

        Returns:
            Dict mapping parameter names to their values.
        """

        return {
            "population_size":       self.population_size,
            "generations":           self.generations,
            "n_triangles":           self.n_triangles,
            "crossover_rate":        self.crossover_rate,
            "mutation_rate":         self.mutation_rate,
            "mutation_function":     getattr(self.mutation_function, "__name__", None),
            "crossover_function":    getattr(self.crossover_function, "__name__", None),
            "elitism":               self.elitism,
            "selection_type":        self.selection_type,
            "adaptive_mutation":     self.adaptive_mutation,
            "mutation_rate_bounds":  self.mutation_rate_bounds,
            "stagnation_window":     self.stagnation_window,
            "random_immigrants":     self.random_immigrants,
            "local_search_steps":    0,   # kept for backwards compatibility with saved JSONs
            "max_edge_length":       None,
            "triangle_alpha_range":  list(self.triangle_alpha_range),
            "evaluation_backend":    self.evaluation_backend,
            "n_jobs":                self.n_jobs,
            "initialization":        "seeded" if self.seeded else "random",
        }
