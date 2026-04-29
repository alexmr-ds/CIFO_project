"""Implements a Hybrid PSO-GA for triangle-based image approximation.

Each particle is a triangle individual. PSO velocity updates guide particles
toward personal and global bests in the continuous attribute space; GA
crossover recombines structure between particles; GA mutation injects diversity.
"""

import copy
from dataclasses import dataclass

import numpy as np

from .. import population
from . import evaluation

Individual = list[population.Triangle]
FitnessFunction = evaluation.FitnessFunction

_ATTRS_PER_TRIANGLE = 9  # x1, y1, x2, y2, x3, y3, r, g, b  (alpha fixed = 255)


@dataclass(frozen=True)
class HybridPSOGAResult:
    """Captures the output of one HybridPSOGA run."""

    best_fitness: float
    history: list[float]
    best_individual: Individual


def _to_vector(individual: Individual) -> np.ndarray:
    """Flattens an Individual into a 1-D float64 array."""

    return np.array(
        [
            [t.x1, t.y1, t.x2, t.y2, t.x3, t.y3, t.r, t.g, t.b]
            for t in individual
        ],
        dtype=np.float64,
    ).ravel()


def _to_individual(
    vector: np.ndarray,
    n_triangles: int,
    image_width: int,
    image_height: int,
) -> Individual:
    """Converts a flat float64 array back to an Individual (triangles are opaque)."""

    matrix = vector.reshape(n_triangles, _ATTRS_PER_TRIANGLE)
    result: Individual = []
    for row in matrix:
        result.append(
            population.Triangle(
                x1=int(np.clip(round(row[0]), 0, image_width - 1)),
                y1=int(np.clip(round(row[1]), 0, image_height - 1)),
                x2=int(np.clip(round(row[2]), 0, image_width - 1)),
                y2=int(np.clip(round(row[3]), 0, image_height - 1)),
                x3=int(np.clip(round(row[4]), 0, image_width - 1)),
                y3=int(np.clip(round(row[5]), 0, image_height - 1)),
                r=int(np.clip(round(row[6]), 0, 255)),
                g=int(np.clip(round(row[7]), 0, 255)),
                b=int(np.clip(round(row[8]), 0, 255)),
                a=255,
            )
        )
    return result


class HybridPSOGA:
    """Hybrid Particle Swarm Optimization + Genetic Algorithm.

    PSO moves each particle (individual) through the continuous triangle
    attribute space toward its personal best and the global best.
    After each PSO step, optional GA crossover recombines a particle with a
    randomly chosen partner, and optional GA mutation perturbs individual
    triangles to escape local optima.

    Inertia weight decays linearly from ``inertia`` to ``inertia_min`` over
    the run, shifting from exploration to exploitation as generations progress.
    """

    def __init__(
        self,
        target: np.ndarray,
        fitness_function: FitnessFunction,
        population_size: int,
        generations: int,
        n_triangles: int = 100,
        inertia: float = 0.9,
        inertia_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        crossover_function=None,
        crossover_rate: float = 0.7,
        mutation_function=None,
        mutation_rate: float = 0.1,
        local_search_steps: int = 0,
        max_edge_length: int | None = None,
        initial_population: list[Individual] | None = None,
        evaluation_backend: str = "sequential",
        n_jobs: int | None = None,
        chunksize: int | None = None,
        progress: bool = False,
        progress_interval: int = 10,
    ) -> None:
        """
        Args:
            target: RGB target image with shape (H, W, 3).
            fitness_function: Callable evaluating (target, generated) → float.
            population_size: Number of particles.
            generations: Number of iterations.
            n_triangles: Triangles per individual.
            inertia: Initial inertia weight (controls momentum).
            inertia_min: Inertia floor reached by the final generation.
            c1: Cognitive coefficient (attraction to personal best).
            c2: Social coefficient (attraction to global best).
            crossover_function: Optional GA crossover operator.
            crossover_rate: Probability a particle crosses with a random partner.
            mutation_function: Optional GA mutation operator.
            mutation_rate: Mutation probability passed to the operator.
            local_search_steps: Hill-climbing steps on global best each generation.
            max_edge_length: If set, no triangle edge may exceed this pixel length.
            initial_population: Optional pre-built swarm (e.g. from a prior GA run).
            evaluation_backend: ``sequential``, ``thread``, or ``process``.
            n_jobs: Worker count for parallel backends.
            chunksize: Process-pool batch size.
            progress: Print one-line progress each ``progress_interval`` generations.
            progress_interval: Print frequency when progress is enabled.
        """

        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError("target must have shape (H, W, 3).")
        if population_size <= 0:
            raise ValueError("population_size must be greater than zero.")
        if generations <= 0:
            raise ValueError("generations must be greater than zero.")
        if n_triangles <= 0:
            raise ValueError("n_triangles must be greater than zero.")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if local_search_steps < 0:
            raise ValueError("local_search_steps must be non-negative.")
        if progress_interval <= 0:
            raise ValueError("progress_interval must be greater than zero.")

        self.target = target.astype(np.float32)
        self.target_uint8 = target.astype(np.uint8)
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.n_triangles = n_triangles
        self.inertia = inertia
        self.inertia_min = inertia_min
        self.c1 = c1
        self.c2 = c2
        self.crossover_function = crossover_function
        self.crossover_rate = crossover_rate
        self.mutation_function = mutation_function
        self.mutation_rate = mutation_rate
        self.local_search_steps = local_search_steps
        self.max_edge_length = max_edge_length
        self.initial_population = initial_population
        self.evaluation_backend = evaluation.normalize_evaluation_backend(
            evaluation_backend
        )
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.progress = progress
        self.progress_interval = progress_interval
        self.image_height, self.image_width = self.target.shape[:2]

        dims = n_triangles * _ATTRS_PER_TRIANGLE
        tile = [
            self.image_width - 1, self.image_height - 1,
            self.image_width - 1, self.image_height - 1,
            self.image_width - 1, self.image_height - 1,
            255, 255, 255,
        ]
        self._lower = np.zeros(dims, dtype=np.float64)
        self._upper = np.tile(tile, n_triangles).astype(np.float64)
        self._v_max = (self._upper - self._lower) * 0.15

    def _decode(self, vector: np.ndarray) -> Individual:
        """Converts a position vector to an Individual, clamping edges if needed."""

        ind = _to_individual(vector, self.n_triangles, self.image_width, self.image_height)
        if self.max_edge_length is not None:
            for triangle in ind:
                population.clamp_triangle_edges(
                    triangle, self.max_edge_length, self.image_width, self.image_height
                )
        return ind

    def _local_search(
        self, individual: Individual, fitness_value: float
    ) -> tuple[Individual, float]:
        """Hill-climbs on one individual, accepting only improvements."""

        current = copy.deepcopy(individual)
        current_fitness = float(fitness_value)
        delta = 20

        for _ in range(self.local_search_steps):
            candidate = copy.deepcopy(current)
            t = candidate[int(np.random.randint(0, len(candidate)))]
            t.x1 = max(0, min(self.image_width - 1, t.x1 + int(np.random.randint(-delta, delta + 1))))
            t.y1 = max(0, min(self.image_height - 1, t.y1 + int(np.random.randint(-delta, delta + 1))))
            t.x2 = max(0, min(self.image_width - 1, t.x2 + int(np.random.randint(-delta, delta + 1))))
            t.y2 = max(0, min(self.image_height - 1, t.y2 + int(np.random.randint(-delta, delta + 1))))
            t.x3 = max(0, min(self.image_width - 1, t.x3 + int(np.random.randint(-delta, delta + 1))))
            t.y3 = max(0, min(self.image_height - 1, t.y3 + int(np.random.randint(-delta, delta + 1))))
            t.r = max(0, min(255, t.r + int(np.random.randint(-30, 31))))
            t.g = max(0, min(255, t.g + int(np.random.randint(-30, 31))))
            t.b = max(0, min(255, t.b + int(np.random.randint(-30, 31))))
            candidate_fitness = evaluation.compute_individual_fitness(
                candidate, self.target, self.fitness_function,
                self.image_width, self.image_height,
            )
            if candidate_fitness < current_fitness:
                current = candidate
                current_fitness = candidate_fitness

        return current, current_fitness

    def run(self) -> HybridPSOGAResult:
        """Runs the hybrid PSO-GA and returns the best result."""

        if self.initial_population is not None:
            base = self.initial_population[:self.population_size]
            init_pop = [copy.deepcopy(ind) for ind in base]
            while len(init_pop) < self.population_size:
                init_pop.extend(copy.deepcopy(base[:self.population_size - len(init_pop)]))
        else:
            init_pop = population.create_target_seeded_population(
                target=self.target_uint8,
                population_size=self.population_size,
                n_triangles=self.n_triangles,
                image_width=self.image_width,
                image_height=self.image_height,
            )

        positions = np.array([_to_vector(ind) for ind in init_pop])
        velocities = np.zeros_like(positions)
        pbest_positions = positions.copy()
        pbest_fitness = np.full(self.population_size, np.inf)

        gbest_position: np.ndarray | None = None
        gbest_fitness = float("inf")
        gbest_individual: Individual | None = None
        history: list[float] = []

        executor = None
        try:
            if self.evaluation_backend != "sequential":
                executor = evaluation.create_evaluation_executor(
                    self.evaluation_backend, self.n_jobs, self.target,
                    self.fitness_function, self.image_width, self.image_height,
                )

            for generation in range(self.generations):
                current_individuals = [
                    self._decode(positions[i])
                    for i in range(self.population_size)
                ]

                if self.evaluation_backend == "sequential":
                    fitness_values = evaluation.compute_population_fitness_sequential(
                        current_individuals, self.target, self.fitness_function,
                        self.image_width, self.image_height,
                    )
                else:
                    fitness_values = evaluation.compute_population_fitness_with_executor(
                        executor, self.evaluation_backend, current_individuals,
                        self.target, self.fitness_function,
                        self.image_width, self.image_height, self.chunksize,
                    )

                for i, fit in enumerate(fitness_values):
                    if fit < pbest_fitness[i]:
                        pbest_fitness[i] = fit
                        pbest_positions[i] = positions[i].copy()
                    if fit < gbest_fitness:
                        gbest_fitness = fit
                        gbest_position = positions[i].copy()
                        gbest_individual = copy.deepcopy(current_individuals[i])

                if self.local_search_steps > 0 and gbest_individual is not None:
                    improved, improved_fitness = self._local_search(gbest_individual, gbest_fitness)
                    if improved_fitness < gbest_fitness:
                        gbest_fitness = improved_fitness
                        gbest_individual = improved
                        gbest_position = _to_vector(gbest_individual)

                history.append(float(gbest_fitness))

                if self.progress and generation % self.progress_interval == 0:
                    print(
                        f"[PSO-GA] gen {generation + 1}/{self.generations} | "
                        f"best={gbest_fitness:.6f} | "
                        f"inertia={self.inertia - (self.inertia - self.inertia_min) * (generation / max(1, self.generations - 1)):.3f}"
                    )

                if generation == self.generations - 1:
                    break

                inertia = self.inertia - (self.inertia - self.inertia_min) * (
                    generation / max(1, self.generations - 1)
                )

                for i in range(self.population_size):
                    r1 = np.random.random(len(positions[i]))
                    r2 = np.random.random(len(positions[i]))

                    cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                    social = self.c2 * r2 * (gbest_position - positions[i])
                    velocities[i] = inertia * velocities[i] + cognitive + social
                    velocities[i] = np.clip(velocities[i], -self._v_max, self._v_max)
                    positions[i] = np.clip(positions[i] + velocities[i], self._lower, self._upper)

                    # GA crossover: recombine with a random other particle
                    if (
                        self.crossover_function is not None
                        and np.random.random() < self.crossover_rate
                    ):
                        partner_idx = int(np.random.randint(0, self.population_size))
                        partner = self._decode(positions[partner_idx])
                        self_individual = self._decode(positions[i])
                        child = self.crossover_function(self_individual, partner, self.crossover_rate)
                        positions[i] = _to_vector(child)

                    # GA mutation: perturb the particle
                    if self.mutation_function is not None:
                        ind = self._decode(positions[i])
                        mutated = self.mutation_function(
                            ind, self.mutation_rate,
                            self.image_width, self.image_height,
                            (255, 255),
                        )
                        positions[i] = _to_vector(mutated)
                        positions[i] = np.clip(positions[i], self._lower, self._upper)

        finally:
            if executor is not None:
                executor.shutdown()

        if gbest_individual is None:
            raise RuntimeError("HybridPSOGA did not produce a best individual.")

        return HybridPSOGAResult(
            best_fitness=float(gbest_fitness),
            history=history,
            best_individual=gbest_individual,
        )
