"""
Diversity-preserving GA variants: fitness sharing and restricted mating.

Both classes are drop-in subclasses of GeneticAlgorithm — they accept the
same constructor arguments and return the same (best_fitness, history) tuple
from run().  They only change *how* selection pressure is shaped.

FitnessSharingGA
    Penalises individuals that live in crowded parts of the search space by
    inflating their effective RMSE before selection.  Multiple good-but-
    different solutions can coexist because no single niche can dominate.

RestrictedMatingGA
    Controls which pairs of individuals are allowed to mate.  After selecting
    parent1 normally, parent2 is chosen from a pool of candidates using one
    of three strategies (unidirectional, bidirectional, best_partial_match).
"""

import numpy as np

from .algorithm import GeneticAlgorithm, Individual


# ---------------------------------------------------------------------------
# Shared gene-vector helpers
# ---------------------------------------------------------------------------

def _to_norm_vec(ind: Individual, image_width: int, image_height: int) -> np.ndarray:
    """Flatten one individual to a [0, 1]-normalised gene vector."""
    W, H = image_width, image_height
    vec: list[float] = []
    for t in ind:
        vec += [
            t.x1 / W, t.y1 / H,
            t.x2 / W, t.y2 / H,
            t.x3 / W, t.y3 / H,
            t.r / 255, t.g / 255, t.b / 255, t.a / 255,
        ]
    return np.array(vec, dtype=np.float32)


def _quantize(vec: np.ndarray, n_bins: int) -> np.ndarray:
    """Map a normalised vector to discrete bin indices (for Hamming distance)."""
    return (vec * n_bins).astype(int).clip(0, n_bins - 1)


def _hamming(q1: np.ndarray, q2: np.ndarray) -> float:
    """Normalised Hamming distance between two quantised gene vectors."""
    return float(np.sum(q1 != q2)) / len(q1)


def _l1(v1: np.ndarray, v2: np.ndarray) -> float:
    """Mean absolute difference between two normalised gene vectors."""
    return float(np.mean(np.abs(v1 - v2)))


# ---------------------------------------------------------------------------
# Fitness Sharing
# ---------------------------------------------------------------------------

class FitnessSharingGA(GeneticAlgorithm):
    """
    GA with fitness sharing to preserve population diversity.

    After raw RMSE is computed, each individual's fitness is multiplied by
    its *niche count* — the total sharing it receives from nearby neighbours.
    Individuals in crowded genotypic regions get a higher (worse) effective
    RMSE, so selection pressure favours less-explored areas of the space.

    Global-best tracking always uses raw (unmodified) RMSE, so the best
    solution found is never corrupted by the sharing penalty.

    How niche count is computed
    ---------------------------
    1. Each individual's normalised gene vector is quantised into ``n_bins``
       discrete bins per gene, approximating a Hamming representation.
    2. For every pair (i, j), the sharing function is::

           sh(d_ij) = max(0,  1  −  d_ij / sigma_share)

       where d_ij is the normalised Hamming distance.
    3. The niche count for individual i is the sum of sh values over the
       whole population (it is at least 1, counting itself).
    4. Shared fitness  =  raw_fitness × niche_count   (higher = worse for RMSE).

    Args:
        sigma_share: Niche radius in [0, 1] Hamming space.  Individuals closer
                     than sigma_share are considered part of the same niche.
                     Smaller values create finer, more isolated niches.
        n_bins:      Number of quantisation bins per gene.  More bins give a
                     finer Hamming resolution but more computation.
    """

    def __init__(self, *args, sigma_share: float = 0.3, n_bins: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_share = sigma_share
        self.n_bins      = n_bins

    # Override only _evaluate_population so the rest of run() is unchanged.
    def _evaluate_population(self, executor=None) -> list[float]:
        """Evaluate population, then inflate fitness by niche count."""
        # Parent call updates best_fitness / best_individual with raw RMSE
        raw_fitness = super()._evaluate_population(executor)
        return self._apply_sharing(raw_fitness)

    def _apply_sharing(self, raw_fitness: list[float]) -> list[float]:
        """Return shared fitness values (raw × niche_count for each individual)."""
        pop = self.population
        # Quantise every individual once
        Q = [
            _quantize(_to_norm_vec(ind, self.image_width, self.image_height), self.n_bins)
            for ind in pop
        ]
        shared: list[float] = []
        for i, raw_f in enumerate(raw_fitness):
            niche_count = sum(
                max(0.0, 1.0 - _hamming(Q[i], Q[j]) / self.sigma_share)
                for j in range(len(pop))
            )
            # niche_count >= 1 (the individual always shares with itself)
            shared.append(raw_f * max(1.0, niche_count))
        return shared


# ---------------------------------------------------------------------------
# Restricted Mating
# ---------------------------------------------------------------------------

class RestrictedMatingGA(GeneticAlgorithm):
    """
    GA with restricted mating to encourage genetic diversity.

    After parent1 is selected by the standard tournament strategy, parent2 is
    chosen from a pool of K tournament candidates using one of three strategies:

    ``"unidirectional"``
        parent1 is chosen normally.
        parent2 = the pool candidate MOST genetically distant from parent1.
        One-directional: only parent1 enforces the distance preference.

    ``"bidirectional"``
        Sample K candidates for each parent slot independently.
        Pick the pair (p1, p2) with the maximum mutual genetic distance.
        Symmetric: both parents are pushed to be as different as possible.

    ``"best_partial_match"``
        parent1 is chosen normally.
        The gene vector is split into two equal halves (first 50 % of
        triangles / last 50 %).
        parent2 = the pool candidate that maximises::

            distance_in_first_half  −  distance_in_second_half

        This encourages recombining diverse *background* triangles (first
        half) while preserving a shared foundation in the *detail* triangles
        (second half).

    In all cases the genetic distance is the mean absolute difference of the
    normalised [0, 1] gene vectors (L1 distance), which is cheaper to compute
    than Hamming and more sensitive to small continuous changes.

    Args:
        mating_type:    One of ``"unidirectional"``, ``"bidirectional"``,
                        ``"best_partial_match"``.
        candidate_pool: Number of candidates (K) sampled for parent2 selection.
    """

    _VALID_TYPES = {"unidirectional", "bidirectional", "best_partial_match"}

    def __init__(
        self,
        *args,
        mating_type:    str = "unidirectional",
        candidate_pool: int = 5,
        **kwargs,
    ):
        if mating_type not in self._VALID_TYPES:
            raise ValueError(
                f"mating_type must be one of {self._VALID_TYPES}, got {mating_type!r}"
            )
        super().__init__(*args, **kwargs)
        self.mating_type    = mating_type
        self.candidate_pool = candidate_pool

    # -- helpers -------------------------------------------------------------

    def _vec(self, ind: Individual) -> np.ndarray:
        return _to_norm_vec(ind, self.image_width, self.image_height)

    def _pool(self, fitness_values: list[float]) -> list[Individual]:
        """Sample candidate_pool individuals via the configured selection."""
        return [self.select_parent(fitness_values) for _ in range(self.candidate_pool)]

    # -- Override select_parents from GeneticAlgorithm -----------------------

    def select_parents(
        self,
        fitness_values: list[float],
    ) -> tuple[Individual, Individual]:
        """Dispatch to the configured mating strategy."""
        if self.mating_type == "unidirectional":
            return self._unidirectional(fitness_values)
        if self.mating_type == "bidirectional":
            return self._bidirectional(fitness_values)
        return self._best_partial_match(fitness_values)

    # -- Mating strategies ---------------------------------------------------

    def _unidirectional(
        self, fitness_values: list[float]
    ) -> tuple[Individual, Individual]:
        """parent1 normal; parent2 = most distant candidate from parent1."""
        parent1 = self.select_parent(fitness_values)
        v1      = self._vec(parent1)
        pool    = self._pool(fitness_values)
        parent2 = max(pool, key=lambda c: _l1(v1, self._vec(c)))
        return parent1, parent2

    def _bidirectional(
        self, fitness_values: list[float]
    ) -> tuple[Individual, Individual]:
        """Both parents chosen as the most mutually distant pair from two pools."""
        pool1 = self._pool(fitness_values)
        pool2 = self._pool(fitness_values)
        best_dist = -1.0
        best_pair = (pool1[0], pool2[0])
        for p1 in pool1:
            v1 = self._vec(p1)
            for p2 in pool2:
                d = _l1(v1, self._vec(p2))
                if d > best_dist:
                    best_dist = d
                    best_pair = (p1, p2)
        return best_pair

    def _best_partial_match(
        self, fitness_values: list[float]
    ) -> tuple[Individual, Individual]:
        """
        parent1 normal; parent2 scores highest on::

            dist(first_half)  −  dist(second_half)

        — diverse in the first 50 % of genes, similar in the last 50 %.
        """
        parent1 = self.select_parent(fitness_values)
        v1      = self._vec(parent1)
        mid     = len(v1) // 2
        pool    = self._pool(fitness_values)

        def _score(c: Individual) -> float:
            v2 = self._vec(c)
            return float(np.mean(np.abs(v1[:mid] - v2[:mid]))) \
                 - float(np.mean(np.abs(v1[mid:] - v2[mid:])))

        parent2 = max(pool, key=_score)
        return parent1, parent2
