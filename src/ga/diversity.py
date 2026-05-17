"""
Diversity-preserving GA variants: fitness sharing and restricted mating.

Both classes are drop-in subclasses of GeneticAlgorithm — they accept the
same constructor arguments and return the same (best_fitness, history) tuple
from run().  They only change *how* selection pressure is shaped.

FitnessSharingGA
    Penalises individuals that live in crowded parts of the search space by
    inflating their effective RMSE before selection.

RestrictedMatingGA
    Controls which pairs of individuals are allowed to mate.  After selecting
    parent1 normally, parent2 is chosen from a pool of candidates using one
    of three strategies (unidirectional, bidirectional, best_partial_match).
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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


def _col_entropy(col: np.ndarray, n_bins: int = 20) -> float:
    """Shannon entropy (bits) of a 1-D array of normalised values in [0, 1]."""
    hist, _ = np.histogram(col, bins=n_bins, range=(0.0, 1.0))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _make_diversity_callback(
    log: dict, ga_ref: list, W: int, H: int, pheno_interval: int, pheno_scale: float
):
    """
    Return a progress_callback that records genotypic and phenotypic diversity.

    The callback captures log and ga_ref by reference so it can append to
    them each generation without any extra arguments.
    """
    from .. import rendering as _rendering

    def _cb(gl: dict) -> None:
        ga  = ga_ref[0]
        gen = gl["generation"]
        pop = ga.population
        G   = np.array([_to_norm_vec(ind, W, H) for ind in pop])

        log["gen"].append(gen)
        log["best_fitness"].append(gl["global_best_fitness"])
        log["geno_var"].append(float(np.mean(np.var(G, axis=0))))
        log["geno_entropy"].append(
            float(np.mean([_col_entropy(G[:, d]) for d in range(G.shape[1])]))
        )

        if gen % pheno_interval == 0:
            ph   = max(1, int(H * pheno_scale))
            pw   = max(1, int(W * pheno_scale))
            imgs = np.stack([
                np.array(_rendering.render_individual(ind, pw, ph), dtype=np.float32) / 255.0
                for ind in pop
            ])
            log["pheno_gen"].append(gen)
            log["pheno_var"].append(float(np.mean(np.var(imgs, axis=0))))

            n_sample  = min(400, ph * pw)
            px_idx    = np.random.choice(ph * pw, n_sample, replace=False)
            chan_ents: list[float] = []
            for c in range(3):
                ch = imgs[:, :, :, c].reshape(len(pop), -1)[:, px_idx]
                for px_vals in ch.T:
                    hist, _ = np.histogram(px_vals, bins=16, range=(0.0, 1.0))
                    p = hist / hist.sum()
                    p = p[p > 0]
                    chan_ents.append(float(-np.sum(p * np.log2(p))))
            log["pheno_entropy"].append(float(np.mean(chan_ents)))

    return _cb


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
        sigma_share: Niche radius in [0, 1] Hamming space.
        n_bins:      Number of quantisation bins per gene.
    """

    def __init__(self, *args, sigma_share: float = 0.3, n_bins: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_share = sigma_share
        self.n_bins      = n_bins

    def _evaluate_population(self, executor=None) -> list[float]:
        """Evaluate population with raw RMSE, then inflate by niche count."""
        raw_fitness = super()._evaluate_population(executor)
        return self._apply_sharing(raw_fitness)

    def _apply_sharing(self, raw_fitness: list[float]) -> list[float]:
        """Return shared fitness values (raw × niche_count for each individual)."""
        pop = self.population
        Q   = [
            _quantize(_to_norm_vec(ind, self.image_width, self.image_height), self.n_bins)
            for ind in pop
        ]
        shared: list[float] = []
        for i, raw_f in enumerate(raw_fitness):
            niche_count = sum(
                max(0.0, 1.0 - _hamming(Q[i], Q[j]) / self.sigma_share)
                for j in range(len(pop))
            )
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
        parent2 = the pool candidate MOST genetically distant from parent1.

    ``"bidirectional"``
        Sample K candidates for each parent slot independently.
        Pick the pair (p1, p2) with the maximum mutual genetic distance.

    ``"best_partial_match"``
        parent2 maximises::

            distance_in_first_half  −  distance_in_second_half

        Encourages diverse background triangles while preserving a shared
        foundation in the detail triangles.

    Genetic distance uses mean absolute difference of normalised gene vectors
    (L1), which is cheaper and more sensitive than Hamming for continuous values.

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

    def _vec(self, ind: Individual) -> np.ndarray:
        return _to_norm_vec(ind, self.image_width, self.image_height)

    def _pool(self, fitness_values: list[float]) -> list[Individual]:
        """Sample candidate_pool individuals via the configured selection."""
        return [self.select_parent(fitness_values) for _ in range(self.candidate_pool)]

    def select_parents(
        self, fitness_values: list[float]
    ) -> tuple[Individual, Individual]:
        """Dispatch to the configured mating strategy."""
        if self.mating_type == "unidirectional":
            return self._unidirectional(fitness_values)
        if self.mating_type == "bidirectional":
            return self._bidirectional(fitness_values)
        return self._best_partial_match(fitness_values)

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
        """parent2 scores highest on dist(first_half) − dist(second_half)."""
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


class FitnessSharingRestrictedMatingGA(FitnessSharingGA, RestrictedMatingGA):
    """Combines fitness sharing and restricted mating in a single GA.

    ``FitnessSharingGA._evaluate_population`` applies the niche penalty;
    ``RestrictedMatingGA.select_parents`` enforces distance-based pairing.
    Both layers are active simultaneously — Python MRO wires them together.
    """


# ---------------------------------------------------------------------------
# Diversity tracking
# ---------------------------------------------------------------------------

def run_diversity_trial(
    ga_class,
    pipeline: str,
    extra_kwargs: dict,
    base_dict: dict,
    results_dir: Path | str,
    image_height: int,
    image_width: int,
    pheno_interval: int = 5,
    pheno_scale: float = 0.20,
) -> dict:
    """
    Run one diversity-tracked GA trial and cache the result to disk.

    Returns the cached log immediately on subsequent calls with the same pipeline.

    Tracked signals per generation: gen, best_fitness, geno_var, geno_entropy,
    pheno_gen, pheno_var, pheno_entropy.
    """
    results_dir = Path(results_dir)
    cache = results_dir / f"{pipeline}_diversity.json"

    if cache.exists():
        with open(cache) as f:
            log = json.load(f)
        print(f"✓ '{pipeline}' diversity — loaded from cache")
        return log

    H, W = image_height, image_width
    log: dict = dict(
        gen=[], best_fitness=[],
        geno_var=[], geno_entropy=[],
        pheno_gen=[], pheno_var=[], pheno_entropy=[],
    )
    _ref: list = [None]

    ga = ga_class(
        **{**base_dict, **extra_kwargs},
        progress_callback=_make_diversity_callback(log, _ref, W, H, pheno_interval, pheno_scale),
    )
    _ref[0] = ga
    ga.run()

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(log, f)
    print(f"  ✓ '{pipeline}' diversity done — saved to cache")
    return log


# Must be module-level so ProcessPoolExecutor can pickle it on macOS spawn.
def _run_diversity_worker(
    ga_class_name: str,
    pipeline: str,
    extra_kwargs: dict,
    base_dict: dict,
    image_height: int,
    image_width: int,
    pheno_interval: int,
    pheno_scale: float,
) -> tuple[str, dict]:
    """Worker that runs one diversity-tracked GA trial; returns (pipeline, log)."""
    if ga_class_name == "GeneticAlgorithm":
        from .algorithm import GeneticAlgorithm as _cls
    elif ga_class_name == "FitnessSharingGA":
        _cls = FitnessSharingGA
    elif ga_class_name == "RestrictedMatingGA":
        _cls = RestrictedMatingGA
    elif ga_class_name == "FitnessSharingRestrictedMatingGA":
        _cls = FitnessSharingRestrictedMatingGA
    else:
        raise ValueError(f"Unknown GA class: {ga_class_name!r}")

    H, W = image_height, image_width
    log: dict = dict(
        gen=[], best_fitness=[],
        geno_var=[], geno_entropy=[],
        pheno_gen=[], pheno_var=[], pheno_entropy=[],
    )
    _ref: list = [None]

    ga = _cls(
        **{**base_dict, **extra_kwargs},
        progress_callback=_make_diversity_callback(log, _ref, W, H, pheno_interval, pheno_scale),
        evaluation_backend="sequential",
    )
    _ref[0] = ga
    ga.run()
    return pipeline, log


def run_diversity_batch(
    configs: dict[str, tuple[str, str, dict]],
    base_dict: dict,
    results_dir: Path | str,
    image_height: int,
    image_width: int,
    pheno_interval: int = 5,
    pheno_scale: float = 0.20,
) -> dict[str, dict]:
    """
    Run one diversity trial per config using a single shared ProcessPoolExecutor.

    Cached configs are loaded from disk instantly; only uncached ones are dispatched.

    Args:
        configs:        Ordered dict mapping display_label →
                        ``(pipeline, ga_class_name, extra_kwargs)``.
        base_dict:      Base GA constructor arguments shared by all configs.
        results_dir:    Directory where ``{pipeline}_diversity.json`` files live.
        image_height:   Canvas height in pixels.
        image_width:    Canvas width in pixels.
        pheno_interval: Render every N generations for phenotypic metrics.
        pheno_scale:    Downscale factor for rendered images.

    Returns:
        Dict mapping each display_label → diversity log dict.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logs: dict[str, dict]              = {}
    to_run: dict[str, tuple[str, str, dict]] = {}

    for label, (pipeline, ga_class_name, extra_kwargs) in configs.items():
        cache = results_dir / f"{pipeline}_diversity.json"
        if cache.exists():
            with open(cache) as f:
                logs[label] = json.load(f)
            print(f"  ✓ '{pipeline}' diversity — loaded from cache", flush=True)
        else:
            to_run[label] = (pipeline, ga_class_name, extra_kwargs)
            print(f"  ~ '{pipeline}' diversity — will run", flush=True)

    if not to_run:
        return logs

    n       = len(to_run)
    workers = min(n, os.cpu_count() or 1)
    print(f"  dispatching {n} diversity trial(s)  workers={workers}", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_diversity_worker,
                ga_class_name, pipeline, extra_kwargs, base_dict,
                image_height, image_width, pheno_interval, pheno_scale,
            ): label
            for label, (pipeline, ga_class_name, extra_kwargs) in to_run.items()
        }
        for future in as_completed(futures):
            label          = futures[future]
            pipeline, log  = future.result()
            logs[label]    = log
            cache          = results_dir / f"{pipeline}_diversity.json"
            with open(cache, "w") as f:
                json.dump(log, f)
            print(f"  ✓ '{pipeline}' diversity done — saved to cache", flush=True)

    return logs
