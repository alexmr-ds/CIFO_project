"""Greedy one-triangle-at-a-time initialization for triangle-based GA populations.

Builds one individual by placing triangles greedily: for each slot, K random
candidate triangles are evaluated and the one that most reduces RMSE is kept.
The rest of the population is generated as small mutations of this base individual.
"""

import copy

import numpy as np
from PIL import Image, ImageDraw

from .. import population
from . import mutate

Individual = list[population.Triangle]


def _render_triangle_onto(
    canvas: np.ndarray,
    triangle: population.Triangle,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Returns a new float32 array with one opaque triangle composited on top."""

    img = Image.fromarray(canvas.astype(np.uint8), "RGB").convert("RGBA")
    overlay = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw.polygon(
        [(triangle.x1, triangle.y1), (triangle.x2, triangle.y2), (triangle.x3, triangle.y3)],
        fill=(triangle.r, triangle.g, triangle.b, 255),
    )
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(composited, dtype=np.float32)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Computes normalized RMSE between two float32 arrays in [0, 255]."""

    return float(np.sqrt(np.mean(((a - b) / 255.0) ** 2)))


def _sample_candidate(
    target: np.ndarray,
    image_width: int,
    image_height: int,
) -> population.Triangle:
    """Creates one random triangle with color sampled from the target."""

    sx = int(np.random.randint(0, image_width))
    sy = int(np.random.randint(0, image_height))
    return population.Triangle(
        x1=int(np.random.randint(0, image_width)),
        y1=int(np.random.randint(0, image_height)),
        x2=int(np.random.randint(0, image_width)),
        y2=int(np.random.randint(0, image_height)),
        x3=int(np.random.randint(0, image_width)),
        y3=int(np.random.randint(0, image_height)),
        r=int(target[sy, sx, 0]),
        g=int(target[sy, sx, 1]),
        b=int(target[sy, sx, 2]),
        a=255,
    )


def create_greedy_individual(
    target: np.ndarray,
    n_triangles: int,
    image_width: int,
    image_height: int,
    candidates_per_triangle: int = 200,
    verbose: bool = False,
) -> Individual:
    """Builds one individual by placing triangles one at a time greedily.

    For each triangle slot, ``candidates_per_triangle`` random candidates are
    tried and the one that most reduces RMSE against the target is kept.
    Colors are sampled from the target image so candidates start meaningful.

    Args:
        target: RGB target image array with shape (H, W, 3).
        n_triangles: Number of triangles to place.
        image_width: Canvas width in pixels.
        image_height: Canvas height in pixels.
        candidates_per_triangle: How many random candidates to try per slot.
            Higher = better quality but slower. 200 is a good default.
        verbose: Print progress every 10 triangles.

    Returns:
        One Individual (list of Triangles) built greedily.
    """

    target_f32 = target.astype(np.float32)
    canvas = np.zeros((image_height, image_width, 3), dtype=np.float32)
    individual: Individual = []

    for slot in range(n_triangles):
        current_rmse = _rmse(canvas, target_f32)
        best_triangle = None
        best_rmse = current_rmse

        for _ in range(candidates_per_triangle):
            candidate = _sample_candidate(target, image_width, image_height)
            candidate_canvas = _render_triangle_onto(canvas, candidate, image_width, image_height)
            candidate_rmse = _rmse(candidate_canvas, target_f32)
            if candidate_rmse < best_rmse:
                best_rmse = candidate_rmse
                best_triangle = candidate

        if best_triangle is None:
            best_triangle = _sample_candidate(target, image_width, image_height)

        individual.append(best_triangle)
        canvas = _render_triangle_onto(canvas, best_triangle, image_width, image_height)

        if verbose and (slot + 1) % 10 == 0:
            print(f"  greedy: placed {slot + 1}/{n_triangles} triangles  RMSE={best_rmse:.4f}")

    return individual


def create_greedy_seeded_population(
    target: np.ndarray,
    population_size: int,
    n_triangles: int,
    image_width: int,
    image_height: int,
    candidates_per_triangle: int = 200,
    seed_mutation_rate: float = 0.05,
    verbose: bool = True,
) -> list[Individual]:
    """Builds a full population seeded from one greedily constructed individual.

    One greedy individual is built first, then the rest of the population is
    generated as lightly mutated copies so every individual starts near the
    greedy solution rather than at a random point.

    Args:
        target: RGB target image array with shape (H, W, 3).
        population_size: Number of individuals to return.
        n_triangles: Triangles per individual.
        image_width: Canvas width in pixels.
        image_height: Canvas height in pixels.
        candidates_per_triangle: Candidates tried per triangle slot in greedy build.
        seed_mutation_rate: Mutation rate applied when creating population variants.
        verbose: Print greedy construction progress.

    Returns:
        List of ``population_size`` individuals.
    """

    if verbose:
        print(f"Building greedy individual ({n_triangles} triangles, {candidates_per_triangle} candidates each)...")

    base = create_greedy_individual(
        target=target,
        n_triangles=n_triangles,
        image_width=image_width,
        image_height=image_height,
        candidates_per_triangle=candidates_per_triangle,
        verbose=verbose,
    )

    pop: list[Individual] = [copy.deepcopy(base)]
    for _ in range(population_size - 1):
        variant = copy.deepcopy(base)
        mutate.focused_triangle_mutation(
            variant,
            mutation_rate=seed_mutation_rate,
            image_width=image_width,
            image_height=image_height,
            triangle_alpha_range=(255, 255),
        )
        pop.append(variant)

    if verbose:
        print(f"Greedy population ready ({population_size} individuals).")

    return pop
