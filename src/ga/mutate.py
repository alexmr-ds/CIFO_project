"""
Mutation operators for triangle-based individuals.

Mutation introduces random changes into an individual *after* crossover.
Without mutation the GA can only recombine existing genetic material and
will quickly converge to a local optimum.  Mutation is the mechanism that
lets the search escape those local optima by exploring new areas of the
search space.

All operators share the same signature so they are interchangeable in the
GA config:

    mutate_fn(individual, mutation_rate, image_width, image_height,
              triangle_alpha_range) -> individual

Mutation rate
-------------
The ``mutation_rate`` is the per-triangle probability of being mutated.
A rate of 0.1 means each triangle has a 10 % chance of changing.
Higher rates explore more but can destroy good solutions; lower rates
preserve solutions but converge slowly.

Available operators
-------------------
random_triangle_mutation
    Picks one random attribute of the triangle and nudges it by a small
    random delta.  Simple, low-disruption, and the default for all runs.

focused_triangle_mutation
    Mutates *all* vertex and colour attributes of each selected triangle at
    once (9 attributes vs. 1).  Each mutation is ~9× more impactful, so
    this operator explores larger jumps in the search space.  Occasionally
    performs a full random reset of the triangle instead of a nudge.
"""

import numpy as np

from .. import population


def random_triangle_mutation(
    individual: list[population.Triangle],
    mutation_rate: float,
    image_width: int,
    image_height: int,
    triangle_alpha_range: population.AlphaRange = (
        population.TRIANGLE_ALPHA_RANGE
    ),
) -> list[population.Triangle]:
    """
    Mutates a randomly selected attribute of each triangle with probability mutation_rate.

    For each triangle that is selected for mutation:
      - A single attribute is chosen at random from the 10 possible fields
        (x1, y1, x2, y2, x3, y3, r, g, b, a).
      - A small delta is added to that attribute (±15 for coordinates,
        ±25 for colour/alpha).
      - The result is clamped to the valid range so no triangle escapes
        the canvas or uses out-of-range colour values.

    This is the most conservative mutation operator — it makes small,
    incremental changes that are unlikely to destroy a good individual.

    Args:
        individual:           List of triangles to mutate (modified in place).
        mutation_rate:        Per-triangle probability of being mutated (0.0–1.0).
        image_width:          Canvas width — clamps x-coordinates.
        image_height:         Canvas height — clamps y-coordinates.
        triangle_alpha_range: Inclusive (min, max) alpha range for clamping.

    Returns:
        The mutated individual (same object, modified in place).
    """

    min_alpha, max_alpha = population.validate_triangle_alpha_range(
        triangle_alpha_range
    )

    for triangle in individual:
        # Skip this triangle with probability (1 - mutation_rate)
        if np.random.random() >= mutation_rate:
            continue

        # Pick exactly one attribute to change this mutation event
        attribute = np.random.choice(
            ["x1", "y1", "x2", "y2", "x3", "y3", "r", "g", "b", "a"]
        )

        if attribute.startswith("x"):
            # Horizontal coordinate: nudge by ±15 pixels, clamp to canvas
            delta = int(np.random.randint(-15, 16))
            value = getattr(triangle, attribute) + delta
            setattr(triangle, attribute, max(0, min(image_width - 1, value)))

        elif attribute.startswith("y"):
            # Vertical coordinate: nudge by ±15 pixels, clamp to canvas
            delta = int(np.random.randint(-15, 16))
            value = getattr(triangle, attribute) + delta
            setattr(triangle, attribute, max(0, min(image_height - 1, value)))

        elif attribute == "a":
            # Alpha channel: nudge by ±25, respect the configured alpha range
            delta = int(np.random.randint(-25, 26))
            value = getattr(triangle, attribute) + delta
            setattr(triangle, attribute, max(min_alpha, min(max_alpha, value)))

        else:
            # RGB colour channel: nudge by ±25, clamp to [0, 255]
            delta = int(np.random.randint(-25, 26))
            value = getattr(triangle, attribute) + delta
            setattr(triangle, attribute, max(0, min(255, value)))

    return individual


def focused_triangle_mutation(
    individual: list[population.Triangle],
    mutation_rate: float,
    image_width: int,
    image_height: int,
    triangle_alpha_range: population.AlphaRange = (
        population.TRIANGLE_ALPHA_RANGE
    ),
    position_delta: int = 30,
    color_delta: int = 45,
    full_reset_prob: float = 0.1,
) -> list[population.Triangle]:
    """
    Mutates all vertex and colour attributes of each selected triangle at once.

    Unlike ``random_triangle_mutation`` which touches one attribute per event,
    this operator changes all six vertex coordinates and all three RGB channels
    simultaneously for each selected triangle.  This makes each mutation
    roughly 9× more impactful, allowing larger jumps in the fitness landscape.

    Two mutation modes are applied at random per selected triangle:
      - **Nudge** (90 % of the time): each attribute gets a small random delta
        (±position_delta for coordinates, ±color_delta for colours).
      - **Full reset** (10 % of the time): the triangle is re-randomised
        completely — this gives the GA a way to escape very deep local optima
        by occasionally discarding a bad triangle entirely.

    Args:
        individual:           List of triangles to mutate (modified in place).
        mutation_rate:        Per-triangle probability of being mutated (0.0–1.0).
        image_width:          Canvas width — clamps x-coordinates.
        image_height:         Canvas height — clamps y-coordinates.
        triangle_alpha_range: Inclusive (min, max) alpha range for clamping.
        position_delta:       Maximum coordinate nudge in pixels (default ±30).
        color_delta:          Maximum colour channel nudge (default ±45).
        full_reset_prob:      Probability of a full random reset instead of nudge.

    Returns:
        The mutated individual (same object, modified in place).

    Raises:
        ValueError: If mutation_rate is outside [0, 1].
    """

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1.")

    min_alpha, max_alpha = population.validate_triangle_alpha_range(
        triangle_alpha_range
    )

    for triangle in individual:
        # Skip this triangle with probability (1 - mutation_rate)
        if np.random.random() >= mutation_rate:
            continue

        if np.random.random() < full_reset_prob:
            # --- Full random reset ---
            # Completely randomise the triangle's geometry and colour.
            # This is a large, disruptive change that helps escape local optima.
            triangle.x1 = int(np.random.randint(0, image_width))
            triangle.y1 = int(np.random.randint(0, image_height))
            triangle.x2 = int(np.random.randint(0, image_width))
            triangle.y2 = int(np.random.randint(0, image_height))
            triangle.x3 = int(np.random.randint(0, image_width))
            triangle.y3 = int(np.random.randint(0, image_height))
            triangle.r = int(np.random.randint(0, 256))
            triangle.g = int(np.random.randint(0, 256))
            triangle.b = int(np.random.randint(0, 256))
        else:
            # --- Small coordinated nudge ---
            # All six vertex coordinates and all three colour channels are
            # perturbed by small independent deltas in one mutation event.
            triangle.x1 = max(0, min(image_width - 1,  triangle.x1 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.y1 = max(0, min(image_height - 1, triangle.y1 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.x2 = max(0, min(image_width - 1,  triangle.x2 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.y2 = max(0, min(image_height - 1, triangle.y2 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.x3 = max(0, min(image_width - 1,  triangle.x3 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.y3 = max(0, min(image_height - 1, triangle.y3 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.r = max(0, min(255, triangle.r + int(np.random.randint(-color_delta, color_delta + 1))))
            triangle.g = max(0, min(255, triangle.g + int(np.random.randint(-color_delta, color_delta + 1))))
            triangle.b = max(0, min(255, triangle.b + int(np.random.randint(-color_delta, color_delta + 1))))

        # Alpha is handled separately for both modes:
        # - If alpha is fixed (min == max), keep it exactly at that value.
        # - Otherwise nudge it within the configured range.
        if min_alpha == max_alpha:
            triangle.a = min_alpha
        else:
            triangle.a = max(min_alpha, min(max_alpha, triangle.a + int(np.random.randint(-30, 31))))

    return individual
