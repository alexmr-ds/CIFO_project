"""Implements mutation operators for triangle-based individuals."""

import numpy as np

from .. import population


def random_triangle_mutation(
    individual: list[population.Triangle],
    mutation_rate: float,
    image_width: int,
    image_height: int,
) -> list[population.Triangle]:
    """Mutates triangle attributes in an individual."""

    for triangle in individual:
        if np.random.random() >= mutation_rate:
            continue

        attribute = np.random.choice(
            ["x1", "y1", "x2", "y2", "x3", "y3", "r", "g", "b", "a"]
        )

        if attribute.startswith("x"):
            delta = int(np.random.randint(-15, 16))
            value = getattr(triangle, attribute) + delta
            setattr(triangle, attribute, max(0, min(image_width - 1, value)))
        elif attribute.startswith("y"):
            delta = int(np.random.randint(-15, 16))
            value = getattr(triangle, attribute) + delta
            setattr(triangle, attribute, max(0, min(image_height - 1, value)))
        elif attribute == "a":
            delta = int(np.random.randint(-25, 26))
            value = getattr(triangle, attribute) + delta
            setattr(triangle, attribute, max(20, min(255, value)))
        else:
            delta = int(np.random.randint(-25, 26))
            value = getattr(triangle, attribute) + delta
            setattr(triangle, attribute, max(0, min(255, value)))

    return individual
