"""Implements mutation operators for triangle-based individuals."""

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
    """Mutates triangle attributes in an individual."""

    min_alpha, max_alpha = population.validate_triangle_alpha_range(
        triangle_alpha_range
    )

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
            setattr(triangle, attribute, max(min_alpha, min(max_alpha, value)))
        else:
            delta = int(np.random.randint(-25, 26))
            value = getattr(triangle, attribute) + delta
            setattr(triangle, attribute, max(0, min(255, value)))

    return individual


def volatile_triangle_mutation(
    individual: list[population.Triangle],
    mutation_rate: float,
    image_width: int,
    image_height: int,
    triangle_alpha_range: population.AlphaRange = (
        population.TRIANGLE_ALPHA_RANGE
    ),
    small_mutation_prob: float = 0.85,
) -> list[population.Triangle]:
    """Mutates triangle attributes with mostly local changes."""

    min_alpha, max_alpha = population.validate_triangle_alpha_range(
        triangle_alpha_range
    )

    for triangle in individual:
        if np.random.random() >= mutation_rate:
            continue

        attribute = np.random.choice(
            ["x1", "y1", "x2", "y2", "x3", "y3", "r", "g", "b", "a"]
        )

        # Usually: small local mutation
        if np.random.random() < small_mutation_prob:
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
                setattr(triangle, attribute, max(min_alpha, min(max_alpha, value)))

            else:
                delta = int(np.random.randint(-25, 26))
                value = getattr(triangle, attribute) + delta
                setattr(triangle, attribute, max(0, min(255, value)))

        # Sometimes: large disruptive mutation
        else:
            if attribute.startswith("x"):
                setattr(triangle, attribute, int(np.random.randint(0, image_width)))

            elif attribute.startswith("y"):
                setattr(triangle, attribute, int(np.random.randint(0, image_height)))

            elif attribute == "a":
                setattr(
                    triangle,
                    attribute,
                    int(np.random.randint(min_alpha, max_alpha + 1)),
                )

            else:
                setattr(triangle, attribute, int(np.random.randint(0, 256)))

    return individual
