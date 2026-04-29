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


def gaussian_triangle_mutation(
    individual: list[population.Triangle],
    mutation_rate: float,
    image_width: int,
    image_height: int,
    triangle_alpha_range: population.AlphaRange = (
        population.TRIANGLE_ALPHA_RANGE
    ),
    position_sigma: float = 0.05,
    color_sigma: float = 0.08,
    force_opaque: bool = False,
) -> list[population.Triangle]:
    """Mutates triangle genes with Gaussian perturbations."""

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1.")
    if position_sigma < 0.0 or color_sigma < 0.0:
        raise ValueError("position_sigma and color_sigma must be non-negative.")

    min_alpha, max_alpha = population.validate_triangle_alpha_range(
        triangle_alpha_range
    )
    sigma_x = max(1.0, position_sigma * image_width)
    sigma_y = max(1.0, position_sigma * image_height)
    sigma_color = max(1.0, color_sigma * 255.0)

    for triangle in individual:
        if np.random.random() >= mutation_rate:
            continue

        triangle.x1 = int(np.clip(np.round(np.random.normal(triangle.x1, sigma_x)), 0, image_width - 1))
        triangle.y1 = int(np.clip(np.round(np.random.normal(triangle.y1, sigma_y)), 0, image_height - 1))
        triangle.x2 = int(np.clip(np.round(np.random.normal(triangle.x2, sigma_x)), 0, image_width - 1))
        triangle.y2 = int(np.clip(np.round(np.random.normal(triangle.y2, sigma_y)), 0, image_height - 1))
        triangle.x3 = int(np.clip(np.round(np.random.normal(triangle.x3, sigma_x)), 0, image_width - 1))
        triangle.y3 = int(np.clip(np.round(np.random.normal(triangle.y3, sigma_y)), 0, image_height - 1))

        triangle.r = int(np.clip(np.round(np.random.normal(triangle.r, sigma_color)), 0, 255))
        triangle.g = int(np.clip(np.round(np.random.normal(triangle.g, sigma_color)), 0, 255))
        triangle.b = int(np.clip(np.round(np.random.normal(triangle.b, sigma_color)), 0, 255))

        if force_opaque:
            triangle.a = 255
        else:
            triangle.a = int(
                np.clip(
                    np.round(np.random.normal(triangle.a, sigma_color)),
                    min_alpha,
                    max_alpha,
                )
            )

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
    """Mutates all vertex and color attributes of each selected triangle at once.

    Unlike single-attribute operators, every selected triangle gets all six
    vertex coordinates and all RGB channels perturbed in the same step,
    making each mutation ~9x more impactful.
    """

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1.")

    min_alpha, max_alpha = population.validate_triangle_alpha_range(
        triangle_alpha_range
    )

    for triangle in individual:
        if np.random.random() >= mutation_rate:
            continue

        if np.random.random() < full_reset_prob:
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
            triangle.x1 = max(0, min(image_width - 1, triangle.x1 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.y1 = max(0, min(image_height - 1, triangle.y1 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.x2 = max(0, min(image_width - 1, triangle.x2 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.y2 = max(0, min(image_height - 1, triangle.y2 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.x3 = max(0, min(image_width - 1, triangle.x3 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.y3 = max(0, min(image_height - 1, triangle.y3 + int(np.random.randint(-position_delta, position_delta + 1))))
            triangle.r = max(0, min(255, triangle.r + int(np.random.randint(-color_delta, color_delta + 1))))
            triangle.g = max(0, min(255, triangle.g + int(np.random.randint(-color_delta, color_delta + 1))))
            triangle.b = max(0, min(255, triangle.b + int(np.random.randint(-color_delta, color_delta + 1))))

        if min_alpha == max_alpha:
            triangle.a = min_alpha
        else:
            triangle.a = max(min_alpha, min(max_alpha, triangle.a + int(np.random.randint(-30, 31))))

    return individual
