"""Generates random triangle individuals for the genetic algorithm."""

from dataclasses import dataclass

import numpy as np

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 400
N_TRIANGLES = 100
AlphaRange = tuple[int, int]
TRIANGLE_ALPHA_RANGE: AlphaRange = (20, 255)


@dataclass
class Triangle:
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    r: int
    g: int
    b: int
    a: int


def create_random_triangle(
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
) -> Triangle:
    """
    Creates one random triangle.

    Each triangle is defined by:
    - three vertices: (x1, y1), (x2, y2), (x3, y3)
    - one RGBA color: (r, g, b, a)
    """

    min_alpha, max_alpha = validate_triangle_alpha_range(triangle_alpha_range)

    return Triangle(
        x1=np.random.randint(0, image_width),
        y1=np.random.randint(0, image_height),
        x2=np.random.randint(0, image_width),
        y2=np.random.randint(0, image_height),
        x3=np.random.randint(0, image_width),
        y3=np.random.randint(0, image_height),
        r=np.random.randint(0, 256),
        g=np.random.randint(0, 256),
        b=np.random.randint(0, 256),
        a=np.random.randint(min_alpha, max_alpha + 1),
    )


def create_random_individual(
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
) -> list[Triangle]:
    """
    Creates one individual.

    One individual represents a complete candidate image composed of
    multiple triangles.
    """

    return [
        create_random_triangle(
            image_width=image_width,
            image_height=image_height,
            triangle_alpha_range=triangle_alpha_range,
        )
        for _ in range(n_triangles)
    ]


def create_population(
    population_size: int = 1000,
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
) -> list[list[Triangle]]:
    """
    Generates the initial population for the Genetic Algorithm.

    Each individual in the population is a candidate solution composed of
    multiple triangles.
    """

    return [
        create_random_individual(
            n_triangles=n_triangles,
            image_width=image_width,
            image_height=image_height,
            triangle_alpha_range=triangle_alpha_range,
        )
        for _ in range(population_size)
    ]


def validate_triangle_alpha_range(
    triangle_alpha_range: AlphaRange,
) -> AlphaRange:
    """Validates and normalizes the inclusive triangle alpha range."""

    if not isinstance(triangle_alpha_range, tuple) or len(triangle_alpha_range) != 2:
        raise ValueError("triangle_alpha_range must be a tuple of two integers.")

    min_alpha, max_alpha = triangle_alpha_range

    if not isinstance(min_alpha, int) or not isinstance(max_alpha, int):
        raise ValueError("triangle_alpha_range must contain integers.")

    if not 0 <= min_alpha <= 255 or not 0 <= max_alpha <= 255:
        raise ValueError("triangle_alpha_range values must be between 0 and 255.")

    if min_alpha > max_alpha:
        raise ValueError(
            "triangle_alpha_range minimum must be less than or equal to maximum."
        )

    return min_alpha, max_alpha
