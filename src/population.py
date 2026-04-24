"""Generates a random individual (100 random triagles) for the Genetic Algorithm"""

from dataclasses import dataclass

import numpy as np

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 400
N_TRIANGLES = 100


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
    image_width: int = IMAGE_WIDTH, image_height: int = IMAGE_HEIGHT
) -> Triangle:
    """
    Creates one random triangle.

    Each triangle is defined by:
    - three vertices: (x1, y1), (x2, y2), (x3, y3)
    - one RGBA color: (r, g, b, a)
    """

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
        a=np.random.randint(20, 256),  # avoid fully invisible triangles
    )


def create_random_individual(
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> list[Triangle]:
    """
    Creates one individual.

    One individual represents a complete candidate image composed of
    multiple triangles.
    """

    return [
        create_random_triangle(image_width, image_height) for _ in range(n_triangles)
    ]


def create_population(
    population_size: int = 1000,
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> list[list[Triangle]]:
    """
    Generates the initial population for the Genetic Algorithm.

    Each individual in the population is a candidate solution composed of
    multiple triangles.
    """

    return [
        create_random_individual(
            n_triangles=n_triangles, image_width=image_width, image_height=image_height
        )
        for _ in range(population_size)
    ]
