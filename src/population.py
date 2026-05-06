"""Generates random and target-biased triangle populations."""

from dataclasses import dataclass

import numpy as np

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 400
N_TRIANGLES = 100
AlphaRange = tuple[int, int]
TRIANGLE_ALPHA_RANGE: AlphaRange = (5, 255)


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
    target: np.ndarray | None = None,
    seeded: bool = False,
) -> Triangle:
    """
    Creates one triangle with random geometry and optional target-biased color.

    Each triangle is defined by:
    - three vertices: (x1, y1), (x2, y2), (x3, y3)
    - one RGBA color: (r, g, b, a)
    """

    x1 = int(np.random.randint(0, image_width))
    y1 = int(np.random.randint(0, image_height))
    x2 = int(np.random.randint(0, image_width))
    y2 = int(np.random.randint(0, image_height))
    x3 = int(np.random.randint(0, image_width))
    y3 = int(np.random.randint(0, image_height))
    r, g, b = _create_random_rgb(
        target=target,
        seeded=seeded,
        image_width=image_width,
        image_height=image_height,
    )

    return Triangle(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        x3=x3,
        y3=y3,
        r=r,
        g=g,
        b=b,
        a=sample_alpha(triangle_alpha_range=triangle_alpha_range),
    )


def create_random_individual(
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
    target: np.ndarray | None = None,
    seeded: bool = False,
) -> list[Triangle]:
    """
    Creates one individual with random or target-biased triangle colors.

    One individual represents a complete candidate image composed of
    multiple triangles.
    """

    return [
        create_random_triangle(
            image_width=image_width,
            image_height=image_height,
            triangle_alpha_range=triangle_alpha_range,
            target=target,
            seeded=seeded,
        )
        for _ in range(n_triangles)
    ]


def create_population(
    population_size: int = 1000,
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
    target: np.ndarray | None = None,
    seeded: bool = False,
) -> list[list[Triangle]]:
    """
    Generates a random or target-biased population for the Genetic Algorithm.

    Each individual in the population is a candidate solution composed of
    multiple triangles.
    """

    return [
        create_random_individual(
            n_triangles=n_triangles,
            image_width=image_width,
            image_height=image_height,
            triangle_alpha_range=triangle_alpha_range,
            target=target,
            seeded=seeded,
        )
        for _ in range(population_size)
    ]


def create_target_seeded_population(
    target: np.ndarray,
    population_size: int = 1000,
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> list[list[Triangle]]:
    """Creates a population where each triangle's color is sampled from the target image.

    Positions are still random, but colors start near real pixel values, giving
    the GA a much better starting point than fully random initialization.
    """

    result = []
    for _ in range(population_size):
        individual = []
        for _ in range(n_triangles):
            x1 = int(np.random.randint(0, image_width))
            y1 = int(np.random.randint(0, image_height))
            x2 = int(np.random.randint(0, image_width))
            y2 = int(np.random.randint(0, image_height))
            x3 = int(np.random.randint(0, image_width))
            y3 = int(np.random.randint(0, image_height))
            sx = int(np.random.randint(0, image_width))
            sy = int(np.random.randint(0, image_height))
            r, g, b = (
                int(target[sy, sx, 0]),
                int(target[sy, sx, 1]),
                int(target[sy, sx, 2]),
            )
            individual.append(
                Triangle(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    x3=x3,
                    y3=y3,
                    r=r,
                    g=g,
                    b=b,
                    a=255,
                )
            )
        result.append(individual)
    return result


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


def sample_alpha(
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
) -> int:
    """Samples one alpha value uniformly from the configured range."""

    min_alpha, max_alpha = validate_triangle_alpha_range(triangle_alpha_range)

    return int(np.random.randint(min_alpha, max_alpha + 1))


def _create_random_rgb(
    target: np.ndarray | None,
    seeded: bool,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int]:
    """Samples RGB uniformly or from a random target pixel."""

    if not seeded:
        return (
            int(np.random.randint(0, 256)),
            int(np.random.randint(0, 256)),
            int(np.random.randint(0, 256)),
        )

    seed_target = _validate_seed_target(
        target=target,
        image_width=image_width,
        image_height=image_height,
    )
    sample_x = int(np.random.randint(0, image_width))
    sample_y = int(np.random.randint(0, image_height))
    sampled_rgb = seed_target[sample_y, sample_x]

    return int(sampled_rgb[0]), int(sampled_rgb[1]), int(sampled_rgb[2])


def _validate_seed_target(
    target: np.ndarray | None,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Validates the RGB target used for seeded triangle colors."""

    if target is None:
        raise ValueError("target must be provided when seeded=True.")
    if not isinstance(target, np.ndarray):
        raise ValueError("target must be a NumPy array when seeded=True.")
    if target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target must have shape (image_height, image_width, 3).")
    if target.shape[0] != image_height or target.shape[1] != image_width:
        raise ValueError("target dimensions must match image_height and image_width.")

    return target
