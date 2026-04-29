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


def clamp_triangle_edges(
    triangle: Triangle,
    max_edge_length: int,
    image_width: int,
    image_height: int,
) -> None:
    """Clamps triangle vertices in-place so no edge exceeds max_edge_length.

    Fixes each edge by pulling the farther vertex toward the anchor along the
    edge direction. Vertices are also kept within image bounds.
    """

    def _shorten(ax: int, ay: int, bx: int, by: int) -> tuple[int, int]:
        dx, dy = bx - ax, by - ay
        length = (dx * dx + dy * dy) ** 0.5
        if length <= max_edge_length:
            return bx, by
        scale = max_edge_length / length
        nx = int(round(ax + dx * scale))
        ny = int(round(ay + dy * scale))
        return (
            max(0, min(image_width - 1, nx)),
            max(0, min(image_height - 1, ny)),
        )

    triangle.x2, triangle.y2 = _shorten(triangle.x1, triangle.y1, triangle.x2, triangle.y2)
    triangle.x3, triangle.y3 = _shorten(triangle.x1, triangle.y1, triangle.x3, triangle.y3)
    triangle.x3, triangle.y3 = _shorten(triangle.x2, triangle.y2, triangle.x3, triangle.y3)


def create_random_triangle(
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
    max_edge_length: int | None = None,
) -> Triangle:
    """
    Creates one random triangle.

    Each triangle is defined by:
    - three vertices: (x1, y1), (x2, y2), (x3, y3)
    - one RGBA color: (r, g, b, a)

    When max_edge_length is set, all vertices are placed within
    max_edge_length // 2 of a random center point.
    """

    min_alpha, max_alpha = validate_triangle_alpha_range(triangle_alpha_range)

    if max_edge_length is not None:
        r = max(1, max_edge_length // 2)
        cx = int(np.random.randint(0, image_width))
        cy = int(np.random.randint(0, image_height))
        x1 = int(np.clip(cx + np.random.randint(-r, r + 1), 0, image_width - 1))
        y1 = int(np.clip(cy + np.random.randint(-r, r + 1), 0, image_height - 1))
        x2 = int(np.clip(cx + np.random.randint(-r, r + 1), 0, image_width - 1))
        y2 = int(np.clip(cy + np.random.randint(-r, r + 1), 0, image_height - 1))
        x3 = int(np.clip(cx + np.random.randint(-r, r + 1), 0, image_width - 1))
        y3 = int(np.clip(cy + np.random.randint(-r, r + 1), 0, image_height - 1))
    else:
        x1 = int(np.random.randint(0, image_width))
        y1 = int(np.random.randint(0, image_height))
        x2 = int(np.random.randint(0, image_width))
        y2 = int(np.random.randint(0, image_height))
        x3 = int(np.random.randint(0, image_width))
        y3 = int(np.random.randint(0, image_height))

    return Triangle(
        x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3,
        r=int(np.random.randint(0, 256)),
        g=int(np.random.randint(0, 256)),
        b=int(np.random.randint(0, 256)),
        a=int(np.random.randint(min_alpha, max_alpha + 1)),
    )


def create_random_individual(
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
    max_edge_length: int | None = None,
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
            max_edge_length=max_edge_length,
        )
        for _ in range(n_triangles)
    ]


def create_population(
    population_size: int = 1000,
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
    max_edge_length: int | None = None,
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
            max_edge_length=max_edge_length,
        )
        for _ in range(population_size)
    ]


def create_target_seeded_population(
    target: np.ndarray,
    population_size: int = 1000,
    n_triangles: int = N_TRIANGLES,
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    max_edge_length: int | None = None,
) -> list[list[Triangle]]:
    """Creates a population where each triangle's color is sampled from the target image.

    Positions are still random, but colors start near real pixel values, giving
    the GA a much better starting point than fully random initialization.
    """

    result = []
    for _ in range(population_size):
        individual = []
        for _ in range(n_triangles):
            if max_edge_length is not None:
                r = max(1, max_edge_length // 2)
                cx = int(np.random.randint(0, image_width))
                cy = int(np.random.randint(0, image_height))
                x1 = int(np.clip(cx + np.random.randint(-r, r + 1), 0, image_width - 1))
                y1 = int(np.clip(cy + np.random.randint(-r, r + 1), 0, image_height - 1))
                x2 = int(np.clip(cx + np.random.randint(-r, r + 1), 0, image_width - 1))
                y2 = int(np.clip(cy + np.random.randint(-r, r + 1), 0, image_height - 1))
                x3 = int(np.clip(cx + np.random.randint(-r, r + 1), 0, image_width - 1))
                y3 = int(np.clip(cy + np.random.randint(-r, r + 1), 0, image_height - 1))
            else:
                x1 = int(np.random.randint(0, image_width))
                y1 = int(np.random.randint(0, image_height))
                x2 = int(np.random.randint(0, image_width))
                y2 = int(np.random.randint(0, image_height))
                x3 = int(np.random.randint(0, image_width))
                y3 = int(np.random.randint(0, image_height))
            sx = int(np.random.randint(0, image_width))
            sy = int(np.random.randint(0, image_height))
            r_c, g_c, b_c = int(target[sy, sx, 0]), int(target[sy, sx, 1]), int(target[sy, sx, 2])
            individual.append(Triangle(x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3, r=r_c, g=g_c, b=b_c, a=255))
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
