"""
Defines the Triangle dataclass and population generation utilities.

A *population* is a list of *individuals*.
An *individual* is a list of *Triangles* — a complete candidate solution
representing one approximated image.

Each Triangle stores three vertex coordinates and one RGBA colour.
The GA evolves these values over many generations to minimise the
difference between the rendered individual and the target image.

Key constants
-------------
IMAGE_WIDTH / IMAGE_HEIGHT : default canvas size (300 × 400 px)
N_TRIANGLES                : default triangles per individual (100)
TRIANGLE_ALPHA_RANGE       : default alpha range used during initialisation
"""

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed canvas size matching the resized target image in load_image.py.
# Keeping these in sync is critical — render and target must be same shape.
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 400

# Number of triangles that make up one candidate image.
# More triangles → more expressive but also a larger search space and slower
# fitness evaluation.
N_TRIANGLES = 100

# Type alias for the (min_alpha, max_alpha) tuple used throughout the code
AlphaRange = tuple[int, int]

# Default alpha range: triangles can be nearly transparent (5) to fully
# opaque (255).  The lower bound of 5 avoids completely invisible triangles.
TRIANGLE_ALPHA_RANGE: AlphaRange = (5, 255)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class Triangle:
    """
    Represents one coloured triangle in a candidate image.

    Geometry: three vertices in pixel coordinates.
    Colour:   RGBA with values in [0, 255].

    The dataclass is mutable so mutation operators can update fields
    in-place without creating a new object each time.
    """

    # Vertex coordinates — must stay within [0, image_width/height)
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int

    # Colour channels (0–255)
    r: int
    g: int
    b: int
    a: int  # alpha: 0 = fully transparent, 255 = fully opaque


# ---------------------------------------------------------------------------
# Triangle creation
# ---------------------------------------------------------------------------

def create_random_triangle(
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    triangle_alpha_range: AlphaRange = TRIANGLE_ALPHA_RANGE,
    target: np.ndarray | None = None,
    seeded: bool = False,
) -> Triangle:
    """
    Creates one triangle with random geometry and optionally target-biased colour.

    Vertex positions are always uniformly random across the canvas.
    Colour can be:
      - Random (seeded=False): all three RGB channels drawn uniformly from [0, 255].
      - Seeded  (seeded=True):  RGB sampled from a random pixel in the target image,
        giving the GA a much better colour starting point.

    Args:
        image_width:          Canvas width — bounds x-coordinates.
        image_height:         Canvas height — bounds y-coordinates.
        triangle_alpha_range: Inclusive (min, max) alpha range.
        target:               Required when seeded=True; the RGB target image.
        seeded:               Whether to sample colour from the target.

    Returns:
        A new Triangle with random geometry and the chosen colour strategy.
    """

    # All six vertex coordinates are independent uniform integers
    x1 = int(np.random.randint(0, image_width))
    y1 = int(np.random.randint(0, image_height))
    x2 = int(np.random.randint(0, image_width))
    y2 = int(np.random.randint(0, image_height))
    x3 = int(np.random.randint(0, image_width))
    y3 = int(np.random.randint(0, image_height))

    # Delegate colour selection to the shared helper
    r, g, b = _create_random_rgb(
        target=target,
        seeded=seeded,
        image_width=image_width,
        image_height=image_height,
    )

    return Triangle(
        x1=x1, y1=y1,
        x2=x2, y2=y2,
        x3=x3, y3=y3,
        r=r, g=g, b=b,
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
    Creates one individual — a list of randomly initialised triangles.

    Each individual is an independent candidate solution.  The order of
    triangles matters because later ones are painted on top of earlier ones
    during rendering (painter's algorithm).

    Args:
        n_triangles:          How many triangles compose this individual.
        image_width:          Canvas width.
        image_height:         Canvas height.
        triangle_alpha_range: Alpha range for each triangle.
        target:               Target image (needed when seeded=True).
        seeded:               Whether to sample triangle colours from target.

    Returns:
        List of n_triangles Triangle objects.
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
    Generates a full population of randomly initialised individuals.

    This is the generation-0 population that the GA starts from.  Each
    individual is created independently, so the starting diversity is high.

    Args:
        population_size:      Number of individuals in the population.
        n_triangles:          Triangles per individual.
        image_width:          Canvas width.
        image_height:         Canvas height.
        triangle_alpha_range: Alpha range for each triangle.
        target:               Target image (needed when seeded=True).
        seeded:               Whether to sample triangle colours from target.

    Returns:
        List of population_size individuals, each a list of Triangles.
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


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_triangle_alpha_range(
    triangle_alpha_range: AlphaRange,
) -> AlphaRange:
    """
    Validates and returns the alpha range tuple.

    Checks that:
      - It is a 2-tuple of integers.
      - Both values are in [0, 255].
      - min_alpha <= max_alpha.

    Args:
        triangle_alpha_range: The (min_alpha, max_alpha) tuple to validate.

    Returns:
        The validated (min_alpha, max_alpha) tuple unchanged.

    Raises:
        ValueError: If any of the above conditions are violated.
    """

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
    """
    Samples one alpha value uniformly from the configured range.

    Using randint(min, max + 1) makes the range fully inclusive on both ends,
    consistent with how the GA config specifies e.g. (255, 255) for opaque.

    Args:
        triangle_alpha_range: The (min_alpha, max_alpha) inclusive range.

    Returns:
        A random integer alpha value in [min_alpha, max_alpha].
    """

    min_alpha, max_alpha = validate_triangle_alpha_range(triangle_alpha_range)

    # +1 because np.random.randint's upper bound is exclusive
    return int(np.random.randint(min_alpha, max_alpha + 1))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _create_random_rgb(
    target: np.ndarray | None,
    seeded: bool,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int]:
    """
    Samples an RGB colour either uniformly or from a random target pixel.

    When seeded=False the colour is fully random, giving maximum diversity
    but no relationship to the target.

    When seeded=True a random pixel from the target is sampled.  This means
    generation-0 triangles already use colours that exist in the target image,
    which typically accelerates early convergence significantly.

    Args:
        target:       The target image array (required when seeded=True).
        seeded:       Whether to sample from the target.
        image_width:  Used to validate target dimensions.
        image_height: Used to validate target dimensions.

    Returns:
        A (r, g, b) tuple of integers in [0, 255].
    """

    if not seeded:
        # Fully random colour — no information from target
        return (
            int(np.random.randint(0, 256)),
            int(np.random.randint(0, 256)),
            int(np.random.randint(0, 256)),
        )

    # Validate the target before sampling from it
    seed_target = _validate_seed_target(
        target=target,
        image_width=image_width,
        image_height=image_height,
    )

    # Pick a random (x, y) pixel and read its RGB values
    sample_x = int(np.random.randint(0, image_width))
    sample_y = int(np.random.randint(0, image_height))
    sampled_rgb = seed_target[sample_y, sample_x]

    return int(sampled_rgb[0]), int(sampled_rgb[1]), int(sampled_rgb[2])


def _validate_seed_target(
    target: np.ndarray | None,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Validates that the target array is safe to sample RGB values from.

    Raises descriptive errors if the array is missing, has wrong type,
    wrong number of dimensions, or wrong spatial dimensions.

    Args:
        target:       The array to validate.
        image_width:  Expected width in pixels.
        image_height: Expected height in pixels.

    Returns:
        The validated target array unchanged.

    Raises:
        ValueError: If any validation check fails.
    """

    if target is None:
        raise ValueError("target must be provided when seeded=True.")
    if not isinstance(target, np.ndarray):
        raise ValueError("target must be a NumPy array when seeded=True.")
    if target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("target must have shape (image_height, image_width, 3).")
    if target.shape[0] != image_height or target.shape[1] != image_width:
        raise ValueError("target dimensions must match image_height and image_width.")

    return target
