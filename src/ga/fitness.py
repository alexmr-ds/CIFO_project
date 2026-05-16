"""
Fitness function for the triangle-based image approximation GA.

A fitness function measures how different the rendered candidate image is
from the target image.  Lower values are better (the GA minimises fitness).
"""

from collections.abc import Callable

import numpy as np

# Type alias: any callable that takes two (H,W,3) arrays and returns a float
FitnessFunction = Callable[[np.ndarray, np.ndarray], float]


def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between the target image and
    a candidate image rendered from a list of triangles.

    How a candidate image is built
    -------------------------------
    Each individual in the GA is a list of 100 triangles (N_TRIANGLES = 100).
    Every triangle has 10 parameters:

        Geometry  (6):  x1, y1, x2, y2, x3, y3  — vertex coordinates in pixels,
                        each in the range [0, image_width) or [0, image_height).
        Colour    (4):  r, g, b  — red, green, blue channels in [0, 255].
                        a        — alpha (opacity) in [0, 255], where 0 is fully
                                   transparent and 255 is fully opaque.

    Before fitness is evaluated, the rendering pipeline draws all 100 triangles
    on a blank canvas in order (back to front), blending each triangle's colour
    with whatever is already on the canvas using its alpha value.  The result is
    a 300 × 400 RGB image — the ``generated`` array passed to this function.

    How RMSE is computed
    --------------------
    1. Both images (target and generated) are normalised from uint8 [0, 255]
       to float32 [0.0, 1.0] so the result is scale-independent.
    2. The squared difference is computed for every pixel and every colour
       channel (H × W × 3 values in total).
    3. The mean of those squared differences is taken (MSE).
    4. The square root gives RMSE — a single number representing the average
       per-channel pixel error across the whole image.

    A value of 0.0 means the rendered image is a perfect match.
    The GA minimises this value over successive generations.

    Args:
        target:    The reference image, shape (H, W, 3), values in [0, 255].
        generated: The candidate rendered image, shape (H, W, 3), values in [0, 255].

    Returns:
        RMSE as a float in [0, 1]; lower is better.
    """
    target_f    = target.astype(np.float32)    / np.float32(255.0)
    generated_f = generated.astype(np.float32) / np.float32(255.0)
    return float(np.sqrt(np.mean((target_f - generated_f) ** 2)))
