"""
Fitness functions for the triangle-based image approximation GA.

A fitness function measures how different the rendered candidate image is
from the target image.  Lower values are better (the GA minimises fitness).

Available metrics
-----------------
compute_rmse
    Root Mean Squared Error across all pixels and channels.
    This is the primary fitness function used throughout the project.
    Simple, fast, and directly measures pixel-level colour accuracy.

compute_structure_loss
    Edge-based structural similarity loss using luminance gradients.
    Penalises mismatched edges/contours regardless of colour.

compute_rmse_plus_structure
    Weighted blend of RMSE and structure loss.
    Useful when you want the GA to preserve edges as well as colour.
"""

from collections.abc import Callable

import numpy as np

# Type alias: any callable that takes two (H,W,3) arrays and returns a float
FitnessFunction = Callable[[np.ndarray, np.ndarray], float]


# ---------------------------------------------------------------------------
# Private normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalises an RGB image from uint8 [0, 255] to float32 [0, 1].

    Normalising before computing differences ensures that the loss values
    are independent of the pixel value range and always lie in [0, 1].

    Args:
        image: NumPy array with shape (H, W, 3), values in [0, 255].

    Returns:
        float32 array with the same shape, values in [0.0, 1.0].
    """
    return image.astype(np.float32) / np.float32(255.0)


def _to_luminance(image: np.ndarray) -> np.ndarray:
    """
    Converts a normalised RGB image to a 2-D luminance (greyscale) map.

    Uses the standard ITU-R BT.601 luma coefficients:
        Y = 0.299 R + 0.587 G + 0.114 B

    The green channel carries the most perceptual weight because the human
    eye is most sensitive to green light.

    Args:
        image: Normalised float32 array with shape (H, W, 3).

    Returns:
        float32 array with shape (H, W) containing per-pixel luminance.
    """
    return np.tensordot(image, np.array([0.299, 0.587, 0.114], dtype=np.float32), axes=([2], [0]))


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Computes the per-pixel gradient magnitude of a 2-D image.

    Uses central finite differences (np.gradient) for both horizontal and
    vertical directions, then combines them as the Euclidean magnitude.
    High values indicate strong edges or sharp colour transitions.

    Args:
        image: 2-D float32 array (e.g. a luminance map).

    Returns:
        2-D float32 array of gradient magnitudes, same shape as input.
    """
    grad_y, grad_x = np.gradient(image)
    return np.sqrt((grad_x**2) + (grad_y**2))


# ---------------------------------------------------------------------------
# Public fitness functions
# ---------------------------------------------------------------------------

def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between two RGB images.

    RMSE is the primary fitness metric used in this project.  It measures
    the average per-pixel colour difference between the rendered candidate
    and the target image.

    Both images are normalised to [0, 1] before computing the error so
    the result is scale-independent and always in [0, 1].
    A value of 0 means perfect pixel-by-pixel reproduction.
    A value of ~0.29 is typical for a completely random initialisation.

    Args:
        target:    The reference image, shape (H, W, 3).
        generated: The candidate rendered image, shape (H, W, 3).

    Returns:
        RMSE as a float in [0, 1]; lower is better.
    """

    normalized_target = _normalize_image(target)
    normalized_generated = _normalize_image(generated)

    # Mean of squared differences across all pixels and all 3 channels
    mse = np.mean((normalized_target - normalized_generated) ** 2)

    return float(np.sqrt(mse))


def compute_structure_loss(target: np.ndarray, generated: np.ndarray) -> float:
    """
    Computes a structural loss based on the difference in edge gradients.

    Instead of comparing raw pixel colours, this metric compares the *shape*
    of edges in both images.  It converts each image to luminance, computes
    the gradient magnitude (edge strength map), and measures their mean
    absolute difference.

    This is useful as a secondary term to encourage the GA to preserve the
    outline/contour structure of the target, not just its colours.

    Args:
        target:    The reference image, shape (H, W, 3).
        generated: The candidate rendered image, shape (H, W, 3).

    Returns:
        Mean absolute edge-gradient difference as a float; lower is better.
    """

    normalized_target = _normalize_image(target)
    normalized_generated = _normalize_image(generated)

    # Extract edge maps from luminance representations of each image
    target_edges = _gradient_magnitude(_to_luminance(normalized_target))
    generated_edges = _gradient_magnitude(_to_luminance(normalized_generated))

    return float(np.mean(np.abs(target_edges - generated_edges)))


def compute_rmse_plus_structure(
    target: np.ndarray,
    generated: np.ndarray,
    rmse_weight: float = 1.0,
    structure_weight: float = 0.35,
) -> float:
    """
    Computes a weighted combination of RMSE and structural edge loss.

    Blending both metrics encourages the GA to simultaneously:
      - Match pixel colours (RMSE term)
      - Preserve edge shapes and contours (structure term)

    The default weights (1.0 RMSE + 0.35 structure) treat colour accuracy
    as the dominant objective while still penalising structural mismatch.

    Args:
        target:           The reference image, shape (H, W, 3).
        generated:        The candidate rendered image, shape (H, W, 3).
        rmse_weight:      Multiplier for the RMSE component (default 1.0).
        structure_weight: Multiplier for the structure component (default 0.35).

    Returns:
        Combined fitness score as a float; lower is better.

    Raises:
        ValueError: If both weights are zero or either is negative.
    """

    if rmse_weight < 0.0 or structure_weight < 0.0:
        raise ValueError("rmse_weight and structure_weight must be non-negative.")
    if rmse_weight == 0.0 and structure_weight == 0.0:
        raise ValueError("At least one fitness weight must be greater than zero.")

    rmse = compute_rmse(target, generated)
    structure_loss = compute_structure_loss(target, generated)

    return float((rmse_weight * rmse) + (structure_weight * structure_loss))
