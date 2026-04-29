"""Computes image fitness metrics for triangle-based GA runs."""

from collections.abc import Callable

import numpy as np

FitnessFunction = Callable[[np.ndarray, np.ndarray], float]


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalizes an RGB image to float32 values in [0, 1]."""

    return image.astype(np.float32) / np.float32(255.0)


def _to_luminance(image: np.ndarray) -> np.ndarray:
    """Converts a normalized RGB image to luminance."""

    return np.tensordot(image, np.array([0.299, 0.587, 0.114], dtype=np.float32), axes=([2], [0]))


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Computes per-pixel gradient magnitude with finite differences."""

    grad_y, grad_x = np.gradient(image)

    return np.sqrt((grad_x**2) + (grad_y**2))


def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Root Mean Squared Error between two RGB images."""

    normalized_target = _normalize_image(target)
    normalized_generated = _normalize_image(generated)
    mse = np.mean((normalized_target - normalized_generated) ** 2)

    return float(np.sqrt(mse))


def compute_structure_loss(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes an edge-structure loss from luminance gradient differences."""

    normalized_target = _normalize_image(target)
    normalized_generated = _normalize_image(generated)
    target_edges = _gradient_magnitude(_to_luminance(normalized_target))
    generated_edges = _gradient_magnitude(_to_luminance(normalized_generated))

    return float(np.mean(np.abs(target_edges - generated_edges)))


def compute_rmse_plus_structure(
    target: np.ndarray,
    generated: np.ndarray,
    rmse_weight: float = 1.0,
    structure_weight: float = 0.35,
) -> float:
    """Computes a weighted blend of RMSE and edge-structure loss."""

    if rmse_weight < 0.0 or structure_weight < 0.0:
        raise ValueError("rmse_weight and structure_weight must be non-negative.")
    if rmse_weight == 0.0 and structure_weight == 0.0:
        raise ValueError("At least one fitness weight must be greater than zero.")

    rmse = compute_rmse(target, generated)
    structure_loss = compute_structure_loss(target, generated)

    return float((rmse_weight * rmse) + (structure_weight * structure_loss))


def make_rmse_structure_fitness(
    rmse_weight: float = 1.0,
    structure_weight: float = 0.35,
) -> FitnessFunction:
    """Builds a configured RMSE+structure fitness callable."""

    if rmse_weight < 0.0 or structure_weight < 0.0:
        raise ValueError("rmse_weight and structure_weight must be non-negative.")
    if rmse_weight == 0.0 and structure_weight == 0.0:
        raise ValueError("At least one fitness weight must be greater than zero.")

    def blended_fitness(target: np.ndarray, generated: np.ndarray) -> float:
        return compute_rmse_plus_structure(
            target=target,
            generated=generated,
            rmse_weight=rmse_weight,
            structure_weight=structure_weight,
        )

    return blended_fitness
