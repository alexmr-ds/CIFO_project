"""Computes RMSE between target and generated RGB images."""

import numpy as np

def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error between two images.

    Both images must have shape (H, W, 3).
    """

    # Ensure float to avoid overflow
    target = target.astype(np.float32)
    generated = generated.astype(np.float32)

    mse = np.mean((target - generated) ** 2)
    rmse = np.sqrt(mse)

    return rmse
