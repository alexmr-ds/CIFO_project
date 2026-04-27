"""Computes image fitness metrics for triangle-based GA runs."""

import numpy as np


def compute_rmse(target: np.ndarray, generated: np.ndarray) -> float:
    """Computes the Root Mean Squared Error between two RGB images."""

    # Ensure float math to avoid uint8 overflow during subtraction.
    target = target.astype(np.float32) / np.float32(255.0)
    generated = generated.astype(np.float32) / np.float32(255.0)

    mse = np.mean((target - generated) ** 2)
    rmse = np.sqrt(mse)

    return float(rmse)
