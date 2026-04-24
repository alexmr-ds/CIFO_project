"""Loads and resizes the target image into a NumPy RGB array."""

import numpy as np
from PIL import Image

def load_target_image(path: str) -> np.ndarray:
    """
    Loads the target image and converts it to a NumPy array.

    The output is a (H, W, 3) array with RGB values in [0, 255].
    """

    image = Image.open(path).convert("RGB")
    image = image.resize((300, 400))  # ensure correct size

    return np.array(image)
