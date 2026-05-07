"""
Loads the target image from disk and converts it to a NumPy RGB array.

The GA fitness function compares the rendered triangle image against this
array pixel-by-pixel, so the format must be consistent:
  - Shape  : (H, W, 3)  — height × width × RGB channels
  - Dtype  : uint8 (values 0–255)
  - Size   : always resized to (300 width × 400 height) to keep compute manageable
"""

import numpy as np
from PIL import Image


def load_target_image(path: str) -> np.ndarray:
    """
    Loads a target image from disk and returns it as a NumPy RGB array.

    The image is:
      1. Opened from the given path via Pillow.
      2. Converted to RGB (drops any alpha channel, handles greyscale).
      3. Resized to a fixed 300×400 canvas so all experiments use the
         same resolution and the render/fitness cost stays predictable.
      4. Converted to a NumPy uint8 array with shape (400, 300, 3).

    Args:
        path: Absolute or relative file path to the target image.

    Returns:
        NumPy array of shape (H, W, 3) with uint8 RGB values in [0, 255].
    """

    # Open the image and normalize it to plain RGB (no alpha, no greyscale)
    image = Image.open(path).convert("RGB")

    # Fix the canvas size so every experiment runs on identical pixel grids. The original image size is 300×400.
    image = image.resize((300, 400))

    return np.array(image)
