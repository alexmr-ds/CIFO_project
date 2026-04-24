"""Renders triangle individuals as images or NumPy arrays."""

import numpy as np
from PIL import Image, ImageDraw

from . import population


def render_individual(
    individual: list[population.Triangle],
    image_width: int = population.IMAGE_WIDTH,
    image_height: int = population.IMAGE_HEIGHT,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Renders an individual into an RGB image for visualization.

    Each triangle is drawn sequentially using its vertex coordinates and RGBA color.
    Later triangles are composited over earlier ones, preserving transparency.
    """

    image = Image.new("RGBA", (image_width, image_height), background_color + (255,))

    for triangle in individual:
        overlay = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))

        draw = ImageDraw.Draw(overlay, "RGBA")

        vertices = [
            (triangle.x1, triangle.y1),
            (triangle.x2, triangle.y2),
            (triangle.x3, triangle.y3),
        ]

        color = (triangle.r, triangle.g, triangle.b, triangle.a)

        draw.polygon(vertices, fill=color)

        image = Image.alpha_composite(image, overlay)

    return image.convert("RGB")


def image_to_array(
    source: list[population.Triangle] | Image.Image,
    image_width: int = population.IMAGE_WIDTH,
    image_height: int = population.IMAGE_HEIGHT,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Converts an individual or a PIL image into a NumPy RGB array.

    The output is a (H, W, 3) float32 array ready for RMSE computation.
    """

    if isinstance(source, Image.Image):
        image = source.convert("RGB")
    else:
        image = render_individual(
            source,
            image_width=image_width,
            image_height=image_height,
            background_color=background_color,
        )

    return np.array(image, dtype=np.float32)
