"""
Renders triangle individuals as PIL images or NumPy arrays.

A triangle individual is a list of Triangle dataclasses.  Each triangle
carries three vertex coordinates (x1,y1), (x2,y2), (x3,y3) and an RGBA
colour.  This module turns that abstract representation into a pixel image
so the GA fitness function can compare it against the target.

Rendering strategy
------------------
Triangles are drawn one at a time onto a transparent overlay, then
alpha-composited onto the canvas.  This correctly handles semi-transparent
triangles: later triangles blend on top of earlier ones without destroying
the colour information already present underneath.
"""

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
    Renders a list of triangles into an RGB PIL Image.

    The canvas starts as a solid black (or custom) background.  Each
    triangle is painted onto a fresh transparent overlay, which is then
    alpha-composited onto the canvas.  The final image is returned as RGB
    (no alpha channel) so it can be directly compared to the target array.

    Args:
        individual:       List of Triangle objects to draw.
        image_width:      Canvas width in pixels (default 300).
        image_height:     Canvas height in pixels (default 400).
        background_color: RGB tuple for the initial canvas colour.

    Returns:
        A PIL Image in RGB mode with the triangles composited on top.
    """

    # Check whether every triangle in this individual is fully opaque.
    # When alpha is fixed at 255 (the baseline config) we can use a fast path
    # that draws directly onto a single canvas without any per-triangle image
    # allocation or alpha compositing — roughly 5-6× faster.
    all_opaque = all(t.a == 255 for t in individual)

    if all_opaque:
        # --- Fast path: all triangles are fully opaque ---
        # Draw directly onto an RGB canvas with a single ImageDraw.
        # No overlay images are created, no alpha_composite calls needed.
        image = Image.new("RGB", (image_width, image_height), background_color)
        draw = ImageDraw.Draw(image, "RGB")
        for triangle in individual:
            vertices = [
                (triangle.x1, triangle.y1),
                (triangle.x2, triangle.y2),
                (triangle.x3, triangle.y3),
            ]
            draw.polygon(vertices, fill=(triangle.r, triangle.g, triangle.b))
        return image

    # --- Slow path: at least one semi-transparent triangle ---
    # Allocate one reusable overlay and clear it between triangles instead
    # of creating a brand-new image object for each triangle.
    # This is still correct for any alpha value and avoids repeated allocation.
    image = Image.new("RGBA", (image_width, image_height), background_color + (255,))
    overlay = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
    clear_box = (0, 0, image_width, image_height)

    for triangle in individual:
        # Clear the overlay by pasting a transparent rectangle over it —
        # much faster than allocating a new Image object each iteration.
        overlay.paste((0, 0, 0, 0), clear_box)

        draw = ImageDraw.Draw(overlay, "RGBA")
        vertices = [
            (triangle.x1, triangle.y1),
            (triangle.x2, triangle.y2),
            (triangle.x3, triangle.y3),
        ]
        draw.polygon(vertices, fill=(triangle.r, triangle.g, triangle.b, triangle.a))

        # Blend the new triangle on top of everything drawn so far
        image = Image.alpha_composite(image, overlay)

    # Strip the alpha channel before returning — the fitness function expects
    # plain RGB arrays matching the shape of the target image.
    return image.convert("RGB")


def image_to_array(
    source: list[population.Triangle] | Image.Image,
    image_width: int = population.IMAGE_WIDTH,
    image_height: int = population.IMAGE_HEIGHT,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Converts a triangle individual or a PIL Image into a float32 NumPy array.

    This is the bridge between the rendered image and the fitness function:
    the GA calls this to turn a candidate individual into a pixel array that
    can be compared against the target array via RMSE.

    Args:
        source:           Either a list of Triangle objects or an existing PIL Image.
        image_width:      Canvas width used when rendering from triangles.
        image_height:     Canvas height used when rendering from triangles.
        background_color: Background colour used when rendering from triangles.

    Returns:
        NumPy array of shape (H, W, 3) with float32 values in [0, 255].
    """

    if isinstance(source, Image.Image):
        # Already a PIL Image — just normalize the colour mode
        image = source.convert("RGB")
    else:
        # Render the list of triangles first, then convert
        image = render_individual(
            source,
            image_width=image_width,
            image_height=image_height,
            background_color=background_color,
        )

    # float32 is used (not uint8) because the fitness functions normalize
    # to [0, 1] by dividing by 255.0, and float32 avoids precision loss.
    return np.array(image, dtype=np.float32)
