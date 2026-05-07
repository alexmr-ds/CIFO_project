"""
Top-level package for the genetic image approximation project.

This package bundles all sub-modules needed to:
  - Load a target image (load_image)
  - Represent and generate triangle populations (population)
  - Render triangles to pixel images (rendering)
  - Run and analyse the Genetic Algorithm (ga)

Importing `src` automatically exposes all four sub-modules so notebooks
only need `from src import ...` rather than multiple deep imports.
"""

from . import ga, load_image, population, rendering
from .ga import fitness

__all__ = ["load_image", "population", "rendering", "fitness", "ga"]
