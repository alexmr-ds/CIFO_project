"""Implements crossover operators for triangle-based individuals."""

import copy

import numpy as np

from .. import population


def single_point_crossover(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> list[population.Triangle]:
    """Creates one child using single-point triangle-level crossover."""

    if len(parent1) < 2 or len(parent2) < 2:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    if np.random.random() >= crossover_rate:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    crossover_point = int(np.random.randint(1, min(len(parent1), len(parent2))))
    child = parent1[:crossover_point] + parent2[crossover_point:]

    return copy.deepcopy(child)
