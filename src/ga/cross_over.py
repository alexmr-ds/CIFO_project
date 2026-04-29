"""Implements crossover operators for triangle-based individuals."""

import copy

import numpy as np

from .. import population

# ----------------------------------
# One children cross-over
# ----------------------------------


def single_point_crossover(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> list[population.Triangle]:
    """Creates one child using single-point triangle-level crossover.

    Splits both parents at a single random point and concatenates the
    first segment of parent1 with the second segment of parent2.

    Args:
        parent1: First parent individual (list of Triangles).
        parent2: Second parent individual (list of Triangles).
        crossover_rate: Probability of performing crossover. If not
            triggered, a random parent is returned as-is.

    Returns:
        A new child individual as a deep-copied list of Triangles.
    """
    if len(parent1) < 2 or len(parent2) < 2:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    if np.random.random() >= crossover_rate:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    crossover_point = int(np.random.randint(1, min(len(parent1), len(parent2))))
    child = parent1[:crossover_point] + parent2[crossover_point:]

    return copy.deepcopy(child)


def two_point_crossover(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> list[population.Triangle]:
    """Creates one child using two-point triangle-level crossover.

    Picks two random cut points and swaps the middle segment between
    parents, producing: [parent1 | parent2 | parent1].
    This preserves both ends of parent1 while injecting a contiguous
    block of genetic material from parent2, reducing positional
    disruption compared to single-point crossover.

    Args:
        parent1: First parent individual (list of Triangles).
        parent2: Second parent individual (list of Triangles).
        crossover_rate: Probability of performing crossover. If not
            triggered, a random parent is returned as-is.

    Returns:
        A new child individual as a deep-copied list of Triangles.
    """
    if len(parent1) < 3 or len(parent2) < 3:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    if np.random.random() >= crossover_rate:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    min_len = min(len(parent1), len(parent2))
    point1, point2 = sorted(np.random.choice(range(1, min_len), size=2, replace=False))

    child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]

    return copy.deepcopy(child)


# ----------------------------------
# Two children cross-over
# ----------------------------------
def single_point_crossover_two_children(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> tuple[list[population.Triangle], list[population.Triangle]]:
    """Creates two children using single-point triangle-level crossover.

    Splits both parents at a single random point and returns the two
    complementary recombinations:

        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

    This preserves more genetic material from both parents than producing
    only one child.

    Args:
        parent1: First parent individual (list of Triangles).
        parent2: Second parent individual (list of Triangles).
        crossover_rate: Probability of performing crossover. If not
            triggered, deep copies of both parents are returned.

    Returns:
        A tuple containing two new child individuals.
    """
    if len(parent1) < 2 or len(parent2) < 2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    if np.random.random() >= crossover_rate:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    min_len = min(len(parent1), len(parent2))
    crossover_point = int(np.random.randint(1, min_len))

    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return copy.deepcopy(child1), copy.deepcopy(child2)


def two_point_crossover_two_children(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> tuple[list[population.Triangle], list[population.Triangle]]:
    """Creates two children using two-point triangle-level crossover.

    Selects two cut points and swaps the middle segment between parents,
    returning the complementary recombinations:

        child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]

    This preserves both edge segments from each parent and exchanges a
    contiguous block, improving genetic diversity while maintaining
    structural coherence.

    Args:
        parent1: First parent individual (list of Triangles).
        parent2: Second parent individual (list of Triangles).
        crossover_rate: Probability of performing crossover.

    Returns:
        A tuple containing two new child individuals.
    """
    if len(parent1) < 3 or len(parent2) < 3:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    if np.random.random() >= crossover_rate:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    min_len = min(len(parent1), len(parent2))

    point1, point2 = sorted(np.random.choice(range(1, min_len), size=2, replace=False))

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return copy.deepcopy(child1), copy.deepcopy(child2)


def whole_triangle_crossover(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> list[population.Triangle]:
    """Creates one child by inheriting each triangle slot from one parent."""

    if len(parent1) != len(parent2):
        raise ValueError("whole_triangle_crossover requires equal-length parents.")

    if np.random.random() >= crossover_rate:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    child: list[population.Triangle] = []
    for triangle1, triangle2 in zip(parent1, parent2, strict=True):
        source_triangle = triangle1 if np.random.random() < 0.5 else triangle2
        child.append(copy.deepcopy(source_triangle))

    return child
