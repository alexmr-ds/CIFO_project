"""
Crossover operators for triangle-based individuals.

Crossover (recombination) takes two parent individuals and combines their
genetic material to create one or two children.  The idea is that if two
different parents each carry some good triangles, a child that inherits the
best from both might outperform either parent.

All operators work at the *triangle level*: the crossover point splits the
list of triangles, not the internal fields of a single triangle.

Available operators
-------------------
single_point_crossover (→ 1 child)
    Split both parents at a single random point; take the first half of
    parent1 and the second half of parent2.  Simple and effective.

two_point_crossover (→ 1 child)
    Split at two random points; the child inherits parent1's ends and
    parent2's middle segment.  Preserves more of each parent's structure.

two_point_crossover_two_children (→ 2 children)
    Same as two-point but produces both complementary recombinations,
    doubling the number of offspring per pair of parents.

Crossover rate
--------------
Every operator accepts a ``crossover_rate`` probability.  When the rate is
not triggered (random draw ≥ rate), a random parent is returned unchanged.
This allows pure reproduction (no recombination) to co-exist with crossover.
"""

import copy

import numpy as np

from .. import population


# ---------------------------------------------------------------------------
# One-child operators
# ---------------------------------------------------------------------------

def single_point_crossover(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> list[population.Triangle]:
    """
    Creates one child using single-point triangle-level crossover.

    A random cut point is chosen along the triangle list.  The child
    inherits all triangles *before* the cut from parent1 and all triangles
    *from* the cut onward from parent2.

    Example (8 triangles, cut at index 3):
        parent1: [A B C | D E F G H]
        parent2: [a b c | d e f g h]
        child  : [A B C   d e f g h]

    If crossover is not triggered, one parent is returned at random.

    Args:
        parent1:        First parent individual.
        parent2:        Second parent individual.
        crossover_rate: Probability of performing crossover (0.0–1.0).

    Returns:
        A new child individual as a deep-copied list of Triangles.
    """

    # Need at least 2 triangles in each parent to have a meaningful split
    if len(parent1) < 2 or len(parent2) < 2:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    # With probability (1 - crossover_rate) skip crossover and clone a parent
    if np.random.random() >= crossover_rate:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    # Pick a cut point that leaves at least one triangle from each parent
    crossover_point = int(np.random.randint(1, min(len(parent1), len(parent2))))
    child = parent1[:crossover_point] + parent2[crossover_point:]

    # Deep copy so mutations to the child never affect the parents
    return copy.deepcopy(child)


def two_point_crossover(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> list[population.Triangle]:
    """
    Creates one child using two-point triangle-level crossover.

    Two random cut points divide the triangle lists into three segments.
    The child inherits the *outer* segments from parent1 and the *middle*
    segment from parent2:

        child = parent1[:p1] + parent2[p1:p2] + parent1[p2:]

    This preserves both ends of parent1 while injecting a contiguous block
    of genetic material from parent2.  Compared to single-point crossover,
    it reduces positional disruption for triangles near the ends of the list.

    Args:
        parent1:        First parent individual.
        parent2:        Second parent individual.
        crossover_rate: Probability of performing crossover (0.0–1.0).

    Returns:
        A new child individual as a deep-copied list of Triangles.
    """

    # Need at least 3 triangles to fit two distinct cut points
    if len(parent1) < 3 or len(parent2) < 3:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    if np.random.random() >= crossover_rate:
        fallback_parent = parent1 if np.random.random() < 0.5 else parent2
        return copy.deepcopy(fallback_parent)

    min_len = min(len(parent1), len(parent2))

    # Draw two distinct cut points and sort them so point1 < point2
    point1, point2 = sorted(np.random.choice(range(1, min_len), size=2, replace=False))

    child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]

    return copy.deepcopy(child)


# ---------------------------------------------------------------------------
# Two-child operators
# ---------------------------------------------------------------------------

def single_point_crossover_two_children(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> tuple[list[population.Triangle], list[population.Triangle]]:
    """
    Creates two complementary children using single-point triangle-level crossover.

    Uses the same single cut point as ``single_point_crossover`` but produces
    both recombinations simultaneously:

        child1 = parent1[:cut] + parent2[cut:]
        child2 = parent2[:cut] + parent1[cut:]

    Returning two children per pair of parents increases the offspring pool
    without additional parent selections.

    Args:
        parent1:        First parent individual.
        parent2:        Second parent individual.
        crossover_rate: Probability of performing crossover (0.0–1.0).

    Returns:
        A tuple of two new child individuals, each a deep-copied list of Triangles.
    """

    if len(parent1) < 2 or len(parent2) < 2:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    if np.random.random() >= crossover_rate:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    cut = int(np.random.randint(1, min(len(parent1), len(parent2))))

    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]

    return copy.deepcopy(child1), copy.deepcopy(child2)


def two_point_crossover_two_children(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> tuple[list[population.Triangle], list[population.Triangle]]:
    """
    Creates two complementary children using two-point triangle-level crossover.

    Uses the same two cut points as ``two_point_crossover`` but produces
    both recombinations simultaneously:

        child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]

    Returning two children per pair of parents increases the offspring pool
    without requiring additional parent selections, which can speed up
    population turnover.

    Args:
        parent1:        First parent individual.
        parent2:        Second parent individual.
        crossover_rate: Probability of performing crossover (0.0–1.0).

    Returns:
        A tuple of two new child individuals, each a deep-copied list of Triangles.
    """

    if len(parent1) < 3 or len(parent2) < 3:
        # Not enough triangles for two cut points — return copies of parents
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    if np.random.random() >= crossover_rate:
        # Crossover not triggered — return unmodified copies
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    min_len = min(len(parent1), len(parent2))

    # Two distinct, sorted cut points within the valid index range
    point1, point2 = sorted(np.random.choice(range(1, min_len), size=2, replace=False))

    # Both complementary recombinations: each child gets what the other doesn't
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return copy.deepcopy(child1), copy.deepcopy(child2)
