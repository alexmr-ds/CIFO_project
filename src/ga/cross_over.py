"""
Crossover operators for triangle-based individuals.

Crossover, also called recombination, takes two parent individuals and combines
their genetic material to create one or two children. The idea is that if two
different parents each carry some useful triangles, a child that inherits useful
parts from both may outperform either parent.

Most operators work directly at the triangle-list level: crossover splits or
recombines the list of triangles, not the internal fields of a single triangle.

Available operators
-------------------
single_point_crossover (→ 1 child)
    Split both parents at a single random point; take the first part from
    parent1 and the remaining part from parent2. Simple and effective.

single_point_crossover_two_children (→ 2 children)
    Same as single-point crossover but returns both complementary children:
    one child starts from parent1 and the other starts from parent2.

two_point_crossover (→ 1 child)
    Split at two random points; the child inherits parent1's outer segments
    and parent2's middle segment. Preserves more of each parent's structure.

two_point_crossover_two_children (→ 2 children)
    Same as two-point crossover but produces both complementary recombinations,
    doubling the number of offspring per pair of parents.

cycle_crossover (→ 2 children)
    Index-based Cycle Crossover (CX). Since Triangle objects are not unique
    permutation genes, synthetic randomized index permutations are generated
    first. Cycles are then traced over those index permutations, and triangles
    are inherited according to whether their index belongs to the selected cycle.

pmx_crossover (→ 2 children)
    Index-based Partially Matched Crossover (PMX). Standard PMX assumes unique
    permutation genes, so this implementation first creates synthetic randomized
    index permutations. PMX is applied to those indices, and the resulting
    index arrays are used to reorder and mix the Triangle objects.

Crossover rate
--------------
Every operator accepts a ``crossover_rate`` probability. When the rate is not
triggered, the parents are returned unchanged or one parent is cloned, depending
on whether the operator produces one child or two children. This allows pure
reproduction, meaning no recombination, to co-exist with crossover.
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


def cycle_crossover(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> tuple[list[population.Triangle], list[population.Triangle]]:
    """
    Performs an index-based cycle crossover on two individuals, returning two new offspring.

    Standard Cycle Crossover (CX) requires parent representations to be permutations
    of unique elements to successfully trace mapping cycles. Because candidate images
    are lists of Triangle objects (which do not inherently form unique permutations),
    this function generates synthetic, randomized index permutations to define the cycles.

    During crossover:
    - If a specific list index belongs to the randomly generated cycle, the offspring
      inherits the triangle at that position from its primary parent.
    - If the index is not in the cycle, it crosses over and inherits the triangle
      from the other parent.

    Args:
        parent1:        First parent individual.
        parent2:        Second parent individual.
        crossover_rate: Probability of performing crossover (0.0–1.0).

    Returns:
        A tuple of two new child individuals as deep-copied lists of Triangles.
    """

    if np.random.random() >= crossover_rate:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    # Ensure parents have content
    if len(parent1) == 0 or len(parent2) == 0:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    n = len(parent1)

    # 1. Create permutations of the INDICES (The "INVERSION" step)
    idx_perm1 = np.random.permutation(n).tolist()
    idx_perm2 = np.random.permutation(n).tolist()

    # 2. Randomly choose a starting POSITION in the index permutation array
    initial_pos = np.random.randint(0, n)
    cycle_positions = [initial_pos]
    current_pos = initial_pos

    # 3. Traverse the cycle on the INDEX PERMUTATIONS
    while True:
        value_perm2 = idx_perm2[current_pos]
        next_pos = idx_perm1.index(value_perm2)

        if next_pos == initial_pos:
            break

        cycle_positions.append(next_pos)
        current_pos = next_pos

    # 4. Extract the actual indices that belong to the cycle
    cycle_indices = set(idx_perm1[pos] for pos in cycle_positions)

    # 5. Create offspring based on the cycle with DEEP COPYING
    offspring1 = []
    offspring2 = []

    for i in range(n):
        if i in cycle_indices:
            # Keep the triangle from the original parent, but copy it
            offspring1.append(copy.deepcopy(parent1[i]))
            offspring2.append(copy.deepcopy(parent2[i]))
        else:
            # Cross over (take triangle from the other parent), but copy it
            offspring1.append(copy.deepcopy(parent2[i]))
            offspring2.append(copy.deepcopy(parent1[i]))

    return offspring1, offspring2


def pmx_crossover(
    parent1: list[population.Triangle],
    parent2: list[population.Triangle],
    crossover_rate: float,
) -> tuple[list[population.Triangle], list[population.Triangle]]:
    """
    Performs an index-based Partially Matched Crossover (PMX) on two individuals.

    Standard PMX requires parent representations to be permutations of unique
    elements to successfully resolve duplicate genes during crossover. Because
    candidate images are lists of Triangle objects (which do not inherently form
    unique permutations), this function generates synthetic, randomized index
    permutations to act as unique IDs. It runs the strict PMX algorithm on these
    indices and uses the resulting arrays to reorder and mix the Triangle objects.

    The PMX process:
    1. A random contiguous "swath" (substring) is selected via two cut points.
    2. Inside the swath, child indices are directly copied from their respective parents.
    3. Outside the swath, indices are inherited from the opposite parent. If a duplicate
       index is found, it is resolved using a mapping dictionary created from the swath.
    """

    if np.random.random() >= crossover_rate:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    if len(parent1) == 0 or len(parent2) == 0:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    n = len(parent1)

    # 1. INITIAL INDEX SHUFFLING (These act as the unique IDs for PMX)
    idx_perm1 = np.random.permutation(n).tolist()
    idx_perm2 = np.random.permutation(n).tolist()

    # 2. PMX LOGIC
    # Randomly select two cut points for the swath
    cx_point1, cx_point2 = sorted(np.random.choice(n, 2, replace=False))

    child_idx1 = [-1] * n
    child_idx2 = [-1] * n

    # Step A: Copy the swaths between the cut points
    for i in range(cx_point1, cx_point2 + 1):
        child_idx1[i] = idx_perm1[i]
        child_idx2[i] = idx_perm2[i]

    # Step B: Build the mappings to fix duplicates
    mapping1 = {idx_perm1[i]: idx_perm2[i] for i in range(cx_point1, cx_point2 + 1)}
    mapping2 = {idx_perm2[i]: idx_perm1[i] for i in range(cx_point1, cx_point2 + 1)}

    # Step C: Fill the positions outside the swath, resolving duplicates via mapping
    for i in range(n):
        if not (cx_point1 <= i <= cx_point2):
            # For Child 1: pull from perm2, resolve transitive mapping collisions
            val1 = idx_perm2[i]
            while val1 in mapping1:
                val1 = mapping1[val1]
            child_idx1[i] = val1

            # For Child 2: pull from perm1, resolve transitive mapping collisions
            val2 = idx_perm1[i]
            while val2 in mapping2:
                val2 = mapping2[val2]
            child_idx2[i] = val2

    # 3. REORDER AT THE END
    offspring1 = []
    offspring2 = []

    for i in range(n):
        # Assemble Child 1
        if cx_point1 <= i <= cx_point2:
            # Inside the swath: index originated from Parent 1
            offspring1.append(copy.deepcopy(parent1[child_idx1[i]]))

            # Inside the swath: index originated from Parent 2
            offspring2.append(copy.deepcopy(parent2[child_idx2[i]]))

        else:
            # Outside the swath: index originated from Parent 2
            offspring1.append(copy.deepcopy(parent2[child_idx1[i]]))

            # Outside the swath: index originated from Parent 1
            offspring2.append(copy.deepcopy(parent1[child_idx2[i]]))

    return offspring1, offspring2
