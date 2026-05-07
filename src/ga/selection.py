"""
Parent selection strategies for the genetic algorithm.

Selection decides *which* individuals get to reproduce.  Applying selection
pressure toward fitter (lower-RMSE) individuals is the core mechanism that
drives evolution — without it, the GA would just be random search.

Three strategies are implemented:
  - Tournament  : sample k individuals, keep the best one.  The default.
  - Ranking     : assign selection probabilities based on rank order.
  - Roulette    : assign selection probabilities inversely proportional to fitness.

All three accept lower-is-better fitness values (RMSE), so "fitter" always
means a *smaller* number.

Choosing a strategy
-------------------
Tournament is preferred for most runs because:
  - It is computationally cheap.
  - Tournament size k controls selection pressure without recomputing all
    probabilities (larger k → more pressure toward the best individual).
  - It handles fitness landscapes with very similar values well, where
    roulette probabilities would be nearly uniform anyway.
"""

import numpy as np

from .. import population

# Type alias used throughout the module
Individual = list[population.Triangle]


def normalize_selection_type(selection_type: str) -> str:
    """
    Normalises a user-supplied selection strategy name to a canonical form.

    Strips whitespace, lowercases, and maps common aliases so callers can
    write "Tournament", "tournament", "roulette-wheel", etc. interchangeably.

    Args:
        selection_type: Raw strategy name from the config.

    Returns:
        One of ``"tournament"``, ``"ranking"``, or ``"roulette"``.

    Raises:
        ValueError: If the name does not match any known strategy.
    """

    normalized = selection_type.strip().lower().replace("-", "_").replace(" ", "_")

    # Map of accepted spellings → canonical name
    aliases = {
        "tournament":    "tournament",
        "ranking":       "ranking",
        "roulette":      "roulette",
        "roulette_wheel": "roulette",  # common alternative spelling
    }

    if normalized not in aliases:
        raise ValueError(
            "selection_type must be one of: tournament, ranking, roulette."
        )

    return aliases[normalized]


def select_parent(
    population_data: list[Individual],
    fitness_values: list[float],
    selection_type: str,
    tournament_size: int = 3,
) -> Individual:
    """
    Selects and returns one parent individual using the configured strategy.

    This is the single entry point called by the GA each time it needs a
    parent for crossover.  It delegates to the appropriate strategy function
    after normalising and validating the inputs.

    Args:
        population_data: The current population (list of individuals).
        fitness_values:  Corresponding fitness for each individual (lower = better).
        selection_type:  Strategy name (e.g. ``"tournament"``).
        tournament_size: Number of candidates drawn in tournament selection.

    Returns:
        One individual chosen according to the selection strategy.

    Raises:
        ValueError: If population_data is empty or lengths mismatch.
    """

    if not population_data:
        raise ValueError("population_data must not be empty.")
    if len(population_data) != len(fitness_values):
        raise ValueError(
            "population_data and fitness_values must have the same length."
        )

    normalized_type = normalize_selection_type(selection_type)

    if normalized_type == "tournament":
        return tournament_selection(
            population_data,
            fitness_values,
            tournament_size=tournament_size,
        )
    if normalized_type == "ranking":
        return ranking_selection(population_data, fitness_values)

    # Default fallback: roulette wheel
    return roulette_wheel_selection(population_data, fitness_values)


def tournament_selection(
    population_data: list[Individual],
    fitness_values: list[float],
    tournament_size: int = 4,
) -> Individual:
    """
    Selects a parent by running a small tournament.

    Algorithm:
      1. Randomly sample ``tournament_size`` individuals (with replacement
         if the population is smaller than tournament_size).
      2. Return the one with the lowest (best) fitness value.

    Larger tournament sizes apply more selection pressure — the best
    individual wins more often.  Smaller sizes preserve more diversity.

    Args:
        population_data: The current population.
        fitness_values:  Fitness for each individual (lower = better).
        tournament_size: Number of candidates in each tournament.

    Returns:
        The individual with the lowest fitness among the sampled candidates.

    Raises:
        ValueError: If tournament_size is not positive.
    """

    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive.")

    # Use replacement when the population is smaller than the tournament size
    # so we can always draw the requested number of candidates.
    replace = len(population_data) < tournament_size
    candidate_indices = np.random.choice(
        len(population_data), size=tournament_size, replace=replace
    )

    # Pick the candidate with the smallest (best) fitness value
    best_index = min(candidate_indices, key=lambda index: fitness_values[int(index)])

    return population_data[int(best_index)]


def ranking_selection(
    population_data: list[Individual],
    fitness_values: list[float],
) -> Individual:
    """
    Selects a parent using rank-proportional selection probabilities.

    Instead of using raw fitness values (which can be very close together),
    individuals are sorted by fitness and assigned selection probabilities
    based purely on their rank:

        P(rank 1) ∝ N,  P(rank 2) ∝ N-1, ...,  P(rank N) ∝ 1

    The best individual has the highest probability; the worst has the
    lowest.  This approach is more robust than roulette when fitness values
    cluster tightly, because rank differences are always evenly spaced.

    Args:
        population_data: The current population.
        fitness_values:  Fitness for each individual (lower = better).

    Returns:
        One individual sampled proportionally to its rank.
    """

    # argsort returns indices that would sort fitness ascending (best first)
    ranked_indices = np.argsort(fitness_values)

    # Assign linearly decreasing weights: rank-1 individual gets N points,
    # rank-2 gets N-1, ..., rank-N gets 1.
    weights = np.arange(len(ranked_indices), 0, -1, dtype=np.float64)
    probabilities = weights / weights.sum()

    # Sample a position in the ranked list according to the probabilities
    selected_position = int(np.random.choice(len(ranked_indices), p=probabilities))

    return population_data[int(ranked_indices[selected_position])]


def roulette_wheel_selection(
    population_data: list[Individual],
    fitness_values: list[float],
) -> Individual:
    """
    Selects a parent using inverse-fitness roulette-wheel (fitness-proportionate) selection.

    Because we *minimise* fitness, we need to invert the values before
    computing selection probabilities:

        weight_i = 1 / (fitness_i - min_fitness + 1)

    Subtracting the minimum first prevents division by zero when the best
    individual has fitness 0.  Adding 1 keeps weights finite even when
    fitness differences are very small.

    Individuals with lower fitness (better solutions) receive higher weights
    and therefore a larger slice of the roulette wheel.

    Args:
        population_data: The current population.
        fitness_values:  Fitness for each individual (lower = better).

    Returns:
        One individual sampled proportionally to its inverted fitness.
    """

    fitness_array = np.asarray(fitness_values, dtype=np.float64)

    # Shift so the minimum fitness maps to weight 1/(0+1)=1.0
    # This avoids division by zero and keeps the weights well-conditioned.
    shifted_fitness_values = fitness_array - np.min(fitness_array)
    weights = 1.0 / (shifted_fitness_values + 1.0)

    # Normalise to a proper probability distribution summing to 1
    probabilities = weights / weights.sum()

    selected_index = int(np.random.choice(len(population_data), p=probabilities))

    return population_data[selected_index]
