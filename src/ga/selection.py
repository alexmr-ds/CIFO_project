"""Implements parent selection strategies for the genetic algorithm."""

import numpy as np

from .. import population

Individual = list[population.Triangle]


def normalize_selection_type(selection_type: str) -> str:
    """Normalizes the configured selection strategy name."""

    normalized = selection_type.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "tournament": "tournament",
        "ranking": "ranking",
        "roulette": "roulette",
        "roulette_wheel": "roulette",
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
    """Dispatches parent selection according to the configured strategy."""

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

    return roulette_wheel_selection(population_data, fitness_values)


def tournament_selection(
    population_data: list[Individual],
    fitness_values: list[float],
    tournament_size: int = 4,
) -> Individual:
    """Selects a parent using tournament selection."""

    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive.")

    replace = len(population_data) < tournament_size
    candidate_indices = np.random.choice(
        len(population_data), size=tournament_size, replace=replace
    )
    best_index = min(candidate_indices, key=lambda index: fitness_values[int(index)])

    return population_data[int(best_index)]


def ranking_selection(
    population_data: list[Individual], fitness_values: list[float]
) -> Individual:
    """Selects a parent using rank-based probabilities."""

    ranked_indices = np.argsort(fitness_values)
    weights = np.arange(len(ranked_indices), 0, -1, dtype=np.float64)
    probabilities = weights / weights.sum()
    selected_position = int(np.random.choice(len(ranked_indices), p=probabilities))

    return population_data[int(ranked_indices[selected_position])]


def roulette_wheel_selection(
    population_data: list[Individual], fitness_values: list[float]
) -> Individual:
    """Selects a parent using inverse-fitness roulette-wheel probabilities."""

    fitness_array = np.asarray(fitness_values, dtype=np.float64)
    shifted_fitness_values = fitness_array - np.min(fitness_array)
    weights = 1.0 / (shifted_fitness_values + 1.0)
    probabilities = weights / weights.sum()
    selected_index = int(np.random.choice(len(population_data), p=probabilities))

    return population_data[selected_index]
