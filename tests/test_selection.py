"""Tests parent selection dispatch and keyword compatibility."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from src.ga import selection


class TournamentSelectionTests(TestCase):
    """Covers tournament selection keyword usage."""

    def setUp(self) -> None:
        """Creates simple candidate identities and fitness values."""

        self.population_data = [["first"], ["second"], ["third"]]
        self.fitness_values = [3.0, 1.0, 2.0]

    def test_tournament_selection_accepts_tournament_size_keyword(self) -> None:
        """Direct tournament calls accept the renamed keyword."""

        with patch(
            "src.ga.selection.np.random.choice",
            return_value=np.array([0, 1]),
        ) as choice:
            selected = selection.tournament_selection(
                self.population_data,
                self.fitness_values,
                tournament_size=2,
            )

        self.assertEqual(["second"], selected)
        self.assertEqual(2, choice.call_args.kwargs["size"])

    def test_select_parent_dispatches_tournament_size_keyword(self) -> None:
        """The generic dispatcher forwards tournament_size without using k."""

        with patch(
            "src.ga.selection.np.random.choice",
            return_value=np.array([0, 2]),
        ) as choice:
            selected = selection.select_parent(
                self.population_data,
                self.fitness_values,
                selection_type="tournament",
                tournament_size=2,
            )

        self.assertEqual(["third"], selected)
        self.assertEqual(2, choice.call_args.kwargs["size"])
