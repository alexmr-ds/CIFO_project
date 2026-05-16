"""Exports the genetic algorithm package public interface."""

from . import cross_over, diversity, fitness, mutate, parallel
from .algorithm import GeneticAlgorithm
from .diversity import FitnessSharingGA, RestrictedMatingGA, FitnessSharingRestrictedMatingGA
from .parallel import GAConfig, TrialSummary, run_single_ga, run_trials, run_grid_search, run_variants_batch
from .plotting import (
    plot_grid_search_results,
    plot_diversity_comparison,
    plot_method_comparison,
    print_grid_search_summary,
    print_method_comparison_summary,
    print_diversity_summary,
    plot_single_diversity,
    print_single_diversity_summary,
    plot_cumulative_convergence,
)

__all__ = [
    "GeneticAlgorithm",
    "FitnessSharingGA",
    "RestrictedMatingGA",
    "FitnessSharingRestrictedMatingGA",
    "diversity",
    "fitness",
    "cross_over",
    "mutate",
    "parallel",
    "GAConfig",
    "TrialSummary",
    "run_single_ga",
    "run_trials",
    "run_grid_search",
    "run_variants_batch",
    "plot_grid_search_results",
    "plot_diversity_comparison",
    "plot_method_comparison",
    "print_grid_search_summary",
    "print_method_comparison_summary",
    "print_diversity_summary",
    "plot_single_diversity",
    "print_single_diversity_summary",
    "plot_cumulative_convergence",
]
