"""Exports the genetic algorithm package public interface."""

from . import cross_over, fitness, mutate, parallel, results, workflow
from .algorithm import GeneticAlgorithm
from .results import load_all_runs, runs_dataframe, save_run
from .workflow import StageConfig, StageResult, StagedRunResult, run_staged_triangle_optimization
from .parallel import GAConfig, TrialSummary, run_single_ga, run_trials, run_grid_search

__all__ = [
    "GeneticAlgorithm",
    "fitness",
    "cross_over",
    "mutate",
    "results",
    "save_run",
    "load_all_runs",
    "runs_dataframe",
    "workflow",
    "StageConfig",
    "StageResult",
    "StagedRunResult",
    "run_staged_triangle_optimization",
    "parallel",
    "GAConfig",
    "TrialSummary",
    "run_single_ga",
    "run_trials",
    "run_grid_search",
]
