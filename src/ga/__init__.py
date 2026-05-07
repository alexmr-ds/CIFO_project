"""Exports the genetic algorithm package public interface."""

from . import cross_over, fitness, greedy, hybrid, legacy, mutate, parallel, results, workflow
from .algorithm import GeneticAlgorithm
from .greedy import create_greedy_seeded_population
from .hybrid import HybridPSOGA, HybridPSOGAResult
from .results import load_all_runs, runs_dataframe, save_run
from .legacy import LegacyPipelineConfig, LegacyRunResult, run_legacy_pipeline
from .workflow import StageConfig, StageResult, StagedRunResult, run_staged_triangle_optimization
from .parallel import GAConfig, run_single_ga

__all__ = [
    "GeneticAlgorithm",
    "fitness",
    "cross_over",
    "mutate",
    "greedy",
    "create_greedy_seeded_population",
    "results",
    "save_run",
    "load_all_runs",
    "runs_dataframe",
    "hybrid",
    "HybridPSOGA",
    "HybridPSOGAResult",
    "legacy",
    "workflow",
    "LegacyPipelineConfig",
    "LegacyRunResult",
    "run_legacy_pipeline",
    "StageConfig",
    "StageResult",
    "StagedRunResult",
    "run_staged_triangle_optimization",
    "parallel",
    "GAConfig",
    "run_single_ga",
]
