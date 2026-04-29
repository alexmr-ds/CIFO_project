"""Exports the genetic algorithm package public interface."""

from . import cross_over, fitness, legacy, mutate, workflow
from .algorithm import GeneticAlgorithm
from .legacy import LegacyPipelineConfig, LegacyRunResult, run_legacy_pipeline
from .workflow import StageConfig, StageResult, StagedRunResult, run_staged_triangle_optimization

__all__ = [
    "GeneticAlgorithm",
    "fitness",
    "cross_over",
    "mutate",
    "legacy",
    "workflow",
    "LegacyPipelineConfig",
    "LegacyRunResult",
    "run_legacy_pipeline",
    "StageConfig",
    "StageResult",
    "StagedRunResult",
    "run_staged_triangle_optimization",
]
