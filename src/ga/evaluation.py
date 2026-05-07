"""
Fitness evaluation backends for the genetic algorithm.

Evaluating the fitness of every individual in the population is the most
expensive step of each GA generation: each individual must be rendered to
a full pixel image and then compared to the target.  With population_size=100
and a 300×400 canvas this happens 100 times per generation.

This module provides three backends to distribute that work:

  sequential
      Evaluates individuals one-by-one on the main process.
      Simplest, no overhead, used inside parallel.py workers.

  thread
      Uses a ThreadPoolExecutor.  Effective when the bottleneck is I/O or
      when the C-extension code (NumPy, Pillow) releases the GIL, allowing
      true parallel execution even in CPython.

  process
      Uses a ProcessPoolExecutor.  True CPU parallelism with separate memory
      spaces.  Each worker is initialised *once* with the target array and
      fitness function (see _initialize_process_worker) so those large objects
      are not re-sent for every individual, only when the pool starts.

Process worker pattern
----------------------
Sending the target image and fitness function with every job would be
expensive (the target is a ~360 KB float32 array).  Instead, the pool
initialiser runs once per worker process and stores those objects in module-
level globals.  Each job then only receives the lightweight triangle list.
"""

import concurrent.futures
import pickle
from collections.abc import Callable
from itertools import repeat
from typing import Literal

import numpy as np

from .. import population, rendering

# Type aliases used across the GA package
Individual = list[population.Triangle]
FitnessFunction = Callable[[np.ndarray, np.ndarray], float]
EvaluationBackend = Literal["sequential", "thread", "process"]
Executor = concurrent.futures.Executor

# ---------------------------------------------------------------------------
# Process-worker state
# These globals are written once per worker process by _initialize_process_worker
# and then read by every call to _compute_individual_fitness_in_process.
# They are None in the main process — accessing them there is a bug.
# ---------------------------------------------------------------------------
_PROCESS_TARGET: np.ndarray | None = None
_PROCESS_FITNESS_FUNCTION: FitnessFunction | None = None
_PROCESS_IMAGE_WIDTH: int | None = None
_PROCESS_IMAGE_HEIGHT: int | None = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def normalize_evaluation_backend(evaluation_backend: str) -> EvaluationBackend:
    """
    Normalises and validates an evaluation backend name.

    Strips whitespace and lowercases the input so callers can write
    "Sequential", "THREAD", etc. without causing errors.

    Args:
        evaluation_backend: Raw backend name from user config.

    Returns:
        One of ``"sequential"``, ``"thread"``, or ``"process"``.

    Raises:
        ValueError: If the name does not match any known backend.
    """

    normalized_backend = evaluation_backend.strip().lower()

    if normalized_backend == "sequential":
        return "sequential"
    if normalized_backend == "thread":
        return "thread"
    if normalized_backend == "process":
        return "process"

    raise ValueError(
        "evaluation_backend must be 'sequential', 'thread', or 'process'."
    )


def validate_optional_positive_int(value: int | None, name: str) -> None:
    """
    Validates that an optional worker-count parameter is a positive integer.

    ``None`` is always accepted (means "use the backend default").
    Booleans are rejected even though they are technically integers in Python,
    because ``True`` would silently set n_jobs=1.

    Args:
        value: The value to validate, or None to skip validation.
        name:  Parameter name used in the error message.

    Raises:
        ValueError: If value is not None and is not a positive integer.
    """

    if value is None:
        return

    # bool is a subclass of int in Python — explicitly reject it
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be None or a positive integer.")


def validate_process_fitness_function(fitness_function: FitnessFunction) -> None:
    """
    Verifies that a fitness function can be pickled for process-pool workers.

    Python's multiprocessing uses pickle to send work to child processes.
    Lambda functions and closures cannot be pickled, so this check fails
    fast at configuration time rather than crashing mid-run.

    Args:
        fitness_function: The callable to test.

    Raises:
        ValueError: If the function cannot be serialised with pickle.
    """

    try:
        pickle.dumps(fitness_function)
    except Exception as exc:
        raise ValueError(
            "process evaluation requires a picklable fitness_function. "
            "Use a module-level function such as src.ga.fitness.compute_rmse."
        ) from exc


# ---------------------------------------------------------------------------
# Single-individual evaluation
# ---------------------------------------------------------------------------

def compute_individual_fitness(
    individual: Individual,
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
) -> float:
    """
    Renders one individual and returns its fitness score.

    This is the innermost evaluation step:
      1. Render the list of triangles to a pixel array.
      2. Pass the rendered array and the target to the fitness function.

    Args:
        individual:       List of triangles composing the candidate image.
        target:           The reference image array (H, W, 3).
        fitness_function: Callable that computes a scalar from two arrays.
        image_width:      Canvas width for rendering.
        image_height:     Canvas height for rendering.

    Returns:
        Scalar fitness value (lower is better for RMSE).
    """

    generated = rendering.image_to_array(
        individual,
        image_width=image_width,
        image_height=image_height,
    )

    return float(fitness_function(target, generated))


# ---------------------------------------------------------------------------
# Population-level evaluation
# ---------------------------------------------------------------------------

def compute_population_fitness_sequential(
    population_data: list[Individual],
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
) -> list[float]:
    """
    Evaluates all individuals sequentially on the calling process.

    This is the simplest backend: individuals are evaluated one by one in
    the main process.  No inter-process communication overhead, no pickling.
    Used by default inside parallel.py workers to avoid nested process pools.

    Args:
        population_data:  List of individuals to evaluate.
        target:           Reference image array.
        fitness_function: Fitness callable.
        image_width:      Canvas width.
        image_height:     Canvas height.

    Returns:
        List of fitness floats in the same order as population_data.
    """

    return [
        compute_individual_fitness(
            individual,
            target,
            fitness_function,
            image_width,
            image_height,
        )
        for individual in population_data
    ]


def compute_population_fitness_with_executor(
    executor: Executor,
    evaluation_backend: EvaluationBackend,
    population_data: list[Individual],
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
    chunksize: int | None = None,
) -> list[float]:
    """
    Evaluates all individuals using an already-created executor.

    Thread and process backends differ in what arguments they pass to workers:
      - Thread workers receive all arguments per call (target, function, etc.)
        because threads share memory and there is no serialisation cost.
      - Process workers only receive the individual because the target and
        fitness function were already injected via the pool initialiser,
        avoiding repeated serialisation of large arrays.

    Args:
        executor:          An active ThreadPoolExecutor or ProcessPoolExecutor.
        evaluation_backend: ``"thread"`` or ``"process"`` — selects call style.
        population_data:   List of individuals to evaluate.
        target:            Reference image array (used for thread backend).
        fitness_function:  Fitness callable (used for thread backend).
        image_width:       Canvas width (used for thread backend).
        image_height:      Canvas height (used for thread backend).
        chunksize:         Optional process-pool batch size for efficiency.

    Returns:
        List of fitness floats in the same order as population_data.
    """

    if evaluation_backend == "thread":
        # Threads share memory, so we can pass all arguments directly.
        # repeat() creates infinite iterators that broadcast the shared args.
        return list(
            executor.map(
                compute_individual_fitness,
                population_data,
                repeat(target),
                repeat(fitness_function),
                repeat(image_width),
                repeat(image_height),
            )
        )

    # Process backend: the target/function are already in worker globals,
    # so each job only needs to receive the individual itself.
    if chunksize is None:
        return list(
            executor.map(_compute_individual_fitness_in_process, population_data)
        )

    # With chunksize, multiple individuals are sent to each worker at once,
    # reducing inter-process communication overhead for large populations.
    return list(
        executor.map(
            _compute_individual_fitness_in_process,
            population_data,
            chunksize=chunksize,
        )
    )


def create_evaluation_executor(
    evaluation_backend: EvaluationBackend,
    n_jobs: int | None,
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
) -> Executor:
    """
    Creates and returns a configured executor for parallel evaluation.

    For the process backend, the executor is initialised with the target
    array, fitness function, and image dimensions so each worker stores them
    once rather than receiving them with every job submission.

    The returned executor must be used as a context manager or explicitly
    shut down by the caller.

    Args:
        evaluation_backend: ``"thread"`` or ``"process"``.
        n_jobs:             Number of workers (None = backend default).
        target:             Target image array sent to process workers.
        fitness_function:   Fitness function sent to process workers.
        image_width:        Canvas width sent to process workers.
        image_height:       Canvas height sent to process workers.

    Returns:
        An active Executor ready to receive map() calls.
    """

    if evaluation_backend == "thread":
        # ThreadPoolExecutor is simple — no initialiser needed because
        # threads share the main process's memory space.
        return concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs)

    # ProcessPoolExecutor: use the initialiser to pre-load shared data into
    # each worker process so it only travels across the process boundary once.
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_initialize_process_worker,
        initargs=(
            target,
            fitness_function,
            image_width,
            image_height,
        ),
    )


# ---------------------------------------------------------------------------
# Process-worker internals
# ---------------------------------------------------------------------------

def _initialize_process_worker(
    target: np.ndarray,
    fitness_function: FitnessFunction,
    image_width: int,
    image_height: int,
) -> None:
    """
    Stores shared evaluation state in the worker process's global namespace.

    This function runs *once* when each worker process starts up (via the
    ProcessPoolExecutor ``initializer`` parameter).  By storing the target
    array and fitness function as globals, subsequent calls to
    ``_compute_individual_fitness_in_process`` can access them without
    any additional data transfer between the main process and the worker.

    Args:
        target:           Target image array to store in this worker.
        fitness_function: Fitness callable to store in this worker.
        image_width:      Canvas width to store in this worker.
        image_height:     Canvas height to store in this worker.
    """

    global _PROCESS_TARGET
    global _PROCESS_FITNESS_FUNCTION
    global _PROCESS_IMAGE_WIDTH
    global _PROCESS_IMAGE_HEIGHT

    _PROCESS_TARGET = target
    _PROCESS_FITNESS_FUNCTION = fitness_function
    _PROCESS_IMAGE_WIDTH = image_width
    _PROCESS_IMAGE_HEIGHT = image_height


def _compute_individual_fitness_in_process(individual: Individual) -> float:
    """
    Evaluates one individual using the process-local evaluation state.

    This function runs inside a worker process.  It reads the shared target
    and fitness function from the module-level globals set by the initialiser,
    then delegates to ``compute_individual_fitness``.

    Args:
        individual: The triangle list to evaluate (the only data sent per job).

    Returns:
        Scalar fitness value for this individual.

    Raises:
        RuntimeError: If the worker was not properly initialised.
    """

    # Guard against accidental calls from the main process where globals are None
    if (
        _PROCESS_TARGET is None
        or _PROCESS_FITNESS_FUNCTION is None
        or _PROCESS_IMAGE_WIDTH is None
        or _PROCESS_IMAGE_HEIGHT is None
    ):
        raise RuntimeError("Process worker evaluation state was not initialized.")

    return compute_individual_fitness(
        individual,
        _PROCESS_TARGET,
        _PROCESS_FITNESS_FUNCTION,
        _PROCESS_IMAGE_WIDTH,
        _PROCESS_IMAGE_HEIGHT,
    )
