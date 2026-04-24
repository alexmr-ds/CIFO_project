# GeneticAlgorithm Reference

`src.ga.GeneticAlgorithm` evolves triangle-based images to minimize a target-image fitness function. The project is currently operated from notebooks, so the recommended workflow is to use sequential evaluation by default and thread evaluation only when experimenting with notebook-compatible parallelism.

Lower fitness is better because the default fitness function is RMSE.

Internally, `src/ga/algorithm.py` keeps the GA lifecycle focused on population state, selection, operators, best-fitness tracking, and generation flow. Evaluation backend mechanics live in `src/ga/evaluation.py`, and run-log formatting lives in `src/ga/logs.py`.

## Notebook Recommended Usage

```python
from src import fitness, load_image
from src.ga import GeneticAlgorithm

image_path = project_root / "images/girl_pearl_earing.png"
target_image = load_image.load_target_image(image_path)

ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=50,
    elitism=2,
    logs=True,
)

best_fitness, history = ga.run()
best_individual = ga.best_individual
run_logs = ga.run_logs
```

- `best_fitness`: lowest fitness found during the run.
- `history`: global best fitness after each generation.
- `ga.best_individual`: best triangle configuration found globally.
- `ga.run_logs`: summary dictionary populated only when `logs=True`.

## Constructor

```python
ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=50,
    crossover_rate=None,
    mutation_rate=None,
    elitism=0,
    selection_type="tournament",
    logs=False,
    crossover_function=None,
    mutation_function=None,
    evaluation_backend="sequential",
    n_jobs=None,
    chunksize=None,
)
```

## Parameters

- `target: np.ndarray`: Target RGB image array with shape `(H, W, 3)`.
- `fitness_function: Callable[[np.ndarray, np.ndarray], float]`: Function that evaluates `(target, generated)` arrays. Lower values are better.
- `population_size: int`: Number of individuals in the population. Must be greater than `0`.
- `generations: int`: Number of generations to run. Must be greater than `0`.
- `crossover_rate: float | None = None`: Required only when `crossover_function` is provided.
- `mutation_rate: float | None = None`: Required only when `mutation_function` is provided.
- `elitism: int = 0`: Number of top individuals copied unchanged into the next generation. Must be in `[0, population_size]`.
- `selection_type: str = "tournament"`: Parent selection strategy. Valid values are `"tournament"`, `"ranking"`, and `"roulette"`.
- `logs: bool = False`: When `True`, stores run metadata in `ga.run_logs` after `run()`.
- `crossover_function: Callable | None = None`: Optional callback with signature `fn(parent1, parent2, crossover_rate) -> individual`.
- `mutation_function: Callable | None = None`: Optional callback with signature `fn(individual, mutation_rate, image_width, image_height) -> individual`.
- `evaluation_backend: Literal["sequential", "thread", "process"] = "sequential"`: Fitness evaluation backend.
- `n_jobs: int | None = None`: Worker count for thread/process evaluation. `None` uses the executor default.
- `chunksize: int | None = None`: Process-pool map chunksize. Ignored by sequential/thread evaluation.

## Conditional Rate Rules

- If `crossover_function is not None`, `crossover_rate` must be provided.
- If `mutation_function is not None`, `mutation_rate` must be provided.
- If a rate is provided, it must be in `[0.0, 1.0]`.
- If a function is `None` and its rate is omitted, the internal rate resolves to `0.0`.

## Run Behavior

```python
best_fitness, history = ga.run()
```

- `run()` initializes a fresh population every time it is called.
- `best_fitness` is the global best fitness found across all generations.
- `history` stores the global best fitness after each generation.
- `ga.best_individual` stores the best triangle configuration found globally.
- `ga.best_fitness` stores the same value returned as `best_fitness`.
- `ga.history` stores the same values returned as `history`.
- `ga.run_logs` remains `{}` unless `logs=True`.

## Evaluate Behavior

```python
ga.initialize()
fitness_values = ga.evaluate()
```

- `evaluate()` computes fitness for the current population.
- Returned fitness values preserve the same order as `ga.population`.
- `evaluate()` updates `ga.best_fitness` and `ga.best_individual`.
- Calling `evaluate()` before `initialize()` raises a `ValueError`.

## Logging (`logs=True`)

When `logs=True`, `ga.run_logs` is populated after `run()` finishes:

```python
{
    "generations": [
        {
            "generation": 0,
            "generation_best_fitness": 123.4,
            "generation_mean_fitness": 180.2,
            "global_best_fitness": 123.4,
            "evaluation_backend": "sequential",
            "n_jobs": None,
            "chunksize": None,
            "evaluation_duration_seconds": 0.03,
        }
    ],
    "best_fitness": 123.4,
    "best_individual_configuration": [
        {
            "x1": 1,
            "y1": 2,
            "x2": 3,
            "y2": 4,
            "x3": 5,
            "y3": 6,
            "r": 100,
            "g": 120,
            "b": 140,
            "a": 180,
        }
    ],
}
```

The `best_individual_configuration` value is JSON-friendly and can be stored later if a persistence layer is added.

## Selection Strategies

`selection_type` is dispatched in `src/ga/selection.py`:

- `"tournament"`
- `"ranking"`
- `"roulette"`

All strategies assume lower fitness is better.

## Evaluation Backends

### Sequential

Recommended notebook default. Use this while developing, debugging, and validating new notebook cells.

```python
ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=50,
    elitism=2,
    evaluation_backend="sequential",
)
```

### Thread

Notebook-compatible parallel option. It supports notebook-local functions, but it may not speed up this workload because rendering is PIL-based.

```python
ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=50,
    elitism=2,
    evaluation_backend="thread",
    n_jobs=4,
)
```

### Process

Advanced option for larger runs. It is not the recommended notebook default because multiprocessing requires picklable/importable functions and can be fragile inside notebook kernels.

```python
ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=500,
    generations=100,
    elitism=5,
    evaluation_backend="process",
    n_jobs=4,
    chunksize=8,
)
```

Process backend caveats:

- `fitness_function` must be picklable.
- Module-level functions such as `src.fitness.compute_rmse` are safest.
- Avoid lambdas and notebook-local closures.
- Prefer thread or sequential evaluation for normal notebook-only work.

## Operator Injection

Current `src/ga/cross_over.py` and `src/ga/mutate.py` are scaffolded/commented examples. Define notebook-local callables or implement those modules before importing operators from them.

```python
def my_crossover(parent1, parent2, crossover_rate):
    return parent1


def my_mutation(individual, mutation_rate, image_width, image_height):
    return individual


ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=50,
    elitism=2,
    crossover_function=my_crossover,
    crossover_rate=0.8,
    mutation_function=my_mutation,
    mutation_rate=0.1,
)
```

## Common Validation Errors

- `target must have shape (H, W, 3).`
- `population_size must be greater than zero.`
- `generations must be greater than zero.`
- `elitism must be between 0 and population_size.`
- `selection_type must be one of: tournament, ranking, roulette.`
- `evaluation_backend must be 'sequential', 'thread', or 'process'.`
- `n_jobs must be None or a positive integer.`
- `chunksize must be None or a positive integer.`
- `process evaluation requires a picklable fitness_function.`
- `crossover_rate must be provided when its function is set.`
- `mutation_rate must be provided when its function is set.`
- `<rate_name> must be between 0 and 1.`
