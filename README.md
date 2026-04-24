# CIFO Project

Notebook-first prototype for approximating a target image with semi-transparent triangles and a genetic algorithm. The current codebase supports the full core loop: load a target image, generate triangle-based individuals, render candidates, evaluate fitness with RMSE, and evolve a population with configurable selection, elitism, optional genetic operators, and opt-in parallel fitness evaluation.

## Project Status

- Core prototype pipeline is working across image loading, population generation, rendering, fitness evaluation, and GA orchestration.
- The intended workflow is notebook-based, centered on [notebooks/Exploration.ipynb](notebooks/Exploration.ipynb).
- Active modules are `src.load_image`, `src.population`, `src.rendering`, `src.fitness`, and `src.ga.GeneticAlgorithm`.
- `GeneticAlgorithm.run()` returns `(best_fitness, history)`. The best triangle configuration is available separately as `ga.best_individual`.
- `GeneticAlgorithm` supports optional crossover and mutation callbacks. `crossover_rate` and `mutation_rate` are only required when the matching callback is provided.
- Fitness evaluation is sequential by default. `thread` is the notebook-compatible parallel backend; `process` exists but is less reliable inside notebooks.
- GA backend evaluation mechanics live in `src/ga/evaluation.py`, and run-log formatting lives in `src/ga/logs.py`.
- See [GeneticAlgorithm.md](GeneticAlgorithm.md) for full class configuration and behavior details.
- `src/ga/cross_over.py` and `src/ga/mutate.py` currently contain scaffolded/commented operator examples, not active importable operators.
- Validation is currently based on compile/import/smoke checks. There is not yet a formal automated test suite.

## Current Workflow

1. Load the target image with `src.load_image.load_target_image(...)`.
2. Create random individuals or populations with `src.population`.
3. Render an individual for visualization with `src.rendering.render_individual(...)`.
4. Convert the generated image to an array with `src.rendering.image_to_array(...)`.
5. Evaluate the generated array against the target with `src.fitness.compute_rmse(...)` or another compatible fitness function.
6. Run the genetic algorithm with `src.ga.GeneticAlgorithm`.
7. Read the run result from `best_fitness`, `history`, `ga.best_individual`, and optionally `ga.run_logs`.
8. Optionally inject compatible crossover and mutation callables.
9. Optionally use `evaluation_backend="thread"` for notebook-compatible parallel evaluation experiments.

## Setup

This project uses `uv` for environment and dependency management.

```bash
uv sync
uv run python main.py
uv run python -m compileall src main.py
```

## Notebook Workflow

- Open [notebooks/Exploration.ipynb](notebooks/Exploration.ipynb) with the project interpreter after running `uv sync`.
- The first notebook cell adds the project root and `src/ga` to `sys.path` for quick experimentation.
- Prefer `evaluation_backend="sequential"` while developing or debugging.
- Use `evaluation_backend="thread"` if you want notebook-compatible parallel evaluation.
- Avoid `evaluation_backend="process"` in normal notebook work unless the fitness function is importable/picklable and the notebook kernel handles multiprocessing cleanly.

### Minimal Notebook Run

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

`best_fitness` is the lowest fitness found during the run. `history` contains the global best fitness after each generation. `best_individual` is the best triangle configuration found globally.

### Optional Genetic Operators

Optional genetic operators can be injected as notebook-local callables. If an operator is provided, its rate must also be provided.

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

Expected operator signatures:

```python
def my_crossover(parent1, parent2, crossover_rate):
    ...


def my_mutation(individual, mutation_rate, image_width, image_height):
    ...
```

## Parallel Evaluation

Sequential evaluation remains the recommended notebook default:

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

Thread evaluation is the notebook-friendly parallel option. It supports notebook-local functions, but it may not speed up the current PIL rendering and RMSE workload.

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

Process evaluation is available, but it is not the primary notebook workflow. It requires picklable/importable functions, so lambdas and notebook-local closures can fail.

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

Parallel evaluation details:

- `evaluation_backend="sequential"`: safest notebook default and baseline behavior.
- `evaluation_backend="thread"`: notebook-compatible parallel option, but speedup is workload-dependent.
- `evaluation_backend="process"`: advanced option; better suited to future script-based experiment runners than notebook-only use.
- `n_jobs=None`: uses the executor default worker count.
- `chunksize`: controls process-pool batching and is ignored by sequential/thread evaluation.
- During `run()`, non-sequential backends reuse the same executor across generations.
- When `logs=True`, `run_logs` is a summary dictionary with generation fitness logs, `best_fitness`, and a JSON-friendly `best_individual_configuration`.

## Current Limitations

- The project is still prototype-focused and notebook-first.
- There is no formal automated test suite beyond compile/import/smoke checks.
- GA logs are stored in memory only and there is no persistence or visualization layer yet.
- Process evaluation is not optimized for the notebook-only workflow.
- Rendering is still PIL-based and may become the next performance bottleneck after evaluation parallelism.
- Rendering and image constants are still tightly coupled to the current triangle representation.

## Immediate Next Steps

- Add formal tests for GA evaluation backends, fitness ordering, logging, and best-individual tracking.
- Implement production crossover and mutation operators in `src/ga/cross_over.py` and `src/ga/mutate.py`.
- Add notebook cells for comparing sequential and thread evaluation timings.
- Add output saving for best rendered images, run metadata, and generation summaries.
- Consider elite-fitness caching to avoid recomputing unchanged elite individuals.
- Consider render optimization before relying on process-based parallelism in notebooks.

## File Tree

- `README.md`: Project overview, notebook workflow, setup instructions, progress status, limitations, and maintained file tree.
- `GeneticAlgorithm.md`: Detailed notebook-first reference for `src.ga.GeneticAlgorithm`.
- `AGENTS.md`: Repository instructions for GA terminology, API expectations, and project conventions.
- `main.py`: Minimal CLI entrypoint used for simple environment smoke checks.
- `pyproject.toml`: Project metadata and dependency declarations managed with `uv`.
- `uv.lock`: Locked dependency versions for reproducible installs.
- `.gitignore`: Ignore rules for Python, notebook, editor, cache, and generated artifacts.
- `.python-version`: Local Python version pin for the project environment.
- `images/girl_pearl_earing.png`: Sample target image used in notebook and smoke-test workflows.
- `notebooks/Exploration.ipynb`: Notebook for exploring rendering, fitness evaluation, package imports, and GA setup.
- `src/__init__.py`: Exposes the main package submodules.
- `src/load_image.py`: Loads and resizes target images into NumPy arrays.
- `src/population.py`: Defines triangles, image constants, and random factories for individuals and populations.
- `src/rendering.py`: Renders individuals and converts rendered output to arrays.
- `src/fitness.py`: Computes RMSE between target and generated images.
- `src/ga/__init__.py`: Exports the public `GeneticAlgorithm` class.
- `src/ga/algorithm.py`: Orchestrates GA state, optional operators, global-best history, and generation flow.
- `src/ga/cross_over.py`: Contains scaffolded/commented crossover operator examples for triangle individuals.
- `src/ga/evaluation.py`: Computes ordered fitness values with sequential, thread, or process evaluation backends.
- `src/ga/logs.py`: Builds per-generation and run-level GA log dictionaries.
- `src/ga/mutate.py`: Contains scaffolded/commented mutation operator examples for triangle individuals.
- `src/ga/selection.py`: Implements tournament, ranking, and roulette-wheel parent selection strategies.
