# CIFO Project

Notebook-first prototype for approximating a target image with semi-transparent triangles and a genetic algorithm. The current codebase supports the full core loop: load a target image, generate triangle-based individuals, render candidates, evaluate fitness with RMSE or another compatible fitness function, and evolve a population with configurable selection, elitism, optional genetic operators, and opt-in parallel fitness evaluation.

## Project Status

- Core prototype pipeline is working across image loading, population generation, rendering, fitness evaluation, and GA orchestration.
- The intended workflow is notebook-based, centered on [notebooks/Exploration.ipynb](notebooks/Exploration.ipynb).
- Active modules are `src.load_image`, `src.population`, `src.rendering`, `src.ga.fitness`, `src.ga.GeneticAlgorithm`, and `src.ga.workflow`.
- `src.ga.fitness` includes RMSE fitness helpers.
- `src.ga.fitness` also provides RMSE+structure fitness helpers for richer guidance.
- `GeneticAlgorithm.run()` returns `(best_fitness, history)`. The best triangle configuration is available separately as `ga.best_individual`.
- `GeneticAlgorithm` can stream per-generation progress to notebooks and reports mutation telemetry in generation logs.
- `GeneticAlgorithm` supports optional crossover and mutation callbacks. Crossover callbacks may return one child or multiple children. `crossover_rate` and `mutation_rate` are only required when the matching callback is provided.
- `GeneticAlgorithm` supports `triangle_alpha_range` to control triangle transparency during initialization and mutation.
- `GeneticAlgorithm` supports `seeded=True` to initialize triangle RGB values from random target-image pixels.
- Fitness evaluation is sequential by default. `thread` is the notebook-compatible parallel backend; `process` exists but is less reliable inside notebooks.
- GA backend evaluation mechanics live in `src/ga/evaluation.py`, and run-log formatting lives in `src/ga/logs.py`.
- See [GeneticAlgorithm.md](GeneticAlgorithm.md) for full class configuration and behavior details.
- `src/ga/cross_over.py` and `src/ga/mutate.py` contain optional operator examples for triangle individuals.
- `src/ga/workflow.py` provides staged triangle-count orchestration for notebook experiments.
- Validation includes focused unit tests plus compile/import/smoke checks.

## Current Workflow

1. Load the target image with `src.load_image.load_target_image(...)`.
2. Create random or target-biased individuals and populations with configurable alpha ranges through `src.population`.
3. Render an individual for visualization with `src.rendering.render_individual(...)`.
4. Convert the generated image to an array with `src.rendering.image_to_array(...)`.
5. Evaluate the generated array against the target with `src.ga.fitness.compute_rmse(...)` or another compatible fitness function.
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
- The first notebook cell adds the project root to `sys.path` so package imports resolve from the repository root.
- Prefer `evaluation_backend="sequential"` while developing or debugging.
- Use `evaluation_backend="thread"` if you want notebook-compatible parallel evaluation.
- Avoid `evaluation_backend="process"` in normal notebook work unless the fitness function is importable/picklable and the notebook kernel handles multiprocessing cleanly.

### Minimal Notebook Run

```python
from src import load_image
from src.ga import GeneticAlgorithm, fitness

image_path = project_root / "images/girl_pearl_earing.png"
target_image = load_image.load_target_image(image_path)

ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=50,
    elitism=2,
    triangle_alpha_range=(5, 255),
    seeded=True,
    logs=True,
)

best_fitness, history = ga.run()
best_individual = ga.best_individual
run_logs = ga.run_logs
```

`best_fitness` is the lowest fitness found during the run. `history` contains the global best fitness after each generation. `best_individual` is the best triangle configuration found globally.

### Seeded RGB Initialization

Set `seeded=True` to initialize generation-0 triangle colors from exact RGB triplets sampled from random target pixels:

```python
ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=50,
    seeded=True,
)
```

Seeded initialization keeps triangle vertices random, samples alpha uniformly from `triangle_alpha_range`, and applies only when generation 0 is created or backfilled from a short `initial_population`. `random_immigrants` remain fully random in later generations.

### Live Progress Tracking in Notebooks

Use `progress=True` for built-in console updates, or attach `progress_callback` to collect generation telemetry:

```python
from IPython.display import clear_output
from src.ga import GeneticAlgorithm, fitness, mutate

progress_rows = []

def on_progress(generation_log):
    progress_rows.append(generation_log)
    if generation_log["generation"] % 10 == 0:
        clear_output(wait=True)
        print(
            f"gen={generation_log['generation'] + 1} "
            f"best={generation_log['global_best_fitness']:.6f} "
            f"mutated={generation_log['mutated_offspring']} "
            f"triangles_changed={generation_log['mutated_triangles']}"
        )

ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=120,
    generations=80,
    mutation_function=mutate.volatile_triangle_mutation,
    mutation_rate=0.1,
    logs=True,
    progress=True,
    progress_interval=5,
    progress_callback=on_progress,
)
```

### Triangle Transparency

Triangle transparency is controlled by an inclusive alpha range:

```python
ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=50,
    triangle_alpha_range=(40, 180),
)
```

The default is `(5, 255)`, which avoids fully invisible triangles by default. Initial alpha values are sampled uniformly from the configured range. Use `(255, 255)` for fully opaque triangles or `(0, 255)` to allow the full alpha range. Mutation operators receive the same range so mutated alpha values stay within the configured bounds.

### Richer Fitness (RMSE + Structure)

You can blend RMSE with an edge-structure term for better shape preservation:

```python
from src.ga import fitness

hybrid_fitness = fitness.make_rmse_structure_fitness(
    rmse_weight=1.0,
    structure_weight=0.35,
)

ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=hybrid_fitness,
    population_size=100,
    generations=50,
)
```

### Adaptive Mutation and Diversity Control

`GeneticAlgorithm` now supports adaptive mutation schedules and random-immigrant diversity injection:

```python
from src.ga import GeneticAlgorithm, mutate

ga = GeneticAlgorithm(
    target=target_image,
    fitness_function=fitness.compute_rmse,
    population_size=120,
    generations=80,
    mutation_function=mutate.volatile_triangle_mutation,
    mutation_rate=0.12,
    adaptive_mutation=True,
    mutation_rate_bounds=(0.03, 0.25),
    random_immigrants=6,
)
```

### Staged Triangle-Count Workflow

Use staged optimization to grow representation complexity over time:

```python
from src.ga import (
    StageConfig,
    cross_over,
    fitness,
    mutate,
    run_staged_triangle_optimization,
)

hybrid_fitness = fitness.make_rmse_structure_fitness(1.0, 0.35)
stages = [
    StageConfig(
        n_triangles=40,
        generations=50,
        mutation_rate=0.14,
        crossover_rate=0.85,
        adaptive_mutation=True,
        mutation_rate_bounds=(0.06, 0.22),
        stagnation_window=12,
        random_immigrants=6,
    ),
    StageConfig(
        n_triangles=80,
        generations=70,
        mutation_rate=0.10,
        crossover_rate=0.85,
        adaptive_mutation=True,
        mutation_rate_bounds=(0.03, 0.16),
        stagnation_window=18,
        random_immigrants=4,
    ),
    StageConfig(
        n_triangles=120,
        generations=80,
        mutation_rate=0.08,
        crossover_rate=0.8,
        adaptive_mutation=True,
        mutation_rate_bounds=(0.02, 0.12),
        stagnation_window=24,
        random_immigrants=2,
    ),
]

result = run_staged_triangle_optimization(
    target=target_image,
    fitness_function=hybrid_fitness,
    population_size=120,
    stages=stages,
    elitism=2,
    crossover_function=cross_over.two_point_crossover,
    mutation_function=mutate.volatile_triangle_mutation,
    seeded=True,
)

best_fitness = result.best_fitness
history = result.history
best_individual = result.best_individual
stage_results = result.stage_results
```

`StageConfig.stagnation_window` is configured per stage and defaults to `8`, matching `GeneticAlgorithm`.

### Optional Genetic Operators

Optional genetic operators can be injected as notebook-local callables. If an operator is provided, its rate must also be provided.

```python
def my_crossover(parent1, parent2, crossover_rate):
    return parent1


def my_mutation(
    individual,
    mutation_rate,
    image_width,
    image_height,
    triangle_alpha_range,
):
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


def my_mutation(
    individual,
    mutation_rate,
    image_width,
    image_height,
    triangle_alpha_range,
):
    ...
```

`my_crossover(...)` may return a single child individual or a tuple/list of child individuals. `GeneticAlgorithm` will consume as many children as needed to fill the next population.

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
- Generation logs now include `mutation_rate_used`, `offspring_created`, `mutated_offspring`, `mutated_triangles`, and `immigrant_count`.

## Current Limitations

- The project is still prototype-focused and notebook-first.
- Automated coverage is still focused on initialization behavior, operator compatibility, and smoke checks.
- GA logs are stored in memory only and there is no persistence or visualization layer yet.
- Process evaluation is not optimized for the notebook-only workflow.
- Rendering is still PIL-based and may become the next performance bottleneck after evaluation parallelism.
- Rendering and image constants are still coupled to the current triangle representation.

## Immediate Next Steps

- Expand formal tests for GA evaluation backends, fitness ordering, logging, and best-individual tracking.
- Implement production crossover and mutation operators in `src/ga/cross_over.py` and `src/ga/mutate.py`.
- Add notebook cells for comparing sequential and thread evaluation timings.
- Add output saving for best rendered images, run metadata, and generation summaries.
- Consider elite-fitness caching to avoid recomputing unchanged elite individuals.
- Consider render optimization before relying on process-based parallelism in notebooks.

## File Tree

- `README.md`: Project overview, notebook workflow, setup instructions, progress status, limitations, and maintained file tree.
- `GeneticAlgorithm.md`: Detailed notebook-first reference for `src.ga.GeneticAlgorithm`.
- `src/ga/workflow.py`: Staged optimization helpers (`StageConfig`, `run_staged_triangle_optimization`).
- `AGENTS.md`: Repository instructions for GA terminology, API expectations, and project conventions.
- `main.py`: Minimal CLI entrypoint used for simple environment smoke checks.
- `pyproject.toml`: Project metadata and dependency declarations managed with `uv`.
- `uv.lock`: Locked dependency versions for reproducible installs.
- `.gitignore`: Ignore rules for Python, notebook, editor, cache, and generated artifacts.
- `.python-version`: Local Python version pin for the project environment.
- `images/girl_pearl_earing.png`: Sample target image used in notebook and smoke-test workflows.
- `notebooks/Exploration.ipynb`: Notebook for exploring rendering, fitness evaluation, package imports, and GA setup.
- `src/__init__.py`: Exposes the main package submodules and the `src.ga` package.
- `src/load_image.py`: Loads and resizes target images into NumPy arrays.
- `src/population.py`: Defines triangles, image constants, alpha sampling, and random or target-biased factories for individuals and populations.
- `src/rendering.py`: Renders individuals and converts rendered output to arrays.
- `src/ga/AGENTS.md`: Package-specific instructions for GA fitness, operators, and run API conventions.
- `src/ga/__init__.py`: Exports the public `GeneticAlgorithm` class, fitness helpers, and operator modules.
- `src/ga/algorithm.py`: Orchestrates GA state, optional operators, global-best history, and generation flow.
- `src/ga/cross_over.py`: Contains crossover operator examples for triangle individuals.
- `src/ga/evaluation.py`: Computes ordered fitness values with sequential, thread, or process evaluation backends.
- `src/ga/fitness.py`: Computes RMSE and RMSE+structure fitness values between target and generated images.
- `src/ga/logs.py`: Builds per-generation and run-level GA log dictionaries.
- `src/ga/mutate.py`: Contains mutation operator examples for triangle individuals.
- `src/ga/selection.py`: Implements tournament, ranking, and roulette-wheel parent selection strategies.
- `tests/test_seeded_initialization.py`: Unit tests for seeded RGB and alpha-range population initialization.
- `tests/test_rmse_structure_fitness.py`: Unit tests for RMSE+structure fitness factory and process-backend compatibility.
- `tests/test_selection.py`: Unit tests for parent selection dispatch and tournament-size keyword handling.
- `tests/test_crossover_children.py`: Regression tests for single-child and multi-child crossover compatibility in `GeneticAlgorithm`.
- `tests/test_workflow.py`: Regression tests for staged-workflow validation and `GeneticAlgorithm` parameter forwarding.
- `tests/test_imports.py`: Import smoke tests for the top-level package and the cleaned `src.ga` public API.
