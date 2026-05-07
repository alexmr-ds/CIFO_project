# CIFO Project

Notebook-first prototype for approximating a target image with semi-transparent triangles using a genetic algorithm. The pipeline covers image loading, triangle-based individual generation, rendering, RMSE fitness evaluation, and a parallel experiment framework with file-based result caching.

## Project Status

- Core GA pipeline is working: image loading, population generation, rendering, fitness evaluation, and GA orchestration.
- The main workflow is `notebooks/Step_by_step_exploration.ipynb`, which runs systematic parameter contribution analysis (baseline, population size, elitism, crossover type/rate, mutation rate) with 5 trials each and cached results.
- Parallel experiment infrastructure lives in `src/ga/parallel.py` (`GAConfig`, `run_trials`, `run_grid_search`).
- Results are persisted as timestamped JSON files under `results/<experiment>/` and loaded automatically on re-run so cells are instant when nothing changed.
- Available crossover operators: `single_point_crossover`, `single_point_crossover_two_children`, `two_point_crossover`, `two_point_crossover_two_children`.
- Available mutation operator: `random_triangle_mutation`.
- Selection strategies: tournament, ranking, roulette-wheel.

## Setup

```bash
uv sync
uv run python main.py
```

## Notebook Workflow

Open `notebooks/Step_by_step_exploration.ipynb` with the project interpreter after `uv sync`. The first cell adds the project root to `sys.path`.

### Running Parallel Experiments

The recommended way to run experiments in the notebook is via `GAConfig` + `run_trials` / `run_grid_search`:

```python
from src.ga.parallel import GAConfig, run_trials, run_grid_search
from src.ga import fitness, mutate, cross_over

config = GAConfig(
    target=target_array,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=100,
    crossover_function=cross_over.two_point_crossover_two_children,
    crossover_rate=0.9,
    mutation_function=mutate.random_triangle_mutation,
    mutation_rate=0.1,
    elitism=1,
    selection_type="tournament",
    triangle_alpha_range=(255, 255),
)

# Run 5 independent trials, results cached to disk
summary = run_trials(
    config=config,
    n_trials=5,
    pipeline="MyExperiment",
    results_dir=project_root / "results" / "my_experiment",
)

print(summary.mean_fitness, summary.std_fitness)
```

For grid searches over a single parameter:

```python
results = run_grid_search(
    base_config=config,
    grid={"mutation_rate": [0.05, 0.10, 0.20]},
    n_trials=5,
    pipeline_prefix="MutRate",
    results_dir=project_root / "results" / "mutation_rate",
)
# results is keyed by the raw parameter value, e.g. results[0.05]
```

Both functions support partial caching: if some trials already exist on disk, only the missing ones are run.

### Using GeneticAlgorithm Directly

For one-off runs or custom experiments, `GeneticAlgorithm` can be used directly:

```python
from src.ga import GeneticAlgorithm, fitness, mutate, cross_over
from src import load_image

target = load_image.load_target_image("images/girl_pearl_earing.png")

ga = GeneticAlgorithm(
    target=target,
    fitness_function=fitness.compute_rmse,
    population_size=100,
    generations=100,
    crossover_function=cross_over.two_point_crossover_two_children,
    crossover_rate=0.9,
    mutation_function=mutate.random_triangle_mutation,
    mutation_rate=0.1,
    elitism=1,
    selection_type="tournament",
    triangle_alpha_range=(255, 255),
)

best_fitness, history = ga.run()
best_individual = ga.best_individual
```

### Triangle Transparency

Control alpha range during initialization and mutation:

```python
GAConfig(
    ...
    triangle_alpha_range=(255, 255),  # fully opaque
    # triangle_alpha_range=(40, 180)  # semi-transparent
)
```

Use `(255, 255)` for fully opaque triangles (recommended for cleaner convergence). The default `(5, 255)` allows semi-transparent triangles.

## Current Limitations

- Notebook-first; not yet packaged as a standalone CLI experiment runner.
- Rendering is PIL-based and is the main performance bottleneck per generation.
- Grid search supports only single-parameter sweeps; joint parameter sweeps require manual nested loops.

## File Tree

- `README.md` — project overview, setup, and workflow reference.
- `main.py` — minimal CLI smoke-check entrypoint.
- `pyproject.toml` — project metadata and dependencies managed with `uv`.
- `uv.lock` — locked dependency versions.
- `images/girl_pearl_earing.png` — target image used in the notebook.
- `notebooks/Step_by_step_exploration.ipynb` — main analysis notebook: baseline + parameter grid searches with caching and visualization.
- `results/` — cached JSON run results, organised by experiment (`baseline/`, `elitism/`, `population_size/`, `crossover_grid/`, `mutation_rate/`).
- `src/__init__.py` — exposes `load_image`, `population`, `rendering`.
- `src/load_image.py` — loads and resizes target images to NumPy arrays.
- `src/population.py` — `Triangle` dataclass, alpha sampling, random individual/population factories.
- `src/rendering.py` — renders triangle individuals to PIL images and converts to arrays.
- `src/ga/__init__.py` — exports `GeneticAlgorithm` and operator modules.
- `src/ga/algorithm.py` — GA orchestration: selection, crossover, mutation, elitism, history tracking.
- `src/ga/parallel.py` — `GAConfig`, `TrialSummary`, `run_trials`, `run_grid_search` for parallel cached experiments.
- `src/ga/results.py` — saves and loads per-run JSON result files; `runs_dataframe()` for pandas summaries.
- `src/ga/fitness.py` — `compute_rmse` fitness function.
- `src/ga/cross_over.py` — crossover operators: single-point, two-point, one-child and two-children variants.
- `src/ga/mutate.py` — `random_triangle_mutation` operator.
- `src/ga/selection.py` — tournament, ranking, and roulette-wheel parent selection.
- `src/ga/evaluation.py` — sequential, thread, and process fitness evaluation backends.
- `src/ga/logs.py` — per-generation and run-level log formatting.
- `tests/` — unit and smoke tests for initialization, selection, crossover, imports.
