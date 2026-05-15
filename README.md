# CIFO Project

Notebook-first prototype for approximating a target image with triangles evolved by a genetic algorithm. The project covers image loading, triangle-based individual generation, rendering, RMSE and structure-aware fitness evaluation, staged workflows, diversity variants, and parallel experiment helpers with file-based result caching.

## Project Status

- Core GA pipeline is working: image loading, population generation, rendering, fitness evaluation, and GA orchestration.
- The main workflow is `notebooks/Step_by_step_exploration.ipynb`, which runs systematic parameter analysis and cached experiment sweeps from notebooks.
- Parallel experiment infrastructure lives in `src/ga/parallel.py` (`GAConfig`, `run_single_ga`, `run_trials`, `run_grid_search`).
- Staged optimization helpers live in `src/ga/workflow.py`, and diversity-preserving GA variants live in `src/ga/diversity.py`.
- Results are persisted as timestamped JSON files under `results/<experiment>/` and loaded automatically on re-run so cells are instant when nothing changed.
- Available crossover operators: `single_point_crossover`, `single_point_crossover_two_children`, `two_point_crossover`, `two_point_crossover_two_children`, `cycle_crossover`, `pmx_crossover`.
- Available mutation operators: `random_triangle_mutation`, `focused_triangle_mutation`.
- Selection strategies: tournament, ranking, roulette-wheel.

## Setup

```bash
uv sync
uv run python main.py
```

## Notebook Workflow

Open `notebooks/Step_by_step_exploration.ipynb` with the project interpreter after `uv sync`. The first cell adds the project root to `sys.path`.

Use `notebooks/Grid_search_experiment.ipynb` for the guarded grid-search experiment across all configured hyperparameter setups. It keeps fitness sharing active, varies restricted mating and other GA operators, saves raw trial results to `results/grid_search_raw_results.csv`, and writes the aggregated setup summary to `results/grid_search_summary.csv`.

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

### Additional Workflows

- Use `fitness.make_rmse_structure_fitness(...)` when you need a picklable weighted RMSE + structure loss for process-based evaluation.
- Use `run_staged_triangle_optimization(...)` for coarse-to-fine triangle-count schedules.
- Use `FitnessSharingGA` or `RestrictedMatingGA` when you want diversity-preserving selection pressure instead of the baseline GA.

## Current Limitations

- Notebook-first; not yet packaged as a standalone CLI experiment runner.
- Rendering is PIL-based and is the main performance bottleneck per generation.
- Grid search supports only single-parameter sweeps; joint parameter sweeps require manual nested loops.

## File Tree

- `README.md` — project overview, setup, workflows, and current file tree.
- `GeneticAlgorithm.md` — reference guide for the main GA API and notebook usage patterns.
- `main.py` — minimal CLI smoke-check entrypoint.
- `pyproject.toml` — project metadata and dependencies managed with `uv`.
- `uv.lock` — locked dependency versions.
- `images/girl_pearl_earing.png` — target image used in the notebooks.
- `notebooks/Initial_analysis.ipynb` — earlier exploratory notebook for GA experiments.
- `notebooks/Exploration.ipynb` — additional notebook experimentation and analysis.
- `notebooks/Step_by_step_exploration.ipynb` — main experiment notebook with cached sweeps and visualizations.
- `notebooks/Grid_search_experiment.ipynb` — guarded sequential grid-search notebook with resumable CSV output.
- `results/` — cached experiment outputs plus `.gitkeep`; current subdirectories include runs such as `baseline/`, `crossover_grid/`, `elitism/`, `fitness_sharing/`, `mutation_rate/`, and `population_size/`.
- `src/__init__.py` — exposes the top-level package modules used from notebooks.
- `src/load_image.py` — loads and resizes target images to NumPy arrays.
- `src/population.py` — triangle datatypes plus random and seeded population factories.
- `src/rendering.py` — renders triangle individuals to images and arrays.
- `src/ga/__init__.py` — exports the GA public interface, helpers, and workflow entrypoints.
- `src/ga/algorithm.py` — baseline `GeneticAlgorithm` implementation and validation logic.
- `src/ga/cross_over.py` — triangle-list crossover operators, including one-child and two-child variants.
- `src/ga/diversity.py` — diversity-preserving GA variants with fitness sharing and restricted mating.
- `src/ga/evaluation.py` — sequential, thread, and process evaluation backends.
- `src/ga/fitness.py` — RMSE, structure-aware fitness helpers, and the picklable hybrid-fitness factory.
- `src/ga/logs.py` — per-generation and run-level logging payload helpers.
- `src/ga/mutate.py` — conservative and stronger triangle mutation operators.
- `src/ga/parallel.py` — parallel cached trial runners for notebooks.
- `src/ga/grid_search.py` — reusable helpers for the guarded grid-search notebook.
- `src/ga/results.py` — JSON result persistence and pandas loading helpers.
- `src/ga/selection.py` — tournament, ranking, and roulette-wheel parent selection.
- `src/ga/workflow.py` — staged triangle-count optimization workflow utilities.
- `tests/test_crossover_children.py` — validates crossover operators that return one or two children.
- `tests/test_imports.py` — protects the package import surface.
- `tests/test_rmse_structure_fitness.py` — checks the process-safe hybrid fitness factory.
- `tests/test_selection.py` — covers selection strategy behavior.
- `tests/test_workflow.py` — covers staged workflow behavior and process-backend compatibility.
