## Project Context

- This project approximates target images with triangle-based individuals evolved by a genetic algorithm.
- Use `uv` for Python dependency management and command execution.
- Keep `README.md` current when the project scope changes, and update its file tree whenever files are added or removed.

## GA Terminology And API

- Use `fitness` for objective values in GA code, docs, logs, and comments.
- Rename legacy objective-value wording to `fitness` whenever it appears.
- Lower fitness is better because the default fitness function is RMSE.
- `GeneticAlgorithm.run()` returns `(best_fitness, history)`.
- `history` stores the global best fitness after each generation.
- The best triangle configuration is stored on `ga.best_individual`.
- When `logs=True`, `ga.run_logs` is a summary dictionary containing generation fitness logs, `best_fitness`, and a JSON-friendly `best_individual_configuration`.

## Python Conventions

- Every Python module starts with a short one-line triple-quoted docstring stating its purpose.
- Keep inline comments brief and add them only when intent is non-obvious.
- Prefer module-level imports for local modules, such as `from . import models`, then reference names explicitly.
- Standard-library and third-party symbol imports are acceptable when conventional, such as `from dataclasses import dataclass`.
