## GA Package Context

- This package implements the triangle-based genetic algorithm, its operators, fitness evaluation, logging, and parent selection.
- Keep `src.ga.fitness` as the canonical home for image fitness helpers.
- Lower fitness is better. Wrap any naturally higher-is-better metric as a lower-is-better loss before passing it to the GA.
- Process evaluation requires module-level, picklable fitness functions. Avoid lambdas and notebook-local closures with `evaluation_backend="process"`.

## Operator Interfaces

- Crossover functions must use `fn(parent1, parent2, crossover_rate) -> individual`.
- Mutation functions must use `fn(individual, mutation_rate, image_width, image_height, triangle_alpha_range) -> individual`.
- Mutation functions should preserve coordinates within image bounds and alpha values within `triangle_alpha_range`.
- `GeneticAlgorithm.run()` returns `(best_fitness, history)`, and `history` stores the global best fitness after each generation.
