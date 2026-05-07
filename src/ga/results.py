"""Saves and loads experiment run results for cross-run comparison.

Each run is stored as a JSON file in a results directory. Files contain the
pipeline name, all parameters, fitness metrics, convergence history, runtime,
and the best individual so images can be re-rendered without re-running.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from .. import population

Individual = list[population.Triangle]

_DEFAULT_RESULTS_DIR = Path(__file__).parents[2] / "results"


def _individual_to_json(individual: Individual) -> list[dict]:
    """Converts an Individual to a JSON-serializable list of dicts."""

    return [
        {
            "x1": t.x1, "y1": t.y1,
            "x2": t.x2, "y2": t.y2,
            "x3": t.x3, "y3": t.y3,
            "r": t.r, "g": t.g, "b": t.b, "a": t.a,
        }
        for t in individual
    ]


def _individual_from_json(data: list[dict]) -> Individual:
    """Reconstructs an Individual from a saved list of triangle dicts."""

    return [
        population.Triangle(
            x1=t["x1"], y1=t["y1"],
            x2=t["x2"], y2=t["y2"],
            x3=t["x3"], y3=t["y3"],
            r=t["r"], g=t["g"], b=t["b"], a=t["a"],
        )
        for t in data
    ]


def save_run(
    pipeline: str,
    parameters: dict,
    best_fitness: float,
    history: list[float],
    best_individual: Individual,
    runtime_seconds: float,
    notes: str = "",
    results_dir: Path | str = _DEFAULT_RESULTS_DIR,
) -> Path:
    """Saves one experiment run to a timestamped JSON file.

    Args:
        pipeline: Short label for the pipeline, e.g. ``"GA"``, ``"PSO-GA"``.
        parameters: Dict of all configuration values used in the run.
        best_fitness: Final best RMSE achieved.
        history: Per-generation global best fitness list.
        best_individual: Best triangle individual found.
        runtime_seconds: Wall-clock seconds the run took.
        notes: Optional free-text note about this run.
        results_dir: Directory to write JSON files into.

    Returns:
        Path to the saved JSON file.
    """

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_id = f"{pipeline.lower().replace(' ', '_')}_{timestamp}"
    filename = results_dir / f"{run_id}.json"

    payload = {
        "run_id": run_id,
        "pipeline": pipeline,
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": round(runtime_seconds, 1),
        "notes": notes,
        "parameters": parameters,
        "results": {
            "best_fitness": best_fitness,
            "generations_run": len(history),
            "history": history,
        },
        "best_individual": _individual_to_json(best_individual),
    }

    filename.write_text(json.dumps(payload, indent=2))
    print(f"Run saved → {filename.name}")
    return filename


def load_all_runs(
    results_dir: Path | str = _DEFAULT_RESULTS_DIR,
) -> list[dict]:
    """Loads all saved run JSON files and returns them as a list of dicts.

    Each dict has the same structure as what ``save_run`` wrote, with one
    extra key ``"individual"`` containing the reconstructed Individual.

    Returns:
        List of run dicts sorted by timestamp (oldest first).
    """

    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["individual"] = _individual_from_json(data["best_individual"])
            runs.append(data)
        except Exception:
            pass

    return runs


def runs_dataframe(results_dir: Path | str = _DEFAULT_RESULTS_DIR):
    """Returns a pandas DataFrame summarising all saved runs.

    Columns: run_id, pipeline, timestamp, runtime_seconds, notes,
    best_fitness, generations_run, and one column per parameter.
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for runs_dataframe()") from exc

    runs = load_all_runs(results_dir)
    if not runs:
        return pd.DataFrame()

    rows = []
    for r in runs:
        row = {
            "run_id": r["run_id"],
            "pipeline": r["pipeline"],
            "timestamp": r["timestamp"],
            "runtime_seconds": r["runtime_seconds"],
            "notes": r.get("notes", ""),
            "best_fitness": r["results"]["best_fitness"],
            "generations_run": r["results"]["generations_run"],
        }
        row.update(r.get("parameters", {}))
        rows.append(row)

    return pd.DataFrame(rows)
