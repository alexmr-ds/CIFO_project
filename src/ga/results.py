"""
Saves and loads experiment run results as timestamped JSON files.

Every completed GA run is persisted to disk so results can be compared
across experiments without re-running the GA.  Each file contains:
  - Run metadata (id, pipeline label, timestamp, runtime, notes)
  - All configuration parameters that produced the result
  - Fitness metrics (best RMSE, convergence history)
  - The best individual (serialised triangle list) so the image can be
    re-rendered at any time without re-running the GA

Directory layout
----------------
Results are organised by experiment type:
    results/
      baseline/           <- baseline GA runs
      elitism/            <- elitism grid search runs
      population_size/    <- population size grid search runs
      ...

Each file is named:  <pipeline>_<timestamp>.json
e.g.  baseline-ga_2024-03-15T14-22-05.json
"""

import json
import time
from datetime import datetime
from pathlib import Path

from .. import population

Individual = list[population.Triangle]

# Default results directory — two levels above this file, i.e. project root/results
_DEFAULT_RESULTS_DIR = Path(__file__).parents[2] / "results"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _individual_to_json(individual: Individual) -> list[dict]:
    """
    Converts an Individual (list of Triangles) to a JSON-serialisable list.

    Each Triangle becomes a plain dict with 10 integer keys matching the
    Triangle dataclass field names.

    Args:
        individual: List of Triangle objects to serialise.

    Returns:
        List of dicts, one per triangle, with integer field values.
    """

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
    """
    Reconstructs an Individual from a saved list of triangle dicts.

    The inverse of ``_individual_to_json``.  Used when loading saved runs
    so the best individual can be re-rendered without re-running the GA.

    Args:
        data: List of triangle dicts as stored in the JSON file.

    Returns:
        List of Triangle dataclass objects.
    """

    return [
        population.Triangle(
            x1=t["x1"], y1=t["y1"],
            x2=t["x2"], y2=t["y2"],
            x3=t["x3"], y3=t["y3"],
            r=t["r"], g=t["g"], b=t["b"], a=t["a"],
        )
        for t in data
    ]


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

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
    """
    Saves one GA experiment run to a timestamped JSON file.

    The file is written atomically via Pathlib's ``write_text``.  If the
    results directory does not exist it is created automatically.

    The saved JSON schema:
    {
      "run_id":           "<pipeline>_<timestamp>",
      "pipeline":         "<label>",
      "timestamp":        "<ISO 8601>",
      "runtime_seconds":  <float>,
      "notes":            "<string>",
      "parameters":       { <all GA config keys> },
      "results": {
        "best_fitness":    <float>,
        "generations_run": <int>,
        "history":         [<float>, ...]
      },
      "best_individual":  [ { triangle dict }, ... ]
    }

    Args:
        pipeline:         Short label for the pipeline, e.g. "Baseline-GA".
        parameters:       Dict of all configuration values (from ga.params_dict()).
        best_fitness:     Final best RMSE achieved by this run.
        history:          Per-generation global best fitness list.
        best_individual:  The best triangle individual found.
        runtime_seconds:  Wall-clock seconds the run took.
        notes:            Optional free-text annotation for this run.
        results_dir:      Directory to write JSON files into.

    Returns:
        Path to the saved JSON file.
    """

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)  # create dir if it doesn't exist

    # Build a unique, human-readable filename from pipeline name + timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    run_id = f"{pipeline.lower().replace(' ', '_')}_{timestamp}"
    filename = results_dir / f"{run_id}.json"

    payload = {
        "run_id":           run_id,
        "pipeline":         pipeline,
        "timestamp":        datetime.now().isoformat(),
        "runtime_seconds":  round(runtime_seconds, 1),
        "notes":            notes,
        "parameters":       parameters,
        "results": {
            "best_fitness":    best_fitness,
            "generations_run": len(history),
            "history":         history,
        },
        # Store the best individual so it can be re-rendered without re-running
        "best_individual": _individual_to_json(best_individual),
    }

    filename.write_text(json.dumps(payload, indent=2))
    print(f"Run saved → {filename.name}")
    return filename


def load_all_runs(
    results_dir: Path | str = _DEFAULT_RESULTS_DIR,
) -> list[dict]:
    """
    Loads all saved run JSON files from a directory and returns them as dicts.

    Each returned dict mirrors the saved JSON schema with one extra key:
    ``"individual"`` containing the reconstructed Individual (list of
    Triangles) so callers can re-render images without manual parsing.

    Files that fail to parse (e.g. corrupt JSON) are silently skipped.

    Args:
        results_dir: Directory to scan for JSON files.

    Returns:
        List of run dicts sorted by filename (i.e. oldest first, since
        filenames contain timestamps).
    """

    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            # Reconstruct the Triangle objects from the stored list of dicts
            data["individual"] = _individual_from_json(data["best_individual"])
            runs.append(data)
        except Exception:
            # Skip malformed files rather than crashing
            pass

    return runs


def runs_dataframe(results_dir: Path | str = _DEFAULT_RESULTS_DIR):
    """
    Returns a pandas DataFrame summarising all saved runs in a directory.

    Each row is one run.  Fixed columns:
        run_id, pipeline, timestamp, runtime_seconds, notes,
        best_fitness, generations_run

    Plus one additional column for each key in the run's ``parameters`` dict
    (e.g. population_size, mutation_rate, elitism, etc.).

    Useful for quickly comparing results across experiments in a notebook::

        df = runs_dataframe("results/elitism")
        df.sort_values("best_fitness")

    Args:
        results_dir: Directory containing saved run JSON files.

    Returns:
        pandas DataFrame, one row per run.  Empty DataFrame if no runs found.

    Raises:
        ImportError: If pandas is not installed.
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
        # Start with the fixed metadata and result columns
        row = {
            "run_id":           r["run_id"],
            "pipeline":         r["pipeline"],
            "timestamp":        r["timestamp"],
            "runtime_seconds":  r["runtime_seconds"],
            "notes":            r.get("notes", ""),
            "best_fitness":     r["results"]["best_fitness"],
            "generations_run":  r["results"]["generations_run"],
        }
        # Flatten all parameter keys into the same row for easy comparison
        row.update(r.get("parameters", {}))
        rows.append(row)

    return pd.DataFrame(rows)
