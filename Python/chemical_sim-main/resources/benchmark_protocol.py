"""
Fixed experimental protocol for comparing optimizers

"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Tuple


PARAM_LOW: float = 0.0
PARAM_HIGH: float = 1.0
BOUNDS: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (PARAM_LOW, PARAM_HIGH),
    (PARAM_LOW, PARAM_HIGH),
    (PARAM_LOW, PARAM_HIGH),
)

DEFAULT_ROWS: int = 12
DEFAULT_N_TRIALS: int = 50
DEFAULT_N_REPEATS: int = 3
DEFAULT_N_RUNS: int = 5
DEFAULT_BASE_SEED: int = 42

TARGET_PRESETS: Dict[str, Tuple[str, str]] = {
    "hist1": ("resources.histograms", "hist1"),
    "hist2": ("resources.histograms", "hist2"),
    "hist3": ("resources.histograms", "hist3"),
    "target_file": ("", ""),  # stub; use --target-file
}


def load_theory_preset(name: str) -> Dict[Any, float]:
    if name not in TARGET_PRESETS or name == "target_file":
        raise ValueError(f"Unknown preset {name!r}; use one of: hist1, hist2, hist3")
    mod_path, attr = TARGET_PRESETS[name]
    import importlib

    mod = importlib.import_module(mod_path)
    return dict(getattr(mod, attr))


def load_theory_from_file(path: str | Path) -> Dict[Any, Any]:
   
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {p}")
    data = ast.literal_eval(text)
    if not isinstance(data, dict):
        raise TypeError("Expected a dict literal in target file")
    return dict(data)


def protocol_metadata(
    *,
    rows: int,
    n_trials: int,
    n_repeats: int,
    n_runs: int,
    base_seed: int,
    target_label: str,
) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "rows": rows,
        "n_trials": n_trials,
        "n_evaluations_per_trial_budget": n_trials,
        "n_repeats_per_evaluation": n_repeats,
        "param_bounds": {"low": PARAM_LOW, "high": PARAM_HIGH},
        "n_runs": n_runs,
        "base_seed": base_seed,
        "target": target_label,
        "steps_per_trial": None,  # filled in by the caller from simulation_objective.STEPS_PER_TRIAL
        "loss": "weighted_hist_compare (see classes.simulation_objective)",
    }