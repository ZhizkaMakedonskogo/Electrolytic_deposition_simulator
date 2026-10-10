from __future__ import annotations

import ast
import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

from classes.simulation_objective import (
    METRIC_CHOICES,
    METRIC_COMBINED_L_NEW,
    METRIC_RSH_NOISE_ROBUST,
    METRIC_WEIGHTED_L,
    STEPS_PER_TRIAL,
    get_active_metric,
    set_active_metric,
)
from resources import benchmark_protocol as proto
from resources.constants import Algorithm, Mode

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "simulator_config.json"
HIST_DIR = ROOT / "config" / "histograms"

METRIC_UI_LABELS = {
    METRIC_WEIGHTED_L: "weighted_L (old L)",
    METRIC_COMBINED_L_NEW: "combined_L_new (JS+W1)",
    METRIC_RSH_NOISE_ROBUST: "rsh_noise_robust",
}


def metric_combobox_label(metric_name: str) -> str:
    return METRIC_UI_LABELS.get(metric_name, metric_name)


def apply_metric_from_config(cfg: Dict[str, Any]) -> str:
    name = str(cfg.get("metric", METRIC_WEIGHTED_L))
    return set_active_metric(name if name in METRIC_CHOICES else METRIC_WEIGHTED_L)


def _default_config() -> Dict[str, Any]:
    return {
        "version": 1,
        "description": "Parameters: rows=12, n_trials=50, n_repeats=3, metric=weighted_L",
        "save_session_json": True,
        "session_output_dir": "results/simulator_sessions",
        "metric": METRIC_WEIGHTED_L,
        "benchmark_protocol": {
            "rows": proto.DEFAULT_ROWS,
            "n_trials": proto.DEFAULT_N_TRIALS,
            "n_repeats": proto.DEFAULT_N_REPEATS,
            "n_runs": proto.DEFAULT_N_RUNS,
            "base_seed": proto.DEFAULT_BASE_SEED,
            "param_low": proto.PARAM_LOW,
            "param_high": proto.PARAM_HIGH,
            "steps_per_trial": STEPS_PER_TRIAL,
            "targets": ["hist1", "hist2", "hist3"],
            "pso_swarm_size": 0,
        },
        "gui_defaults": {
            "rows": 12,
            "mode": Mode.CONST.value,
            "algorithm": Algorithm.TPE.value,
            "n_trials": 50,
            "n_repeats": 3,
            "simulation_steps": STEPS_PER_TRIAL,
            "theory_preset": "hist1",
            "theory_file": "config/histograms/hist1.txt",
            "param_low": 0.0,
            "param_high": 1.0,
            "add": 0.5,
            "transit": 0.5,
            "merge": 0.5,
            "pso_swarm_size": 0,
        },
    }


def ensure_histogram_files() -> None:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("hist1", "hist2", "hist3"):
        theory = proto.load_theory_preset(name)
        path = HIST_DIR / f"{name}.txt"
        line = repr(theory)
        path.write_text(line + "\n", encoding="utf-8")


def ensure_default_config_file() -> Path:
    ensure_histogram_files()
    if not DEFAULT_CONFIG_PATH.exists():
        DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_config(_default_config(), DEFAULT_CONFIG_PATH)
    return DEFAULT_CONFIG_PATH


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    ensure_default_config_file()
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    if "metric" in data:
        set_active_metric(str(data["metric"]))
    return data


def load_config_file(path: str | Path) -> Dict[str, Any]:
    return load_config(path)


def save_config(updates: Dict[str, Any], path: Optional[str | Path] = None) -> Path:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    base = _default_config()
    if p.exists():
        try:
            base = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    for key, val in updates.items():
        if key == "gui_defaults" and isinstance(val, dict) and isinstance(base.get("gui_defaults"), dict):
            base["gui_defaults"].update(val)
        else:
            base[key] = val
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def parse_theory_text(raw: str) -> Dict[Any, float]:
    text = raw.strip().replace("{", "").replace("}", "")
    if not text:
        return {}
    out: Dict[Any, float] = {}
    for point in text.replace(" ", "").split(","):
        if not point or ":" not in point:
            continue
        a, b = point.split(":", 1)
        key = float(a)
        val = float(b)
        out[key if key != int(key) else int(key)] = val
    return out


def load_theory_file_into_app(app: Any, filepath: Path) -> bool:
    try:
        raw = filepath.read_text(encoding="utf-8")
    except OSError:
        return False
    joined = ",".join(
        ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")
    )
    for candidate in (joined, raw.strip()):
        if not candidate:
            continue
        try:
            if candidate.startswith("{"):
                app.theory = ast.literal_eval(candidate)
            else:
                app.theory = parse_theory_text(candidate)
        except (SyntaxError, ValueError, TypeError):
            continue
        if app.theory:
            return True
    return False


def apply_gui_defaults(app: Any, cfg: Optional[Dict[str, Any]] = None) -> None:
    cfg = cfg or load_config()
    g = cfg.get("gui_defaults", {})
    if not g:
        return

    def _set_entry(widget, value) -> None:
        try:
            widget.config(state="normal")
            widget.delete(0, "end")
            widget.insert(0, str(value))
        except Exception:
            pass

    if "rows" in g:
        _set_entry(app.textbox_height, g["rows"])
    if g.get("mode"):
        app.combobox_mode.set(g["mode"])
    if "add" in g:
        _set_entry(app.textbox_create, g["add"])
    if "transit" in g:
        _set_entry(app.textbox_ts, g["transit"])
    if "merge" in g:
        _set_entry(app.textbox_margin, g["merge"])
    if "simulation_steps" in g:
        _set_entry(app.textbox_count, g["simulation_steps"])
    if "n_trials" in g:
        _set_entry(app.textbox_opti, g["n_trials"])
    if "pso_swarm_size" in g and hasattr(app, "textbox_pso_swarm"):
        _set_entry(app.textbox_pso_swarm, g["pso_swarm_size"])
    if g.get("algorithm"):
        app.combobox_algo.set(g["algorithm"])

    preset = g.get("theory_preset")
    rel = g.get("theory_file")
    paths = []
    if rel:
        paths.append(ROOT / rel)
    if preset:
        paths.append(HIST_DIR / f"{preset}.txt")
        paths.append(ROOT / "config" / "targets" / f"{preset}.txt")
    for p in paths:
        if p.is_file() and load_theory_file_into_app(app, p):
            print(f"Theory loaded from {p}")
            break


def _session_dir(cfg: Dict[str, Any]) -> Path:
    rel = str(cfg.get("session_output_dir", "results/simulator_sessions"))
    d = ROOT / rel
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_session_payload(
    *,
    optimizer_name: str,
    run_result: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    method: Any = None,
) -> Dict[str, Any]:
    cfg = load_config()
    bp = cfg.get("benchmark_protocol", {})
    g = cfg.get("gui_defaults", {})
    payload: Dict[str, Any] = {
        "type": "simulator_gui_session",
        "timestamp": datetime.datetime.now().isoformat(),
        "metric": get_active_metric(),
        "protocol": {
            "rows": bp.get("rows", g.get("rows")),
            "n_trials": g.get("n_trials"),
            "n_repeats": g.get("n_repeats", bp.get("n_repeats")),
            "steps_per_trial": bp.get("steps_per_trial", STEPS_PER_TRIAL),
            "param_bounds": [bp.get("param_low", 0.0), bp.get("param_high", 1.0)],
            "theory_preset": g.get("theory_preset"),
        },
        "optimizer": optimizer_name,
        "gui_context": extra or {},
    }
    if run_result:
        payload["result"] = run_result
    if method is not None:
        payload["best_loss"] = getattr(method, "best_loss", None)
        payload["best_params"] = getattr(method, "best_params", None)
        payload["wall_time_sec"] = getattr(method, "optimization_wall_time_sec", None)
        if getattr(method, "theory", None):
            payload["theory_bins"] = len(method.theory)
    return payload


def save_session_json(payload: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    opt = str(payload.get("optimizer", "session")).replace(" ", "_")[:40]
    path = _session_dir(cfg) / f"simulator_session_{opt}_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_after_optimization(
    method: Any,
    *,
    algo_label: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    cfg = load_config()
    if not cfg.get("save_session_json", True):
        return None
    run_result = None
    if hasattr(method, "result"):
        try:
            run_result = method.result()
        except Exception:
            run_result = None
    payload = build_session_payload(
        optimizer_name=algo_label or type(method).__name__,
        run_result=run_result if isinstance(run_result, dict) else None,
        extra=context,
        method=method,
    )
    path = save_session_json(payload, cfg)
    return path