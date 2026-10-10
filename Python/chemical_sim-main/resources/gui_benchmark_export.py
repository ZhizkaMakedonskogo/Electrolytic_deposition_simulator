"""
Export of the GUI "Compare p(k) modes" results to the benchmark_inverse-*.json format
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from classes.simulation_objective import SimulationObjective, get_active_metric
from resources import benchmark_protocol as proto
from resources.constants import (
    ALGORITHMS_MODE_COMPARE_BUNDLE,
    SERIES_Q_BENCHMARK_KEYS,
    Algorithm,
)

ROOT = Path(__file__).resolve().parent.parent

ALGO_LABEL_TO_BENCHMARK_KEY: Dict[str, str] = {
    Algorithm.DOWNHILL.value: "coord_descent",
    Algorithm.TPE.value: "tpe",
    Algorithm.XNES.value: "xnes",
    Algorithm.CMA_ES.value: "cma",
    Algorithm.GP_BAYES.value: "gp",
    Algorithm.SKOPT_GP.value: "skopt",
    Algorithm.MULTI_FIDELITY.value: "mfo",
    Algorithm.PSO.value: "pso",
    Algorithm.GA.value: "ga",
    Algorithm.DE.value: "de",
    Algorithm.TRUST_REGION_BO.value: "trust_bo",
    Algorithm.TURBO_MT.value: "turbo_mt",
    Algorithm.HYPERBAND_TPE.value: "hyperband",
    Algorithm.LHS_NM.value: "lhs_nm",
    Algorithm.RF_EI.value: "rf_ei",
    Algorithm.CEM.value: "cem",
    Algorithm.SA.value: "sa",
}

BENCH_KEY_TO_JSON_KEY: Dict[str, str] = {
    "tpe": "tpe",
    "skopt": "skopt_gp_ei",
    "xnes": "xnes",
    "de": "de",
    "gp": "gp",
    "mfo": "mfo",
    "pso": "pso",
    "ga": "ga",
    "trust_bo": "trust_bo",
    "lhs_nm": "lhs_nm",
    "rf_ei": "rf_ei",
    "cem": "cem",
    "sa": "sa",
    "cma": "cma",
    "hyperband": "hyperband",
    "turbo_mt": "turbo_mt",
    "coord_descent": "coord_descent",
}

BENCHMARK_JSON_KEY_TO_ALGO_LABEL: Dict[str, str] = {
    "tpe": Algorithm.TPE.value,
    "skopt_gp_ei": Algorithm.SKOPT_GP.value,
    "xnes": Algorithm.XNES.value,
    "de": Algorithm.DE.value,
    "gp": Algorithm.GP_BAYES.value,
    "mfo": Algorithm.MULTI_FIDELITY.value,
    "pso": Algorithm.PSO.value,
    "ga": Algorithm.GA.value,
    "trust_bo": Algorithm.TRUST_REGION_BO.value,
    "lhs_nm": Algorithm.LHS_NM.value,
    "rf_ei": Algorithm.RF_EI.value,
    "cem": Algorithm.CEM.value,
    "sa": Algorithm.SA.value,
    "cma": Algorithm.CMA_ES.value,
    "hyperband": Algorithm.HYPERBAND_TPE.value,
    "turbo_mt": Algorithm.TURBO_MT.value,
    "coord_descent": Algorithm.DOWNHILL.value,
}

_bi_module: Any = None


def _load_benchmark_inverse() -> Any:
    global _bi_module
    if _bi_module is not None:
        return _bi_module
    path = ROOT / "scripts" / "benchmark_inverse.py"
    spec = importlib.util.spec_from_file_location("benchmark_inverse", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load benchmark module from {path}")
    mod = importlib.util.module_from_spec(spec)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    _bi_module = mod
    return mod


def mode_compare_method_keys() -> List[str]:
    keys: List[str] = []
    for algo in ALGORITHMS_MODE_COMPARE_BUNDLE:
        k = ALGO_LABEL_TO_BENCHMARK_KEY.get(algo)
        if k:
            keys.append(k)
    return keys


def benchmark_key_display_name(key: str) -> str:
    k = str(key).strip().lower()
    for label, bk in ALGO_LABEL_TO_BENCHMARK_KEY.items():
        if bk == k:
            return label
    return k


def list_selectable_benchmark_methods() -> List[Tuple[str, str]]:
    bi = _load_benchmark_inverse()
    out: List[Tuple[str, str]] = []
    for key in bi.ALL_METHODS_17:
        out.append((key, benchmark_key_display_name(key)))
    return out


BENCHMARK_PRESETS: Dict[str, List[str]] = {
    "all": mode_compare_method_keys(),
    "series_q": list(SERIES_Q_BENCHMARK_KEYS),
    "pso_tpe": ["tpe", "pso"],
    "ch8_four": ["tpe", "pso", "skopt", "de"],
}


def resolve_method_keys(selected: Optional[List[str]] = None) -> List[str]:
    if not selected:
        return mode_compare_method_keys()
    bi = _load_benchmark_inverse()
    valid = set(bi.ALL_METHODS_17)
    keys = [str(k).strip().lower() for k in selected if str(k).strip().lower() in valid]
    if not keys:
        raise ValueError("No valid optimization method selected.")
    return keys


def _normalize_theory(theory: dict) -> Dict[float, float]:
    return {float(k): float(v) for k, v in theory.items()}


def resolve_target_label(theory: dict) -> str:
    norm = _normalize_theory(theory)
    for name in ("hist1", "hist2", "hist3"):
        try:
            ref = _normalize_theory(proto.load_theory_preset(name))
        except ValueError:
            continue
        if ref == norm:
            return f"preset:{name}"
    return "gui:custom"


class _HeadlessSleeper:
    is_exit = False
    scale_sleep = SimpleNamespace(get=lambda: 0)

    def sleep(self) -> None:
        pass


def experimental_bar_from_params(
    rows: int,
    theory: dict,
    best_params: dict,
    n_repeats: int,
    *,
    sleeper: Optional[Any] = None,
) -> dict:
    """Reconstruct the p(k) histogram from best_params (best repeat)"""
    sim = SimulationObjective(rows=rows, theory=theory, sleeper=sleeper or _HeadlessSleeper())
    best_hist: dict = {}
    best_l = float("inf")
    for rep in range(max(1, n_repeats)):
        if getattr(sleeper, "is_exit", False):
            break
        loss, hist = sim.run_one_repeat(
            float(best_params["add"]),
            float(best_params["transit"]),
            float(best_params["merge"]),
            eval_index=0,
            rep=rep,
        )
        if loss < best_l:
            best_l = loss
            best_hist = hist
    return best_hist


def mode_compare_results_from_run(
    methods: Dict[str, Any],
    theory: dict,
    *,
    rows: int,
    n_repeats: int,
    method_keys: Optional[List[str]] = None,
    sleeper: Optional[Any] = None,
) -> Tuple[List[Tuple[str, float, dict]], List[str]]:
    results: List[Tuple[str, float, dict]] = []
    errors: List[str] = []
    keys = resolve_method_keys(method_keys) if method_keys else mode_compare_method_keys()
    for bench_key in keys:
        json_key = BENCH_KEY_TO_JSON_KEY.get(bench_key, bench_key)
        algo = BENCHMARK_JSON_KEY_TO_ALGO_LABEL.get(json_key, benchmark_key_display_name(bench_key))
        block = methods.get(json_key)
        if block is None:
            errors.append(f"{algo}: no data in the run")
            continue
        bl = block.get("best_loss")
        params = block.get("best_params")
        if bl is None or not isinstance(params, dict):
            errors.append(f"{algo}: empty best_loss/best_params")
            continue
        try:
            br = experimental_bar_from_params(
                rows, theory, params, n_repeats, sleeper=sleeper
            )
            results.append((algo, float(bl), br))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{algo}: {e!s}")
    return results, errors


def run_mode_compare_benchmark(
    theory: dict,
    *,
    rows: int,
    n_trials: int,
    n_repeats: int,
    n_runs: int,
    base_seed: int,
    param_low: float,
    param_high: float,
    metric: str,
    method_keys: Optional[List[str]] = None,
    mfo_budget: float = 0.0,
    pso_swarm_size: int | None = None,
    method_options: Optional[Dict[str, Dict[str, Any]]] = None,
    base_seeds: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Path:
    """Headless run of the selected methods. Saves the benchmark JSON report"""
    bi = _load_benchmark_inverse()
    target_id = resolve_target_label(theory)
    method_keys = resolve_method_keys(method_keys)
    total_steps = len(method_keys) * max(1, n_runs)
    step = 0

    target_block: Dict[str, Any] = {"id": target_id, "runs": []}
    for r in range(n_runs):
        if should_stop and should_stop():
            break
        lhs_seed = base_seed + r * 100_000
        run_record: Dict[str, Any] = {"run_index": r, "lhs_seed": lhs_seed, "methods": {}}
        for m in method_keys:
            if should_stop and should_stop():
                break
            step += 1
            if on_progress:
                on_progress(step, total_steps, m)
            run_record["methods"].update(
                bi.run_single_benchmark_method(
                    m,
                    rows=rows,
                    theory=theory,
                    n_trials=n_trials,
                    n_repeats=n_repeats,
                    lhs_seed=lhs_seed,
                    param_low=param_low,
                    param_high=param_high,
                    mfo_budget=mfo_budget,
                    pso_swarm_size=pso_swarm_size,
                    method_options=method_options,
                )
            )
        target_block["runs"].append(run_record)

    from classes.pso_optimization import resolve_swarm_size

    pso_opts = (method_options or {}).get("pso") or {}
    swarm_from_opts = pso_opts.get("swarm_size")
    swarm_for_meta = (
        int(swarm_from_opts)
        if swarm_from_opts is not None and int(swarm_from_opts) > 0
        else pso_swarm_size
    )
    meta: Dict[str, Any] = {
        "rows": rows,
        "n_trials": n_trials,
        "n_repeats": n_repeats,
        "n_runs": n_runs,
        "base_seed": base_seed,
        "param_bounds": [param_low, param_high],
        "steps_per_trial": bi.STEPS_PER_TRIAL,
        "timestamp": datetime.datetime.now().isoformat(),
        "metric": metric,
        "pso_swarm_size": int(swarm_for_meta or 0),
        "pso_swarm_size_effective": resolve_swarm_size(n_trials, swarm_for_meta),
        "source": "gui_mode_compare",
        "method_keys": method_keys,
        "protocol": "flexible_benchmark",
    }
    if method_options:
        meta["method_options"] = method_options
    if base_seeds and len(base_seeds) > 1:
        meta["base_seeds"] = list(base_seeds)

    payload = {
        "metadata": meta,
        "targets": [target_block],
    }
    payload["summary"] = {target_id: bi._target_risk_summary(target_block)}

    if output_path is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        tag = "-".join(method_keys[:4])
        if len(method_keys) > 4:
            tag += f"-n{len(method_keys)}"
        seed_tag = f"-seed{base_seed}" if base_seeds and len(base_seeds) > 1 else ""
        output_path = ROOT / "results" / f"benchmark_flex-{tag}{seed_tag}-{ts}.json"
    bi.save_benchmark_payload(payload, Path(output_path))
    return Path(output_path)


def _target_hist_arg(target_id: str) -> str:
    tid = str(target_id)
    if tid.startswith("preset:"):
        return tid.split(":", 1)[1]
    return "hist1"


def generate_benchmark_figures(
    json_path: Path,
    *,
    plot_median: bool = True,
    plot_pk_top: bool = True,
    plot_splits: bool = True,
    top_n: int = 5,
    run_index: int = -1,
    lang: str = "ru",
) -> List[Path]:
    """Builds PNGs from a saved benchmark JSON"""
    import subprocess

    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.is_file():
        py = Path(sys.executable)

    json_path = Path(json_path)
    out: List[Path] = []
    out_dir = json_path.parent / "figures"

    if plot_median:
        cmd = [
            str(py),
            str(ROOT / "scripts" / "plot_benchmark_results.py"),
            str(json_path),
            "--out-dir",
            str(out_dir),
            "--style",
            "horizontal",
            "--lang",
            lang,
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=False)
        for p in sorted(out_dir.glob("benchmark_hbar_*.png")):
            if p.stat().st_mtime >= json_path.stat().st_mtime - 2:
                out.append(p)

    if plot_pk_top:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            targets = payload.get("targets") or []
            tid = targets[0].get("id", "preset:hist1") if targets else "preset:hist1"
            hist_arg = _target_hist_arg(tid)
            runs = targets[0].get("runs") or [] if targets else []
            ri = run_index if run_index >= 0 else (len(runs) - 1 if runs else 0)
            cmd = [
                str(py),
                str(ROOT / "scripts" / "plot_benchmark_hist_lines.py"),
                str(json_path),
                "--target",
                hist_arg,
                "--run-index",
                str(max(0, ri)),
                "--top-n",
                str(int(top_n)),
                "--out-dir",
                str(out_dir),
                "--lang",
                lang,
            ]
            subprocess.run(cmd, cwd=str(ROOT), check=False)
            safe = re.sub(r"[^\w\-]+", "_", tid)[:40]
            for p in sorted(out_dir.glob(f"pk_lines_top{top_n}_{safe}_run*.png")):
                if p.stat().st_mtime >= json_path.stat().st_mtime - 2:
                    out.append(p)
        except Exception:
            pass

    if plot_splits:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            targets = payload.get("targets") or []
            tid = targets[0].get("id", "preset:hist1") if targets else "preset:hist1"
            hist_arg = _target_hist_arg(tid)
            runs = targets[0].get("runs") or [] if targets else []
            ri = run_index if run_index >= 0 else (len(runs) - 1 if runs else 0)
            split_dir = out_dir / re.sub(r"[^\w\-]+", "_", json_path.stem)[:48]
            cmd = [
                str(py),
                str(ROOT / "scripts" / "plot_benchmark_hist_splits.py"),
                str(json_path),
                "--target",
                hist_arg,
                "--run-index",
                str(max(0, ri)),
                "--top-n",
                str(int(top_n)),
                "--out-dir",
                str(split_dir),
                "--lang",
                lang,
            ]
            subprocess.run(cmd, cwd=str(ROOT), check=False)
            for p in sorted(split_dir.glob("split_*.png")):
                if p.stat().st_mtime >= json_path.stat().st_mtime - 2:
                    out.append(p)
            for p in sorted(split_dir.glob("theory_*.png")):
                if p.stat().st_mtime >= json_path.stat().st_mtime - 2:
                    out.append(p)
        except Exception:
            pass

    return out


def benchmark_protocol_defaults() -> Dict[str, Any]:
    from resources.simulator_config import load_config

    cfg = load_config()
    bp = dict(cfg.get("benchmark_protocol") or {})
    bp.setdefault("rows", proto.DEFAULT_ROWS)
    bp.setdefault("n_trials", proto.DEFAULT_N_TRIALS)
    bp.setdefault("n_repeats", proto.DEFAULT_N_REPEATS)
    bp.setdefault("n_runs", proto.DEFAULT_N_RUNS)
    bp.setdefault("base_seed", proto.DEFAULT_BASE_SEED)
    bp.setdefault("param_low", proto.PARAM_LOW)
    bp.setdefault("param_high", proto.PARAM_HIGH)
    bp.setdefault("pso_swarm_size", 0)
    return bp