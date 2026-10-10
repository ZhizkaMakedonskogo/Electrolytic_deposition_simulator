#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classes.multi_fidelity_optimizer import MultiFidelityOptimizer
from classes.optuna_integration import OptunaIntegration
from classes.simulation_objective import (
    METRIC_CHOICES,
    STEPS_PER_TRIAL,
    SimulationObjective,
    default_n_init,
    latin_hypercube_initial,
    set_active_metric,
    weighted_hist_compare,
)
from classes.skopt_gp_optimization import SkoptGpOptimization
from classes.xnes_optimization import XnesOptimization
from classes.pso_optimization import PsoOptimization
from classes.ga_optimization import GaOptimization
from classes.trust_region_bo import TrustRegionBoOptimization
from classes.lhs_nelder_mead import LhsNelderMeadOptimization
from classes.rf_ei_optimization import RfEiOptimization
from classes.cem_optimization import CemOptimization
from classes.simulated_annealing_optimization import SimulatedAnnealingOptimization
from classes.differential_evolution_optimization import DifferentialEvolutionOptimization
from resources import benchmark_protocol as proto

from classes.gp_bayesian_optimization import GPBayesianOptimization
from classes.CMA_ES_3 import CmaEsOptunaIntegration
from classes.optuna_hyperband_integration import OptunaHyperbandIntegration
from classes.turbo_mt_optimization import TurboMtOptimization
from classes.coordinate_descent import CoordinateDescent

ALL_METHODS_17: Tuple[str, ...] = (
    "tpe",
    "skopt",
    "xnes",
    "de",
    "pso",
    "ga",
    "trust_bo",
    "lhs_nm",
    "rf_ei",
    "cem",
    "sa",
    "gp",
    "mfo",
    "cma",
    "hyperband",
    "turbo_mt",
    "coord_descent",
)

class _StubLabel:
    def config(self, **kwargs):
        pass

def _after(_ms, fn):
    fn()

class MockDrawer:
    def __init__(self) -> None:
        self.stat_1 = _StubLabel()
        self.stat_2 = _StubLabel()
        self.stat_3 = _StubLabel()
        self.window = SimpleNamespace(
            after=_after,
            winfo_exists=lambda: True,
            update_idletasks=lambda: None,
            update=lambda: None,
        )

    def prepare_draw(self):
        pass

    def complete_draw(self):
        pass

    def draw_bar(self, bar: dict):
        pass

    def draw_point(self, row: int, col: int, color_idx: int):
        pass

class MockSleeper:
    is_exit = False
    scale_sleep = SimpleNamespace(get=lambda: 0)

    def sleep(self):
        pass

def _run_tpe(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    tpe_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = OptunaIntegration(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        lhs_seed=lhs_seed,
        tpe_seed=tpe_seed,
        param_low=param_low,
        param_high=param_high,
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "tpe",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_skopt(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    skopt_rs: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = SkoptGpOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        lhs_seed=lhs_seed,
        skopt_random_state=skopt_rs,
        param_low=param_low,
        param_high=param_high,
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "skopt_gp_ei",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_xnes(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    np.random.seed(rng_seed)
    d = MockDrawer()
    s = MockSleeper()
    opt = XnesOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        param_low=param_low,
        param_high=param_high,
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "xnes",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_de(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    seed: int,
    param_low: float,
    param_high: float,
    *,
    population_size: int = 0,
) -> Dict[str, Any]:

    d = MockDrawer()
    s = MockSleeper()
    opt = DifferentialEvolutionOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        param_low=param_low,
        param_high=param_high,
        rng_seed=seed,
        population_size=int(population_size or 0),
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "differential_evolution",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params or {"add": 0.5, "transit": 0.5, "merge": 0.5},
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
        "note": "Custom DE/rand/1/bin; evaluation count equals n_trials (see DifferentialEvolutionOptimization).",
    }


def _run_gp(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    gpr_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = GPBayesianOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        lhs_seed=lhs_seed,
        gpr_random_state=gpr_seed,
        param_low=param_low,
        param_high=param_high,
    )
    t0 = time.perf_counter()
    opt.optimize(n_trials=n_trials)
    wall = time.perf_counter() - t0
    return {
        "method": "gp_bayesian_sklearn_ei",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": wall,
        "n_trials_effective": n_trials,
    }


def _run_pso(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
    swarm_size: int | None = None,
    *,
    inertia: float | None = None,
    cognitive: float | None = None,
    social: float | None = None,
) -> Dict[str, Any]:
    from classes.pso_optimization import resolve_swarm_size

    d = MockDrawer()
    s = MockSleeper()
    swarm_eff = resolve_swarm_size(n_trials, swarm_size)
    pso_kwargs: Dict[str, Any] = {
        "rows": rows,
        "theory": theory,
        "drawer": d,
        "sleeper": s,
        "n_repeats": n_repeats,
        "param_low": param_low,
        "param_high": param_high,
        "swarm_size": swarm_eff,
        "rng_seed": rng_seed,
    }
    if inertia is not None:
        pso_kwargs["inertia"] = float(inertia)
    if cognitive is not None:
        pso_kwargs["cognitive"] = float(cognitive)
    if social is not None:
        pso_kwargs["social"] = float(social)
    opt = PsoOptimization(**pso_kwargs)
    opt.optimize(n_trials=n_trials)
    return {
        "method": "pso",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
        "swarm_size_effective": swarm_eff,
    }


def _run_ga(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
    *,
    population_size: int | None = None,
    tournament_k: int | None = None,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    pop_eff = (
        int(population_size)
        if population_size is not None and int(population_size) > 0
        else min(16, max(6, n_trials // 3))
    )
    ga_kwargs: Dict[str, Any] = {
        "rows": rows,
        "theory": theory,
        "drawer": d,
        "sleeper": s,
        "n_repeats": n_repeats,
        "param_low": param_low,
        "param_high": param_high,
        "population_size": pop_eff,
        "rng_seed": rng_seed,
    }
    if tournament_k is not None:
        ga_kwargs["tournament_k"] = int(tournament_k)
    opt = GaOptimization(**ga_kwargs)
    opt.optimize(n_trials=n_trials)
    return {
        "method": "ga",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_trust_bo(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = TrustRegionBoOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        param_low=param_low,
        param_high=param_high,
        lhs_seed=lhs_seed,
        rng_seed=rng_seed,
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "trust_region_bo",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_rf_ei(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = RfEiOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        param_low=param_low,
        param_high=param_high,
        lhs_seed=lhs_seed,
        rng_seed=rng_seed,
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "rf_ei",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_cem(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = CemOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        param_low=param_low,
        param_high=param_high,
        rng_seed=rng_seed,
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "cem",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_sa(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = SimulatedAnnealingOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        param_low=param_low,
        param_high=param_high,
        rng_seed=rng_seed,
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "simulated_annealing",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_lhs_nm(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = LhsNelderMeadOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        param_low=param_low,
        param_high=param_high,
        lhs_seed=lhs_seed,
        rng_seed=rng_seed,
        n_starts=min(4, max(2, n_trials // 15)),
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "lhs_nelder_mead",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_cma(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    cma_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = CmaEsOptunaIntegration(rows=rows, theory=theory, drawer=d, sleeper=s)
    opt._n_repeats = n_repeats
    opt._lhs_seed = lhs_seed
    opt._tpe_seed = cma_seed
    opt._param_low = param_low
    opt._param_high = param_high
    opt.optimize(n_trials=n_trials)
    return {
        "method": "cma_es",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_hyperband(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    tpe_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = OptunaHyperbandIntegration(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        lhs_seed=lhs_seed,
        tpe_seed=tpe_seed,
    )
    opt._param_low = param_low
    opt._param_high = param_high
    opt.optimize(n_trials=n_trials)
    return {
        "method": "hyperband_tpe",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
        "warning": (
            "Hyperband may prune trials before full n_repeats; effective simulation count "
            "can be lower than n_trials * n_repeats. Compare with caution vs one-fidelity methods."
        ),
    }


def _run_turbo_mt(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = TurboMtOptimization(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        param_low=param_low,
        param_high=param_high,
        lhs_seed=lhs_seed,
        rng_seed=rng_seed,
    )
    opt.optimize(n_trials=n_trials)
    return {
        "method": "turbo_mt",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "n_trials_effective": n_trials,
    }


def _run_coord_descent(
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    n_starts = min(default_n_init(n_trials), max(1, n_trials // 6))
    starts = latin_hypercube_initial(n_starts, lhs_seed, param_low, param_high)
    t0 = time.perf_counter()
    best_loss = float("inf")
    best_params: Dict[str, float] = {"add": 0.5, "transit": 0.5, "merge": 0.5}
    eval_used = 0
    for si, (add, transit, merge) in enumerate(starts):
        opt = CoordinateDescent(
            rows,
            float(add),
            float(transit),
            float(merge),
            d,
            s,
            theory,
            multiplier=0.05,
        )
        opt._n_repeats = n_repeats
        opt._eval_idx = si * 1000
        opt.optimize()
        eval_used += max(1, len(opt._hist))
        if opt.best_loss < best_loss:
            best_loss = float(opt.best_loss)
            if opt._hist:
                b = min(opt._hist, key=lambda t: t[3])
                best_params = {"add": float(b[0]), "transit": float(b[1]), "merge": float(b[2])}
    wall = time.perf_counter() - t0
    return {
        "method": "coordinate_descent",
        "best_loss": best_loss,
        "best_params": best_params,
        "wall_time_sec": wall,
        "n_starts": n_starts,
        "note": (
            "Coordinate descent: n_starts LHS passes, each with a small fixed number of "
            "evaluate() calls; not aligned 1:1 with n_trials budget of other optimizers."
        ),
    }


def _run_mfo(
    rows: int,
    theory: dict,
    budget: float,
    n_repeats: int,
    lhs_seed: int,
    rng_seed: int,
    param_low: float,
    param_high: float,
) -> Dict[str, Any]:
    d = MockDrawer()
    s = MockSleeper()
    opt = MultiFidelityOptimizer(
        rows=rows,
        theory=theory,
        drawer=d,
        sleeper=s,
        n_repeats=n_repeats,
        lhs_seed=lhs_seed,
        rng_seed=rng_seed,
        param_low=param_low,
        param_high=param_high,
    )
    opt.optimize(budget=budget)
    return {
        "method": "multi_fidelity_rf_ei",
        "best_loss": opt.best_loss,
        "best_params": opt.best_params,
        "wall_time_sec": opt.optimization_wall_time_sec,
        "budget_cost_units": budget,
        "note": "Budget is abstract cost (low/medium/high fidelity); not 1:1 with n_trials evaluations of full STEPS_PER_TRIAL.",
    }


def resolve_theory(target_id: str, target_file: Optional[str]) -> Tuple[dict, str]:
    if target_file:
        theory = proto.load_theory_from_file(target_file)
        label = f"file:{Path(target_file).name}"
    else:
        theory = proto.load_theory_preset(target_id)
        label = f"preset:{target_id}"
    return theory, label


def _cvar(values: List[float], alpha: float) -> float:
    arr = np.asarray([float(x) for x in values if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    q = float(np.quantile(arr, alpha))
    tail = arr[arr >= q]
    if tail.size == 0:
        return q
    return float(np.mean(tail))


def _target_risk_summary(target_block: Dict[str, Any]) -> Dict[str, Any]:
    losses_by_method: Dict[str, List[float]] = {}
    regrets_by_method: Dict[str, List[float]] = {}
    wins_by_method: Dict[str, int] = {}
    top3_by_method: Dict[str, int] = {}
    rank_runs_total = 0

    runs = target_block.get("runs", [])
    for run in runs:
        methods = run.get("methods", {})
        run_losses: Dict[str, float] = {}
        for method_key, block in methods.items():
            if not isinstance(block, dict):
                continue
            method_name = str(block.get("method") or method_key)
            loss = block.get("best_loss")
            if loss is None:
                continue
            lf = float(loss)
            if not np.isfinite(lf):
                continue
            run_losses[method_name] = lf
            losses_by_method.setdefault(method_name, []).append(lf)

        if run_losses:
            run_best = min(run_losses.values())
            for method_name, lf in run_losses.items():
                regrets_by_method.setdefault(method_name, []).append(float(lf - run_best))

            rank_runs_total += 1
            ordered = sorted(run_losses.items(), key=lambda kv: float(kv[1]))
            if ordered:
                best_method = ordered[0][0]
                wins_by_method[best_method] = int(wins_by_method.get(best_method, 0)) + 1
            for idx, (method_name, _loss) in enumerate(ordered, start=1):
                if idx <= 3:
                    top3_by_method[method_name] = int(top3_by_method.get(method_name, 0)) + 1

    rows: List[Dict[str, Any]] = []
    for method_name in sorted(losses_by_method.keys()):
        arr = np.asarray(losses_by_method.get(method_name, []), dtype=np.float64)
        if arr.size == 0:
            continue
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        row = {
            "method": method_name,
            "n_runs": int(arr.size),
            "median_best_loss": float(np.median(arr)),
            "iqr_best_loss": float(q3 - q1),
            "CVaRRegret_0.9": _cvar(regrets_by_method.get(method_name, []), 0.9),
            "CVaRRegret_0.95": _cvar(regrets_by_method.get(method_name, []), 0.95),
            "P_rank_1": (
                float(wins_by_method.get(method_name, 0)) / float(rank_runs_total) if rank_runs_total > 0 else float("nan")
            ),
            "P_rank_le_3": (
                float(top3_by_method.get(method_name, 0)) / float(rank_runs_total) if rank_runs_total > 0 else float("nan")
            ),
        }
        rows.append(row)

    risk_ranked = sorted(
        rows,
        key=lambda r: (
            float(r["CVaRRegret_0.95"]) if np.isfinite(float(r["CVaRRegret_0.95"])) else float("inf"),
            float(r["CVaRRegret_0.9"]) if np.isfinite(float(r["CVaRRegret_0.9"])) else float("inf"),
            float(r["median_best_loss"]),
        ),
    )
    for i, row in enumerate(risk_ranked, 1):
        row["risk_rank"] = i

    return {"per_method": rows, "risk_ranking": risk_ranked, "rank_runs_total": int(rank_runs_total)}


def _method_opts(
    method_options: Dict[str, Dict[str, Any]] | None,
    key: str,
) -> Dict[str, Any]:
    if not method_options:
        return {}
    raw = method_options.get(key) or method_options.get(key.lower())
    return dict(raw) if isinstance(raw, dict) else {}


def run_single_benchmark_method(
    key: str,
    *,
    rows: int,
    theory: dict,
    n_trials: int,
    n_repeats: int,
    lhs_seed: int,
    param_low: float,
    param_high: float,
    mfo_budget: float = 0.0,
    pso_swarm_size: int | None = None,
    method_options: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    k = key.lower()
    opts = _method_opts(method_options, k)
    if k == "tpe":
        return {"tpe": _run_tpe(rows, theory, n_trials, n_repeats, lhs_seed, lhs_seed + 1, param_low, param_high)}
    if k == "skopt":
        return {
            "skopt_gp_ei": _run_skopt(
                rows, theory, n_trials, n_repeats, lhs_seed, lhs_seed + 2, param_low, param_high
            )
        }
    if k == "xnes":
        return {"xnes": _run_xnes(rows, theory, n_trials, n_repeats, lhs_seed + 3, param_low, param_high)}
    if k == "de":
        return {
            "de": _run_de(
                rows,
                theory,
                n_trials,
                n_repeats,
                lhs_seed + 4,
                param_low,
                param_high,
                population_size=int(opts.get("population_size") or 0),
            )
        }
    if k == "gp":
        return {
            "gp": _run_gp(
                rows, theory, n_trials, n_repeats, lhs_seed, lhs_seed + 5, param_low, param_high
            )
        }
    if k == "mfo":
        mfo_opt = float(opts.get("mfo_budget") or 0.0)
        budget = mfo_opt if mfo_opt > 0 else (float(mfo_budget) if mfo_budget > 0 else float(n_trials))
        return {
            "mfo": _run_mfo(
                rows, theory, budget, n_repeats, lhs_seed, lhs_seed + 6, param_low, param_high
            )
        }
    if k == "pso":
        swarm_raw = opts.get("swarm_size")
        swarm_eff = None
        if swarm_raw is not None and int(swarm_raw) > 0:
            swarm_eff = int(swarm_raw)
        elif pso_swarm_size is not None and int(pso_swarm_size) > 0:
            swarm_eff = int(pso_swarm_size)
        return {
            "pso": _run_pso(
                rows,
                theory,
                n_trials,
                n_repeats,
                lhs_seed + 7,
                param_low,
                param_high,
                swarm_size=swarm_eff,
                inertia=opts.get("inertia"),
                cognitive=opts.get("cognitive"),
                social=opts.get("social"),
            )
        }
    if k == "ga":
        pop_raw = opts.get("population_size")
        pop_arg = int(pop_raw) if pop_raw is not None and int(pop_raw) > 0 else None
        tk_raw = opts.get("tournament_k")
        return {
            "ga": _run_ga(
                rows,
                theory,
                n_trials,
                n_repeats,
                lhs_seed + 8,
                param_low,
                param_high,
                population_size=pop_arg,
                tournament_k=int(tk_raw) if tk_raw is not None else None,
            )
        }
    if k == "trust_bo":
        return {
            "trust_bo": _run_trust_bo(
                rows, theory, n_trials, n_repeats, lhs_seed, lhs_seed + 9, param_low, param_high
            )
        }
    if k == "lhs_nm":
        return {
            "lhs_nm": _run_lhs_nm(
                rows, theory, n_trials, n_repeats, lhs_seed, lhs_seed + 10, param_low, param_high
            )
        }
    if k == "rf_ei":
        return {
            "rf_ei": _run_rf_ei(
                rows, theory, n_trials, n_repeats, lhs_seed, lhs_seed + 11, param_low, param_high
            )
        }
    if k == "cem":
        return {"cem": _run_cem(rows, theory, n_trials, n_repeats, lhs_seed + 12, param_low, param_high)}
    if k == "sa":
        return {"sa": _run_sa(rows, theory, n_trials, n_repeats, lhs_seed + 13, param_low, param_high)}
    if k == "cma":
        return {
            "cma": _run_cma(rows, theory, n_trials, n_repeats, lhs_seed, lhs_seed + 14, param_low, param_high)
        }
    if k == "hyperband":
        return {
            "hyperband": _run_hyperband(
                rows, theory, n_trials, n_repeats, lhs_seed, lhs_seed + 15, param_low, param_high
            )
        }
    if k == "turbo_mt":
        return {
            "turbo_mt": _run_turbo_mt(
                rows, theory, n_trials, n_repeats, lhs_seed, lhs_seed + 16, param_low, param_high
            )
        }
    if k == "coord_descent":
        return {
            "coord_descent": _run_coord_descent(
                rows, theory, n_trials, n_repeats, lhs_seed + 17, param_low, param_high
            )
        }
    return {}


def build_benchmark_payload(
    *,
    target_id: str,
    theory: dict,
    method_keys: List[str],
    rows: int,
    n_trials: int,
    n_repeats: int,
    n_runs: int,
    base_seed: int,
    param_low: float,
    param_high: float,
    metric: str,
    mfo_budget: float = 0.0,
    pso_swarm_size: int | None = None,
    method_options: Dict[str, Dict[str, Any]] | None = None,
    source: str = "cli",
) -> Dict[str, Any]:

    from classes.pso_optimization import resolve_swarm_size

    set_active_metric(metric)
    pso_opts = _method_opts(method_options, "pso")
    swarm_from_opts = pso_opts.get("swarm_size")
    swarm_for_meta = (
        int(swarm_from_opts)
        if swarm_from_opts is not None and int(swarm_from_opts) > 0
        else pso_swarm_size
    )
    swarm_meta = resolve_swarm_size(n_trials, swarm_for_meta)
    target_block: Dict[str, Any] = {"id": target_id, "runs": []}
    for r in range(n_runs):
        lhs_seed = base_seed + r * 100_000
        run_record: Dict[str, Any] = {"run_index": r, "lhs_seed": lhs_seed, "methods": {}}
        for m in method_keys:
            run_record["methods"].update(
                run_single_benchmark_method(
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

    meta: Dict[str, Any] = {
        "rows": rows,
        "n_trials": n_trials,
        "n_repeats": n_repeats,
        "n_runs": n_runs,
        "base_seed": base_seed,
        "param_bounds": [param_low, param_high],
        "steps_per_trial": STEPS_PER_TRIAL,
        "timestamp": datetime.datetime.now().isoformat(),
        "metric": metric,
        "pso_swarm_size": int(swarm_for_meta or 0),
        "pso_swarm_size_effective": swarm_meta,
        "source": source,
    }
    if method_options:
        meta["method_options"] = method_options

    payload: Dict[str, Any] = {
        "metadata": meta,
        "targets": [target_block],
    }
    payload["summary"] = {target_id: _target_risk_summary(target_block)}
    return payload


def save_benchmark_payload(payload: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark of optimizers for the inverse problem (no GUI)")
    p.add_argument(
        "--targets",
        nargs="+",
        default=["hist1", "hist2", "hist3"],
        help="Preset target histograms: hist1, hist2, hist3 (see resources/histograms.py)",
    )
    p.add_argument(
        "--target-file",
        type=str,
        default="",
        help="If set, use this file instead of --targets",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["tpe", "skopt", "xnes", "de"],
        choices=list(ALL_METHODS_17),
        help="Optimizers to compare. All 17: " + " ".join(ALL_METHODS_17),
    )
    p.add_argument("--rows", type=int, default=proto.DEFAULT_ROWS)
    p.add_argument("--n-trials", type=int, default=proto.DEFAULT_N_TRIALS)
    p.add_argument("--n-repeats", type=int, default=proto.DEFAULT_N_REPEATS)
    p.add_argument("--n-runs", type=int, default=proto.DEFAULT_N_RUNS)
    p.add_argument("--base-seed", type=int, default=proto.DEFAULT_BASE_SEED)
    p.add_argument("--param-low", type=float, default=proto.PARAM_LOW)
    p.add_argument("--param-high", type=float, default=proto.PARAM_HIGH)
    p.add_argument("--output", type=str, default="", help="Path to the output JSON.")
    p.add_argument(
        "--metric",
        type=str,
        default="weighted_L",
        choices=list(METRIC_CHOICES),
        help="Name of the objective metric for all optimizers in this benchmark.",
    )
    p.add_argument(
        "--mfo-budget",
        type=float,
        default=0.0,
        help="If >0 and mfo is in --methods — MFO cost budget instead of --n-trials",
    )
    p.add_argument(
        "--swarm-size",
        type=int,
        default=0,
        help="PSO swarm size (>0 fixed; 0 = auto min(16, max(6, n_trials//3))). Ignored if pso is not in --methods.",
    )
    args = p.parse_args()

    if args.param_low >= args.param_high:
        print("param-low must be < param-high", file=sys.stderr)
        return 2
    set_active_metric(args.metric)

    target_ids: List[str]
    if args.target_file:
        target_ids = ["__file__"]
    else:
        target_ids = list(args.targets)

    out_path = args.output
    if not out_path:
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_path = f"results/benchmark_inverse-{ts}.json"

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    all_payload: Dict[str, Any] = {
        "metadata": {
            "rows": args.rows,
            "n_trials": args.n_trials,
            "n_repeats": args.n_repeats,
            "n_runs": args.n_runs,
            "base_seed": args.base_seed,
            "param_bounds": [args.param_low, args.param_high],
            "steps_per_trial": STEPS_PER_TRIAL,
            "timestamp": datetime.datetime.now().isoformat(),
            "metric": args.metric,
            "source": "cli",
        },
        "targets": [],
    }

    mfo_budget = float(args.mfo_budget) if args.mfo_budget > 0 else 0.0
    pso_swarm_size = int(args.swarm_size) if int(args.swarm_size) > 0 else None
    for tid in target_ids:
        if tid == "__file__":
            theory, tlabel = resolve_theory("", args.target_file)
        else:
            theory, tlabel = resolve_theory(tid, None)

        block = build_benchmark_payload(
            target_id=tlabel,
            theory=theory,
            method_keys=list(args.methods),
            rows=args.rows,
            n_trials=args.n_trials,
            n_repeats=args.n_repeats,
            n_runs=args.n_runs,
            base_seed=args.base_seed,
            param_low=args.param_low,
            param_high=args.param_high,
            metric=args.metric,
            mfo_budget=mfo_budget,
            pso_swarm_size=pso_swarm_size,
            source="cli",
        )
        block["metadata"]["timestamp"] = all_payload["metadata"]["timestamp"]
        all_payload["targets"].append(block["targets"][0])
        all_payload.setdefault("summary", {}).update(block["summary"])

    if pso_swarm_size is not None:
        from classes.pso_optimization import resolve_swarm_size

        all_payload["metadata"]["pso_swarm_size"] = pso_swarm_size
        all_payload["metadata"]["pso_swarm_size_effective"] = resolve_swarm_size(args.n_trials, pso_swarm_size)

    save_benchmark_payload(all_payload, out_p)
    summary_by_target = all_payload["summary"]

    for tgt_id, block in summary_by_target.items():
        print(f"\n=== Risk ranking for {tgt_id} ===")
        for row in block.get("risk_ranking", []):
            c90 = row.get("CVaRRegret_0.9")
            c95 = row.get("CVaRRegret_0.95")
            med = row.get("median_best_loss")
            p1 = row.get("P_rank_1")
            p3 = row.get("P_rank_le_3")
            print(
                f" {int(row.get('risk_rank', 0)):2d}. {row.get('method', ''):<30} "
                f"CVaRRegret_0.9={c90:.4f}  CVaRRegret_0.95={c95:.4f}  "
                f"P(rank=1)={p1:.3f}  P(rank<=3)={p3:.3f}  median={med:.4f}"
            )

    print(f"Wrote {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())