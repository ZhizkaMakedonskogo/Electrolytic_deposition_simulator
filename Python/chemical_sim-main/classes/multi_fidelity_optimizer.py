"""
Multi-fidelity optimization: Random Forest over (add, transit, merge, one-hot fidelity),
Expected Improvement per unit cost.
"""
from __future__ import annotations

import datetime
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor

from classes.base_comparator import BaseComparator
from classes.board import Board
from classes.simulation_objective import (
    DEFAULT_N_REPEATS,
    PARAM_HIGH,
    PARAM_LOW,
    STEPS_PER_TRIAL,
    SimulationObjective,
    latin_hypercube_initial,
    weighted_hist_compare,
)
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper
from resources.constants import Mode
from resources.utilities import optimization_result_file_path


@dataclass(frozen=True)
class FidelitySpec:
    name: str
    steps: int
    cost: float
    code: int


FIDELITY_LEVELS: Tuple[FidelitySpec, ...] = (
    FidelitySpec("low", 40, 1.0, 0),
    FidelitySpec("medium", 175, 4.0, 1),
    FidelitySpec("high", 500, 10.0, 2),
)

# UI labels 
_R_TRIAL = "Trial"
_R_BUDGET = "budget"
_R_LOSS = "Loss"
_R_BEST = "best"
_R_FID = "fidelity"


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    imp = y_best - mu - xi
    z = imp / sigma
    ei = imp * scipy_stats.norm.cdf(z) + sigma * scipy_stats.norm.pdf(z)
    return np.maximum(ei, 0.0)


def _fidelity_one_hot(code: int) -> np.ndarray:
    v = np.zeros(3, dtype=float)
    v[code] = 1.0
    return v


def _make_features(add: float, transit: float, merge: float, code: int) -> np.ndarray:
    return np.concatenate([[add, transit, merge], _fidelity_one_hot(code)])


def rf_predict_mean_std(rf: RandomForestRegressor, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim == 1:
        X = X.reshape(1, -1)
    preds = np.array([tree.predict(X) for tree in rf.estimators_])
    return preds.mean(axis=0).ravel(), preds.std(axis=0).ravel()


@dataclass
class MFRecord:
    add: float
    transit: float
    merge: float
    fidelity_code: int
    fidelity_name: str
    steps: int
    eps: float
    hist: Dict[Any, int]
    timestamp: float
    cost: float


def correlation_between_fidelities(
    rows: int,
    theory: dict,
    sleeper: Sleeper,
    param_sets: List[Tuple[float, float, float]],
    *,
    n_repeats: int = DEFAULT_N_REPEATS,
) -> Dict[str, float]:
    """Pearson correlation r between the mean eps at the lowest and highest fidelity for fixed parameter sets"""
    obj = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)
    low: List[float] = []
    high: List[float] = []
    lo = FIDELITY_LEVELS[0]
    hi = FIDELITY_LEVELS[-1]
    for i, (a, t, m) in enumerate(param_sets):
        if sleeper.is_exit:
            break
        e_l, _ = obj.evaluate(a, t, m, eval_index=i, n_repeats=n_repeats, steps=lo.steps)
        e_h, _ = obj.evaluate(a, t, m, eval_index=1000 + i, n_repeats=n_repeats, steps=hi.steps)
        low.append(e_l)
        high.append(e_h)
    if len(low) < 2:
        return {"n": float(len(low)), "pearson_r": float("nan")}
    r = np.corrcoef(np.asarray(low), np.asarray(high))[0, 1]
    return {"n": float(len(low)), "pearson_r": float(r)}


class MultiFidelityOptimizer(BaseComparator):
    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        n_repeats: int = DEFAULT_N_REPEATS,
        lhs_seed: int = 42,
        param_low: float = PARAM_LOW,
        param_high: float = PARAM_HIGH,
        acquisition_candidates: int = 384,
        rng_seed: int = 42,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n_repeats = n_repeats
        self._lhs_seed = lhs_seed
        self._param_low = param_low
        self._param_high = param_high
        self._acquisition_candidates = acquisition_candidates
        self._rng = np.random.default_rng(rng_seed)

        self._sim = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)

        self.records: List[MFRecord] = []
        self.hists: Dict[int, dict] = {}
        self.eval_index = 0
        self.current_trial = 0
        self.total_trials = 0

        self.budget_total = 0.0
        self.budget_spent = 0.0
        self.last_fidelity_label = ""

        self.best_loss = float("inf")
        self.best_params: Optional[Dict[str, float]] = None
        self.optimization_wall_time_sec = 0.0

    def hist_compare(self, theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)

    def _update_best(self) -> None:
        if not self.records:
            return
        best_r = min(self.records, key=lambda r: (r.eps, -r.fidelity_code))
        self.best_loss = best_r.eps
        self.best_params = {"add": best_r.add, "transit": best_r.transit, "merge": best_r.merge}
        if best_r.hist:
            self._best_experimental_bar = dict(best_r.hist)

    def _schedule_ui(
        self,
        trial_num: int,
        loss: float,
        add: float,
        transit: float,
        merge: float,
        fidelity_name: str,
    ) -> None:
        def _upd() -> None:
            try:
                if not hasattr(self.drawer, "window") or self.drawer.window is None:
                    return
                try:
                    if not self.drawer.window.winfo_exists():
                        return
                except Exception:
                    return
                if hasattr(self.drawer, "no_ui") and self.drawer.no_ui.get():
                    return
                self.drawer.stat_1.config(
                    text=f"{_R_TRIAL}: {trial_num} | {_R_BUDGET}: {self.budget_spent:.1f}/{self.budget_total:.1f}"
                )
                self.drawer.stat_2.config(
                    text=f"{_R_LOSS}: {loss:.4f} ({_R_BEST}: {self.best_loss:.4f}) | {_R_FID}: {fidelity_name}"
                )
                self.drawer.stat_3.config(text=f"add={add:.3f} | trans={transit:.3f} | merge={merge:.3f}")
                self.drawer.window.update_idletasks()
            except Exception as e:
                print(f"MFO UI update error: {e}")

        try:
            self.drawer.window.after(0, _upd)
        except Exception:
            _upd()

    def _run_one(
        self,
        add: float,
        transit: float,
        merge: float,
        spec: FidelitySpec,
    ) -> Tuple[float, Dict[Any, int]]:
        loss, hist = self._sim.evaluate(
            add,
            transit,
            merge,
            eval_index=self.eval_index,
            n_repeats=self._n_repeats,
            steps=spec.steps,
        )
        self.eval_index += 1
        now = time.time()
        rec = MFRecord(
            add=float(add),
            transit=float(transit),
            merge=float(merge),
            fidelity_code=spec.code,
            fidelity_name=spec.name,
            steps=spec.steps,
            eps=float(loss),
            hist=hist,
            timestamp=now,
            cost=spec.cost,
        )
        self.records.append(rec)
        self.hists[len(self.records) - 1] = hist
        self.budget_spent += spec.cost
        self.current_trial += 1
        self.last_fidelity_label = spec.name

        self._update_best()
        self._schedule_ui(self.current_trial, loss, add, transit, merge, spec.name)
        return loss, hist

    def _fit_rf(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        rf = RandomForestRegressor(
            n_estimators=128,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        from classes.simulation_objective import numpy_losses_for_surrogate_fit

        rf.fit(X, numpy_losses_for_surrogate_fit(y))
        return rf

    def _best_acquisition(
        self,
        rf: RandomForestRegressor,
        y_best: float,
        budget_remaining: float,
    ) -> Optional[Tuple[float, float, float, int]]:
        best_score = -1.0
        best: Optional[Tuple[float, float, float, int]] = None
        lo, hi = self._param_low, self._param_high

        for _ in range(self._acquisition_candidates):
            add = float(self._rng.uniform(lo, hi))
            transit = float(self._rng.uniform(lo, hi))
            merge = float(self._rng.uniform(lo, hi))
            for spec in FIDELITY_LEVELS:
                if spec.cost > budget_remaining + 1e-9:
                    continue
                x = _make_features(add, transit, merge, spec.code).reshape(1, -1)
                mu, sigma = rf_predict_mean_std(rf, x)
                ei = float(_expected_improvement(mu, sigma, y_best, xi=0.01)[0])
                score = ei / spec.cost if spec.cost > 0 else ei
                if score > best_score:
                    best_score = score
                    best = (add, transit, merge, spec.code)

        if best is not None and len(self.records) >= 8:
            a0, t0, m0, c0 = best

            def key(a: float, t: float, m: float) -> Tuple[float, float, float]:
                return (round(a, 3), round(t, 3), round(m, 3))

            k0 = key(a0, t0, m0)
            lows = [r for r in self.records if r.fidelity_code == 0 and key(r.add, r.transit, r.merge) == k0]
            if len(lows) >= 2:
                mean_low = float(np.mean([r.eps for r in lows]))
                thr = float(np.percentile([r.eps for r in self.records if r.fidelity_code == 0], 25))
                if mean_low <= thr and budget_remaining >= FIDELITY_LEVELS[1].cost:
                    for spec in (FIDELITY_LEVELS[1], FIDELITY_LEVELS[2]):
                        if spec.cost <= budget_remaining + 1e-9 and spec.code > c0:
                            best = (a0, t0, m0, spec.code)
                            break

        return best

    def optimize(self, budget: float = 100.0) -> None:
        if budget < 5.0:
            raise ValueError("budget must be >= 5 cost units")

        t0 = time.perf_counter()
        self.records = []
        self.hists = {}
        self.eval_index = 0
        self.current_trial = 0
        self.budget_total = float(budget)
        self.budget_spent = 0.0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.total_trials = int(max(1, round(budget)))

        min_cost = min(f.cost for f in FIDELITY_LEVELS)
        low_spec = FIDELITY_LEVELS[0]

        n_init = min(10, max(3, int(self.budget_total * 0.12)))
        n_init = min(n_init, int(self.budget_total // low_spec.cost))
        n_init = max(3, n_init)

        lhs = latin_hypercube_initial(n_init, self._lhs_seed, self._param_low, self._param_high)
        print(f"MFO: budget={budget:.1f}, initial LHS (low): {n_init} points")
        print("-" * 50)

        for i in range(len(lhs)):
            if self.sleeper.is_exit:
                break
            if self.budget_spent + low_spec.cost > self.budget_total + 1e-9:
                break
            a, t, m = float(lhs[i][0]), float(lhs[i][1]), float(lhs[i][2])
            self._run_one(a, t, m, low_spec)

        while self.budget_total - self.budget_spent >= min_cost - 1e-9 and not self.sleeper.is_exit:
            budget_remaining = self.budget_total - self.budget_spent
            if len(self.records) < 4:
                a = float(self._rng.uniform(self._param_low, self._param_high))
                t = float(self._rng.uniform(self._param_low, self._param_high))
                m = float(self._rng.uniform(self._param_low, self._param_high))
                self._run_one(a, t, m, low_spec)
                continue

            X = np.array([_make_features(r.add, r.transit, r.merge, r.fidelity_code) for r in self.records])
            y = np.array([r.eps for r in self.records], dtype=float)
            y_best = float(np.min(y))

            rf = self._fit_rf(X, y)
            choice = self._best_acquisition(rf, y_best, budget_remaining)
            if choice is None:
                a = float(self._rng.uniform(self._param_low, self._param_high))
                t = float(self._rng.uniform(self._param_low, self._param_high))
                m = float(self._rng.uniform(self._param_low, self._param_high))
                spec = low_spec
                if spec.cost <= budget_remaining + 1e-9:
                    self._run_one(a, t, m, spec)
                else:
                    break
                continue

            a, t, m, code = choice
            spec = FIDELITY_LEVELS[code]
            if self.budget_spent + spec.cost > self.budget_total + 1e-9:
                break
            self._run_one(a, t, m, spec)

        # The same run length and the same loss as in the batch comparison (TPE, PSO, …): not a low-fi bar.
        if self.best_params and not self.sleeper.is_exit:
            bp = self.best_params
            loss_std, hist_std = self._sim.evaluate(
                float(bp["add"]),
                float(bp["transit"]),
                float(bp["merge"]),
                eval_index=self.eval_index,
                n_repeats=DEFAULT_N_REPEATS,
                steps=STEPS_PER_TRIAL,
            )
            self.eval_index += 1
            self.best_loss = float(loss_std)
            if hist_std:
                self._best_experimental_bar = dict(hist_std)

        self.optimization_wall_time_sec = time.perf_counter() - t0
        print("\nMFO finished.")
        if self.best_params:
            print(
                f"Best params: add={self.best_params['add']:.4f}, "
                f"transit={self.best_params['transit']:.4f}, merge={self.best_params['merge']:.4f}"
            )
            print(f"L after the standard evaluation ({STEPS_PER_TRIAL} steps): {self.best_loss:.4f}")
        print(f"Budget spent: {self.budget_spent:.1f} / {self.budget_total:.1f}")
        print(f"Wall time: {self.optimization_wall_time_sec:.2f} s")

    def result(self):
        _time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_path = optimization_result_file_path("MultiFidelityOptimization", _time)

        trials_out: List[Dict[str, Any]] = []
        for i, r in enumerate(self.records):
            trials_out.append(
                {
                    "index": i,
                    "params": {"add": r.add, "transit": r.transit, "merge": r.merge},
                    "fidelity": r.fidelity_name,
                    "steps": r.steps,
                    "cost": r.cost,
                    "value": r.eps,
                    "timestamp": r.timestamp,
                    "hist": r.hist,
                }
            )

        try:
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "metadata": {
                            "optimizer": "Multi-Fidelity (RF + EI/cost)",
                            "budget_total": self.budget_total,
                            "budget_spent": self.budget_spent,
                            "best_loss_observed": self.best_loss,
                            "best_params": self.best_params,
                            "wall_time_sec": self.optimization_wall_time_sec,
                            "n_repeats": self._n_repeats,
                            "fidelity_levels": [
                                {"name": f.name, "steps": f.steps, "cost": f.cost} for f in FIDELITY_LEVELS
                            ],
                            "timestamp": _time,
                        },
                        "trials": trials_out,
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"\nMFO results saved to {result_path}")
        except Exception as e:
            print(f"Save error: {e}")
            result_path = None

        if self.best_params and not self.sleeper.is_exit:
            print("\nFinal validation (500 steps, high fidelity)...")
            final_board = Board(
                rows=self.rows,
                addprob=self.best_params["add"],
                transitprob=self.best_params["transit"],
                mergeprob=self.best_params["merge"],
                current_mode=Mode.CONST,
            )
            for step in range(500):
                final_board.run()
                if step % 50 == 0:
                    try:
                        self.drawer.prepare_draw()
                        final_board.draw(self.drawer)
                        self.drawer.complete_draw()
                        self.drawer.window.update()
                    except Exception as e:
                        print(f"Draw error: {e}")
                    time.sleep(0.05)

            final_bar = final_board.create_bar()
            final_eps = self.hist_compare(self.theory, final_bar)
            print(f"Final loss (500 steps): {final_eps:.4f}")
            try:
                self.drawer.draw_bar(final_bar)
                self.drawer.window.update()
            except Exception as e:
                print(f"Histogram error: {e}")

            print("Pause 3s...")
            time.sleep(3.0)

        return {
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "result_file": result_path,
            "wall_time_sec": self.optimization_wall_time_sec,
            "budget_spent": self.budget_spent,
            "budget_total": self.budget_total,
        }