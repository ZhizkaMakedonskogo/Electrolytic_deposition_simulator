"""
Random Forest surrogate + Expected Improvement (single fidelity)

Tree ensembles capture non-smooth structure without assuming GP smoothness;
uncertainty comes from the disagreement between trees (the same idea as in multi-fidelity RF,
but only the features (add, transit, merge) and the full SimulationObjective budget per trial).

Novelty vs GP-based BO: a different inductive bias for noisy black-box loss on [0,1]^3.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestRegressor

from classes.base_comparator import BaseComparator
from classes.simulation_objective import (
    SimulationObjective,
    default_n_init,
    latin_hypercube_initial,
    weighted_hist_compare,
)
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    imp = y_best - mu - xi
    z = imp / sigma
    ei = imp * scipy_stats.norm.cdf(z) + sigma * scipy_stats.norm.pdf(z)
    return np.maximum(ei, 0.0)


def _rf_mean_std(rf: RandomForestRegressor, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim == 1:
        X = X.reshape(1, -1)
    preds = np.array([tree.predict(X) for tree in rf.estimators_])
    return preds.mean(axis=0).ravel(), preds.std(axis=0).ravel()


class RfEiOptimization(BaseComparator):
    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        n_repeats: int = 3,
        param_low: float = 0.0,
        param_high: float = 1.0,
        lhs_seed: int = 42,
        rng_seed: int = 42,
        acquisition_candidates: int = 512,
        n_estimators: int = 128,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._lhs_seed = int(lhs_seed)
        self._rng = np.random.default_rng(rng_seed)
        self._n_cand = max(64, int(acquisition_candidates))
        self._n_est = max(32, int(n_estimators))

        self._sim = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)

        self.hists: Dict[int, dict] = {}
        self._trials: List[dict] = []
        self.current_trial = 0
        self.total_trials = 0
        self.best_loss = float("inf")
        self.best_params: Optional[Dict[str, float]] = None
        self.optimization_wall_time_sec = 0.0

    @staticmethod
    def hist_compare(theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)

    def _fit_rf(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        rf = RandomForestRegressor(
            n_estimators=self._n_est,
            max_depth=18,
            min_samples_leaf=2,
            random_state=int(self._rng.integers(1 << 30)),
            n_jobs=-1,
        )
        from classes.simulation_objective import numpy_losses_for_surrogate_fit

        rf.fit(X, numpy_losses_for_surrogate_fit(y))
        return rf

    def _record(self, num: int, loss: float, add: float, tr: float, mg: float, hist: dict) -> None:
        self._trials.append(
            {
                "number": num,
                "value": loss,
                "params": {"add": add, "transit": tr, "merge": mg},
                "hist": hist,
            }
        )
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_params = {"add": add, "transit": tr, "merge": mg}
            if hist:
                self._best_experimental_bar = dict(hist)

    def optimize(self, n_trials: int = 50) -> None:
        if n_trials < 4:
            raise ValueError("n_trials must be >= 4")

        self.total_trials = int(n_trials)
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.hists = {}
        self._trials = []

        n_init = default_n_init(n_trials)
        n_init = min(n_init, n_trials)

        lhs = latin_hypercube_initial(n_init, self._lhs_seed, self._low, self._high)
        xs: List[List[float]] = []
        ys: List[float] = []

        t0 = time.perf_counter()
        eval_idx = 0

        for i in range(len(lhs)):
            if eval_idx >= self.total_trials or self.sleeper.is_exit:
                break
            a, t, m = float(lhs[i, 0]), float(lhs[i, 1]), float(lhs[i, 2])
            eval_idx += 1
            self.current_trial = eval_idx
            loss, hist = self._sim.evaluate(a, t, m, eval_index=eval_idx, n_repeats=self._n_repeats)
            self.hists[eval_idx - 1] = hist
            xs.append([a, t, m])
            ys.append(loss)
            self._record(eval_idx - 1, loss, a, t, m, hist)

        X = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)

        while eval_idx < self.total_trials and not self.sleeper.is_exit:
            rf = self._fit_rf(X, y)
            y_best = float(np.min(y))

            best_ei = -1.0
            best_x = None
            for _ in range(self._n_cand):
                cand = np.array(
                    [self._rng.uniform(self._low, self._high) for _ in range(3)],
                    dtype=float,
                ).reshape(1, 3)
                mu, sig = _rf_mean_std(rf, cand)
                ei = float(_expected_improvement(mu, sig, y_best, xi=0.01)[0])
                if ei > best_ei:
                    best_ei = ei
                    best_x = cand.ravel()

            if best_x is None:
                best_x = np.array([self._rng.uniform(self._low, self._high) for _ in range(3)])

            a, t, m = float(best_x[0]), float(best_x[1]), float(best_x[2])
            eval_idx += 1
            self.current_trial = eval_idx
            loss, hist = self._sim.evaluate(a, t, m, eval_index=eval_idx, n_repeats=self._n_repeats)
            self.hists[eval_idx - 1] = hist
            X = np.vstack([X, np.array([[a, t, m]], dtype=float)])
            y = np.append(y, loss)
            self._record(eval_idx - 1, loss, a, t, m, hist)

        self.optimization_wall_time_sec = time.perf_counter() - t0
        print(f"RF+EI finished. best_loss={self.best_loss:.4f} evals={eval_idx}")