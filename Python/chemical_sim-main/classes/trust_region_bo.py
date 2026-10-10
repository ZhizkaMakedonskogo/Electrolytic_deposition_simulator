from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

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


class TrustRegionBoOptimization(BaseComparator):
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
        tr_init_frac: float = 0.45,
        tr_min_frac: float = 0.02,
        shrink_factor: float = 0.5,
        patience: int = 3,
        ei_candidates: int = 512,
        ei_lbfgs_restarts: int = 12,
        rng_seed: int = 42,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._lhs_seed = int(lhs_seed)
        self._tr0 = float(tr_init_frac)
        self._tr_min = float(tr_min_frac)
        self._shrink = float(shrink_factor)
        self._patience = max(1, int(patience))
        self._ei_cand = max(32, int(ei_candidates))
        self._ei_restarts = max(4, int(ei_lbfgs_restarts))
        self._rng = np.random.default_rng(rng_seed)

        self._sim = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)

        self.hists: Dict[int, dict] = {}
        self._trials: List[dict] = []
        self.current_trial = 0
        self.total_trials = 0
        self.best_loss = float("inf")
        self.best_params: Optional[Dict[str, float]] = None
        self.optimization_wall_time_sec = 0.0
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()

    @staticmethod
    def hist_compare(theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)

    def _make_gpr(self) -> GaussianProcessRegressor:
        kernel = (
            C(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=6,
            random_state=int(self._rng.integers(1 << 30)),
            alpha=1e-10,
        )

    @staticmethod
    def _finite_loss_array(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        y = np.nan_to_num(y, posinf=1.0e8, neginf=1.0e8, nan=1.0e8)
        y = np.clip(y, 0.0, 1.0e8)
        return y

    def _fit_surrogate(self, x_list: np.ndarray, y_list: np.ndarray) -> Tuple[GaussianProcessRegressor, np.ndarray]:
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        x_list = np.asarray(x_list, dtype=np.float64)
        x_list = np.nan_to_num(x_list, posinf=0.5, neginf=0.5, nan=0.5)
        y_list = self._finite_loss_array(y_list)
        use_x = len(x_list) >= 2
        xn = self._x_scaler.fit_transform(x_list) if use_x else x_list
        if len(y_list) < 2 or float(np.std(y_list)) < 1e-14:
            self._y_scaler.fit(y_list.reshape(-1, 1))
            yn = np.zeros(len(y_list), dtype=float)
        else:
            yn = self._y_scaler.fit_transform(y_list.reshape(-1, 1)).ravel()
        gpr = self._make_gpr()
        gpr.fit(xn, yn)
        return gpr, yn

    def _neg_ei(self, x_raw: np.ndarray, gpr: GaussianProcessRegressor, y_best_scaled: float) -> float:
        x = x_raw.reshape(1, -1)
        xn = self._x_scaler.transform(x)
        mu, sigma = gpr.predict(xn, return_std=True)
        ei = _expected_improvement(mu, sigma, y_best_scaled, xi=0.01)
        return -float(ei[0])

    def _max_ei_in_box(
        self,
        gpr: GaussianProcessRegressor,
        y_best_scaled: float,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Maximize EI inside the axis-aligned box [lo, hi] via random starts + L-BFGS-B."""
        bounds = [(float(lo[d]), float(hi[d])) for d in range(3)]
        best_x = None
        best_ei = -1.0
        for _ in range(self._ei_restarts):
            x0 = np.array([self._rng.uniform(bounds[d][0], bounds[d][1]) for d in range(3)])
            res = minimize(
                self._neg_ei,
                x0,
                args=(gpr, y_best_scaled),
                bounds=bounds,
                method="L-BFGS-B",
            )
            if not res.success:
                continue
            ei = -res.fun
            if ei > best_ei:
                best_ei = ei
                best_x = res.x
        if best_x is None:
            best_x = np.array([self._rng.uniform(bounds[d][0], bounds[d][1]) for d in range(3)])
            best_ei = 0.0
        return np.clip(best_x, lo, hi), float(best_ei)

    def _random_ei_in_box(
        self,
        gpr: GaussianProcessRegressor,
        y_best_scaled: float,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Fallback: dense random sampling of EI in the box (robust if L-BFGS gets stuck)."""
        best_ei = -1.0
        best_x = (lo + hi) / 2.0
        for _ in range(self._ei_cand):
            x = np.array([self._rng.uniform(lo[d], hi[d]) for d in range(3)])
            v = -self._neg_ei(x, gpr, y_best_scaled)
            if v > best_ei:
                best_ei = v
                best_x = x
        return best_x, float(best_ei)

    def optimize(self, n_trials: int = 50) -> None:
        if n_trials <= 0:
            raise ValueError("n_trials must be > 0")

        self.total_trials = int(n_trials)
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.hists = {}
        self._trials = []

        n_init = default_n_init(n_trials)
        n_init = min(n_init, n_trials)

        x_accum: List[List[float]] = []
        y_accum: List[float] = []

        t0 = time.perf_counter()
        eval_counter = 0

        lhs = latin_hypercube_initial(n_init, self._lhs_seed, self._low, self._high)
        for i in range(len(lhs)):
            if eval_counter >= self.total_trials or self.sleeper.is_exit:
                break
            add, tr, mg = float(lhs[i, 0]), float(lhs[i, 1]), float(lhs[i, 2])
            eval_counter += 1
            self.current_trial = eval_counter
            loss, hist = self._sim.evaluate(add, tr, mg, eval_index=eval_counter, n_repeats=self._n_repeats)
            self.hists[eval_counter - 1] = hist
            x_accum.append([add, tr, mg])
            y_accum.append(loss)
            self._record_trial(eval_counter - 1, loss, add, tr, mg, hist)

        if eval_counter >= self.total_trials:
            self.optimization_wall_time_sec = time.perf_counter() - t0
            return

        x_arr = np.asarray(x_accum, dtype=float)
        y_arr = np.asarray(y_accum, dtype=float)

        tr_len = self._tr0 * (self._high - self._low)
        stall = 0
        best_y_seen = float(np.min(y_arr))

        while eval_counter < self.total_trials and not self.sleeper.is_exit:
            idx_best = int(np.argmin(y_arr))
            center = x_arr[idx_best].copy()

            lo = np.clip(center - 0.5 * tr_len, self._low, self._high)
            hi = np.clip(center + 0.5 * tr_len, self._low, self._high)
            for d in range(3):
                if hi[d] <= lo[d] + 1e-9:
                    hi[d] = min(self._high, lo[d] + 1e-6)

            gpr, yn = self._fit_surrogate(x_arr, y_arr)
            y_best_scaled = float(np.min(yn))

            x_next, ei_val = self._max_ei_in_box(gpr, y_best_scaled, lo, hi)
            if ei_val < 1e-12:
                x_next, _ = self._random_ei_in_box(gpr, y_best_scaled, lo, hi)

            add, tr, mg = float(x_next[0]), float(x_next[1]), float(x_next[2])
            eval_counter += 1
            self.current_trial = eval_counter
            loss, hist = self._sim.evaluate(add, tr, mg, eval_index=eval_counter, n_repeats=self._n_repeats)
            self.hists[eval_counter - 1] = hist

            x_arr = np.vstack([x_arr, x_next.reshape(1, -1)])
            y_arr = np.append(y_arr, loss)
            self._record_trial(eval_counter - 1, loss, add, tr, mg, hist)

            if loss + 1e-9 < best_y_seen:
                best_y_seen = float(loss)
                stall = 0
            else:
                stall += 1
                if stall >= self._patience:
                    tr_len = max(self._tr_min * (self._high - self._low), tr_len * self._shrink)
                    stall = 0

        self.optimization_wall_time_sec = time.perf_counter() - t0
        print(f"Trust-region BO finished. best_loss={self.best_loss:.4f} evals={eval_counter}")

    def _record_trial(self, num: int, loss: float, add: float, tr: float, mg: float, hist: dict) -> None:
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