"""
M trust regions in [0,1]^3, one GP on the full archive, EI over a regional box,
candidate selection with the largest EI; region length adapts based on success/failure streaks.

"""
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
    numpy_losses_for_surrogate_fit,
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


def _pick_m_centers(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    m: int,
    rng: np.random.Generator,
    low: float,
    high: float,
    min_dist: float = 0.05,
) -> np.ndarray:
    """First the best y, pairwise distance > min_dist; if not enough — random points."""
    order = np.argsort(y_arr)
    picked: List[int] = []
    for idx in order:
        x = x_arr[idx]
        ok = True
        for j in picked:
            if float(np.linalg.norm(x - x_arr[j])) < min_dist:
                ok = False
                break
        if ok:
            picked.append(int(idx))
        if len(picked) >= m:
            break
    centers = [x_arr[i].copy() for i in picked]
    while len(centers) < m:
        centers.append(rng.uniform(low, high, size=3))
    return np.stack(centers, axis=0)


class TurboMtOptimization(BaseComparator):
    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        n_trusts: int = 3,
        n_repeats: int = 3,
        param_low: float = 0.0,
        param_high: float = 1.0,
        lhs_seed: int = 42,
        tr_init_frac: float = 0.45,
        tr_min_frac: float = 0.02,
        shrink_factor: float = 0.5,
        success_patience: int = 3,
        failure_patience: int = 3,
        ei_restarts: int = 10,
        rng_seed: int = 42,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._m = max(2, min(8, int(n_trusts)))
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._lhs_seed = int(lhs_seed)
        self._tr0 = float(tr_init_frac)
        self._tr_min = float(tr_min_frac)
        self._shrink = float(shrink_factor)
        self._succ_p = max(1, int(success_patience))
        self._fail_p = max(1, int(failure_patience))
        self._ei_restarts = max(4, int(ei_restarts))
        self._rng = np.random.default_rng(rng_seed)
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

    def _make_gpr(self) -> GaussianProcessRegressor:
        kernel = (
            C(1.0, (1e-2, 1e2))
            * Matern(length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=4,
            random_state=int(self._rng.integers(1 << 30)),
            alpha=1e-10,
        )

    def _fit_surrogate(
        self, x_list: np.ndarray, y_list: np.ndarray
    ) -> Tuple[GaussianProcessRegressor, StandardScaler, StandardScaler, np.ndarray]:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_list = np.asarray(x_list, dtype=np.float64)
        x_list = np.nan_to_num(x_list, posinf=0.5, neginf=0.5, nan=0.5)
        y_list = numpy_losses_for_surrogate_fit(y_list)
        use_x = len(x_list) >= 2
        xn = x_scaler.fit_transform(x_list) if use_x else x_list
        if len(y_list) < 2 or float(np.std(y_list)) < 1e-14:
            y_scaler.fit(y_list.reshape(-1, 1))
            yn = np.zeros(len(y_list), dtype=float)
        else:
            yn = y_scaler.fit_transform(y_list.reshape(-1, 1)).ravel()
        gpr = self._make_gpr()
        gpr.fit(xn, yn)
        return gpr, x_scaler, y_scaler, yn

    def _neg_ei_at(
        self,
        x_raw: np.ndarray,
        gpr: GaussianProcessRegressor,
        x_scaler: StandardScaler,
        y_best_scaled: float,
    ) -> float:
        xn = x_scaler.transform(x_raw.reshape(1, -1))
        mu, sigma = gpr.predict(xn, return_std=True)
        ei = _expected_improvement(mu, sigma, y_best_scaled, xi=0.01)
        return -float(ei[0])

    def _max_ei_in_box(
        self,
        gpr: GaussianProcessRegressor,
        x_scaler: StandardScaler,
        y_best_scaled: float,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        bounds = [(float(lo[d]), float(hi[d])) for d in range(3)]
        best_x = (lo + hi) / 2.0
        best_ei = -1.0
        for _ in range(self._ei_restarts):
            x0 = np.array([self._rng.uniform(bounds[d][0], bounds[d][1]) for d in range(3)])
            res = minimize(
                self._neg_ei_at,
                x0,
                args=(gpr, x_scaler, y_best_scaled),
                bounds=bounds,
                method="L-BFGS-B",
            )
            if not res.success:
                continue
            ei = -res.fun
            if ei > best_ei:
                best_ei = ei
                best_x = res.x
        if best_ei < 0:
            best_x = np.array([self._rng.uniform(bounds[d][0], bounds[d][1]) for d in range(3)])
            best_ei = 0.0
        return np.clip(best_x, lo, hi), float(best_ei)

    def _record(self, num: int, loss: float, add: float, tr: float, mg: float, hist: dict) -> None:
        self._trials.append(
            {"number": num, "value": loss, "params": {"add": add, "transit": tr, "merge": mg}, "hist": hist}
        )
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_params = {"add": add, "transit": tr, "merge": mg}
            if hist:
                self._best_experimental_bar = dict(hist)

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

        span = self._high - self._low
        lengths = np.full(self._m, self._tr0 * span, dtype=float)
        succ = np.zeros(self._m, dtype=int)
        fail = np.zeros(self._m, dtype=int)

        t0 = time.perf_counter()
        eval_counter = 0

        x_list: List[List[float]] = []
        y_list: List[float] = []

        lhs = latin_hypercube_initial(n_init, self._lhs_seed, self._low, self._high)
        for i in range(len(lhs)):
            if eval_counter >= self.total_trials or self.sleeper.is_exit:
                break
            add, tr, mg = float(lhs[i, 0]), float(lhs[i, 1]), float(lhs[i, 2])
            eval_counter += 1
            self.current_trial = eval_counter
            loss, hist = self._sim.evaluate(add, tr, mg, eval_index=eval_counter, n_repeats=self._n_repeats)
            self.hists[eval_counter - 1] = hist
            x_list.append([add, tr, mg])
            y_list.append(loss)
            self._record(eval_counter - 1, loss, add, tr, mg, hist)

        if eval_counter >= self.total_trials or len(x_list) < 4:
            self.optimization_wall_time_sec = time.perf_counter() - t0
            print(f"TuRBO-M finished early. best_loss={self.best_loss:.4f} evals={eval_counter}")
            return

        x_arr = np.asarray(x_list, dtype=float)
        y_arr = np.asarray(y_list, dtype=float)

        print(f"TuRBO-M: M={self._m} trust regions, GP+EI, after LHS={len(x_list)} points")
        print("-" * 50)

        while eval_counter < self.total_trials and not self.sleeper.is_exit:
            prev_best = float(np.min(y_arr))
            centers = _pick_m_centers(x_arr, y_arr, self._m, self._rng, self._low, self._high)
            gpr, x_scaler, _y_scaler, yn = self._fit_surrogate(x_arr, y_arr)
            y_best_scaled = float(np.min(yn))

            best_j = 0
            best_ei = -1.0
            best_x = centers[0].copy()
            for j in range(self._m):
                c = centers[j]
                half = 0.5 * lengths[j]
                lo = np.clip(c - half, self._low, self._high)
                hi = np.clip(c + half, self._low, self._high)
                for d in range(3):
                    if hi[d] <= lo[d] + 1e-9:
                        hi[d] = min(self._high, lo[d] + 1e-6)
                x_cand, ei_val = self._max_ei_in_box(gpr, x_scaler, y_best_scaled, lo, hi)
                if ei_val > best_ei:
                    best_ei = ei_val
                    best_x = x_cand
                    best_j = j

            add, tr, mg = float(best_x[0]), float(best_x[1]), float(best_x[2])
            eval_counter += 1
            self.current_trial = eval_counter
            loss, hist = self._sim.evaluate(add, tr, mg, eval_index=eval_counter, n_repeats=self._n_repeats)
            self.hists[eval_counter - 1] = hist
            x_arr = np.vstack([x_arr, best_x.reshape(1, -1)])
            y_arr = np.append(y_arr, loss)
            self._record(eval_counter - 1, loss, add, tr, mg, hist)

            if loss + 1e-9 < prev_best:
                succ[best_j] += 1
                fail[best_j] = 0
                if succ[best_j] >= self._succ_p:
                    lengths[best_j] = min(0.9 * span, float(lengths[best_j] * 1.25))
                    succ[best_j] = 0
            else:
                fail[best_j] += 1
                succ[best_j] = 0
                if fail[best_j] >= self._fail_p:
                    lengths[best_j] = max(self._tr_min * span, float(lengths[best_j] * self._shrink))
                    fail[best_j] = 0

        self.optimization_wall_time_sec = time.perf_counter() - t0
        print(f"TuRBO-M finished. best_loss={self.best_loss:.4f} evals={eval_counter}")