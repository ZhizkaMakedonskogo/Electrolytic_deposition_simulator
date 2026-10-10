"""
Latin Hypercube + multi-start Nelder-Mead in unconstrained logit space
theta = low + (high-low) * sigmoid(u), u in R^3. space-filling,
then a local derivative-free search with a strict evaluation budget
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from classes.base_comparator import BaseComparator
from classes.simulation_objective import SimulationObjective, latin_hypercube_initial, weighted_hist_compare
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper


def _u_from_theta(theta: float, low: float, high: float, eps: float = 1e-6) -> float:
    t = (theta - low) / (high - low)
    t = float(np.clip(t, eps, 1.0 - eps))
    return float(np.log(t / (1.0 - t)))


class LhsNelderMeadOptimization(BaseComparator):
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
        n_starts: int = 4,
        rng_seed: int = 42,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._lhs_seed = int(lhs_seed)
        self._n_starts = max(1, int(n_starts))
        self._rng = np.random.default_rng(rng_seed)

        self._sim = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)

        self.trials_log: List[dict] = []
        self.current_trial = 0
        self.total_trials = 0
        self.best_loss = float("inf")
        self.best_params: Optional[Dict[str, float]] = None
        self.optimization_wall_time_sec = 0.0
        self._eval_counter = 0

    @staticmethod
    def hist_compare(theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)

    def _theta_from_u(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), -40.0, 40.0)
        s = 1.0 / (1.0 + np.exp(-u))
        return self._low + (self._high - self._low) * s

    def _eval_u(self, u: np.ndarray) -> float:
        if self._eval_counter >= self.total_trials:
            return 1e100
        t = self._theta_from_u(u)
        self._eval_counter += 1
        self.current_trial = self._eval_counter
        loss, hist = self._sim.evaluate(
            float(t[0]),
            float(t[1]),
            float(t[2]),
            eval_index=self._eval_counter,
            n_repeats=self._n_repeats,
        )
        self.trials_log.append(
            {
                "number": len(self.trials_log),
                "value": float(loss),
                "params": {"add": float(t[0]), "transit": float(t[1]), "merge": float(t[2])},
                "hist": hist,
            }
        )
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_params = {"add": float(t[0]), "transit": float(t[1]), "merge": float(t[2])}
            if hist:
                self._best_experimental_bar = dict(hist)
        return float(loss)

    def optimize(self, n_trials: int = 50) -> None:
        if n_trials < 8:
            raise ValueError("n_trials must be >= 8 for LHS + Nelder-Mead")

        self.total_trials = int(n_trials)
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.trials_log = []
        self._eval_counter = 0

        t0 = time.perf_counter()

        n_lhs = max(4, min(self.total_trials // 3, 14))
        n_lhs = min(n_lhs, self.total_trials - 3 * self._n_starts)

        lhs = latin_hypercube_initial(n_lhs, self._lhs_seed, self._low, self._high)
        scored: List[Tuple[float, np.ndarray]] = []

        for i in range(len(lhs)):
            if self._eval_counter >= self.total_trials or self.sleeper.is_exit:
                break
            th = lhs[i]
            u0 = np.array([_u_from_theta(float(th[d]), self._low, self._high) for d in range(3)])
            loss = self._eval_u(u0)
            scored.append((loss, u0.copy()))

        scored.sort(key=lambda z: z[0])
        starts = [u for _, u in scored[: self._n_starts]]
        if not starts:
            self.optimization_wall_time_sec = time.perf_counter() - t0
            return

        n_rest = self.total_trials - self._eval_counter
        per_nm = max(3, n_rest // len(starts))

        for u0 in starts:
            if self._eval_counter >= self.total_trials or self.sleeper.is_exit:
                break
            rem = self.total_trials - self._eval_counter
            if rem < 3:
                break
            maxfev = min(per_nm, rem)

            def fun(uu: np.ndarray) -> float:
                return self._eval_u(np.asarray(uu, dtype=float))

            u0s = np.asarray(u0, dtype=float).ravel()
            minimize(
                fun,
                u0s,
                method="Nelder-Mead",
                options={"maxfev": int(maxfev), "xatol": 1e-4, "fatol": 1e-4},
            )

        self.optimization_wall_time_sec = time.perf_counter() - t0
        print(f"LHS + Nelder-Mead finished. best_loss={self.best_loss:.4f} evals={self._eval_counter}")