"""
Particle Swarm Optimization (PSO)

Search region [param_low, param_high]^3, objective function: SimulationObjective (mean over seeds).
One budget unit = one call to evaluate().
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from classes.base_comparator import BaseComparator
from classes.simulation_objective import SimulationObjective, weighted_hist_compare
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper


def default_swarm_size(n_trials: int) -> int:
    """Automatic swarm size, consistent with the script"""
    nt = max(2, int(n_trials))
    return min(16, max(6, nt // 3))


def resolve_swarm_size(n_trials: int, swarm_size: int | None = None) -> int:
    """Returns the effective swarm size; 0 and None mean the default from n_trials"""
    if swarm_size is not None and int(swarm_size) > 0:
        return max(2, int(swarm_size))
    return default_swarm_size(n_trials)


class PsoOptimization(BaseComparator):
    """
    Canonical PSO (minimization), where velocity is updated with inertia + cognitive + social terms.
    Velocities are bounded by ``vmax_frac`` of the side length of the region in each dimension.
    """

    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        swarm_size: int = 16,
        inertia: float = 0.73,
        cognitive: float = 1.496,
        social: float = 1.496,
        vmax_frac: float = 0.25,
        n_repeats: int = 3,
        param_low: float = 0.0,
        param_high: float = 1.0,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n = max(2, int(swarm_size))
        self._w = float(inertia)
        self._c1 = float(cognitive)
        self._c2 = float(social)
        self._vmax_frac = float(max(1e-6, vmax_frac))
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._span = self._high - self._low
        self._vmax = self._span * self._vmax_frac

        self._sim = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)
        self._rng_seed = rng_seed

        self.trials_log: List[dict] = []
        self.current_trial = 0
        self.total_trials = 0
        self.best_loss = float("inf")
        self.best_params: Optional[Dict[str, float]] = None
        self.optimization_wall_time_sec = 0.0

    @staticmethod
    def hist_compare(theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)

    def _clip_pos(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self._low, self._high)

    def _eval(self, x: np.ndarray, eval_index: int) -> float:
        loss, hist = self._sim.evaluate(
            float(x[0]),
            float(x[1]),
            float(x[2]),
            eval_index=eval_index,
            n_repeats=self._n_repeats,
        )
        self.trials_log.append(
            {
                "number": len(self.trials_log),
                "value": float(loss),
                "params": {"add": float(x[0]), "transit": float(x[1]), "merge": float(x[2])},
                "hist": hist,
            }
        )
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_params = {"add": float(x[0]), "transit": float(x[1]), "merge": float(x[2])}
            if hist:
                self._best_experimental_bar = dict(hist)
        return float(loss)

    def optimize(self, n_trials: int = 50) -> None:
        if n_trials < 2:
            raise ValueError("n_trials must be >= 2")

        self.total_trials = int(n_trials)
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.trials_log = []

        rng = np.random.default_rng(self._rng_seed)
        dim = 3

        # Positions and velocities
        x = rng.uniform(self._low, self._high, size=(self._n, dim))
        v = rng.uniform(-self._vmax, self._vmax, size=(self._n, dim))
        pbest = x.copy()
        f = np.empty(self._n, dtype=float)
        fpbest = np.empty(self._n, dtype=float)

        t0 = time.perf_counter()
        eval_idx = 0

        # Initial evaluations
        for i in range(self._n):
            if eval_idx >= self.total_trials or self.sleeper.is_exit:
                break
            eval_idx += 1
            self.current_trial = eval_idx
            fi = self._eval(x[i], eval_idx)
            f[i] = fi
            fpbest[i] = fi

        if eval_idx == 0:
            self.optimization_wall_time_sec = time.perf_counter() - t0
            return

        g_idx = int(np.argmin(fpbest))
        gbest = pbest[g_idx].copy()

        while eval_idx < self.total_trials and not self.sleeper.is_exit:
            r1 = rng.random((self._n, dim))
            r2 = rng.random((self._n, dim))
            v = self._w * v + self._c1 * r1 * (pbest - x) + self._c2 * r2 * (gbest - x)
            v = np.clip(v, -self._vmax, self._vmax)
            x = self._clip_pos(x + v)

            for i in range(self._n):
                if eval_idx >= self.total_trials or self.sleeper.is_exit:
                    break
                eval_idx += 1
                self.current_trial = eval_idx
                fi = self._eval(x[i], eval_idx)
                f[i] = fi
                if fi < fpbest[i]:
                    fpbest[i] = fi
                    pbest[i] = x[i].copy()
                g_idx = int(np.argmin(fpbest))
                gbest = pbest[g_idx].copy()

        self.optimization_wall_time_sec = time.perf_counter() - t0
        if self.best_params:
            print(f"PSO finished. best_loss={self.best_loss:.4f} evals={eval_idx}")