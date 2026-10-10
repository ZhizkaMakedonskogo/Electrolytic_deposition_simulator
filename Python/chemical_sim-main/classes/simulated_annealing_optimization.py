"""
Simulated Annealing (SA)
A single trajectory with Metropolis acceptance and geometric cooling; proposals are
Gaussian perturbations clipped to the bounds. Complements population-based methods (GA/PSO/CEM)
and surrogate BO (GP/RF).
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from classes.base_comparator import BaseComparator
from classes.simulation_objective import SimulationObjective, weighted_hist_compare
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper


class SimulatedAnnealingOptimization(BaseComparator):
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
        t0: float = 1.0,
        t_min: float = 1e-4,
        cooling: float = 0.92,
        proposal_frac: float = 0.12,
        rng_seed: int = 42,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._t0 = float(t0)
        self._t_min = float(t_min)
        self._cooling = float(min(0.999, max(0.5, cooling)))
        self._prop = float(proposal_frac) * (self._high - self._low)
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

    def _eval(self, x: np.ndarray) -> float:
        self._eval_counter += 1
        self.current_trial = self._eval_counter
        loss, hist = self._sim.evaluate(
            float(x[0]),
            float(x[1]),
            float(x[2]),
            eval_index=self._eval_counter,
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
        self._eval_counter = 0

        dim = 3
        current = self._rng.uniform(self._low, self._high, size=dim)
        t0 = time.perf_counter()

        f_cur = self._eval(current)
        temp = self._t0

        while self._eval_counter < self.total_trials and not self.sleeper.is_exit:
            prop = current + self._rng.normal(0.0, self._prop, size=dim)
            prop = np.clip(prop, self._low, self._high)

            f_new = self._eval(prop)
            if self._eval_counter >= self.total_trials:
                break

            df = f_new - f_cur
            if df < 0 or self._rng.random() < math.exp(-df / max(temp, self._t_min)):
                current = prop.copy()
                f_cur = f_new

            temp = max(self._t_min, temp * self._cooling)

        self.optimization_wall_time_sec = time.perf_counter() - t0
        if self.best_params:
            print(f"Simulated Annealing finished. best_loss={self.best_loss:.4f} evals={self._eval_counter}")