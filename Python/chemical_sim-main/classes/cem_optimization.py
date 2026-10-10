"""
Cross-Entropy Method (CEM)
Supports a factorized Gaussian distribution over theta in R^3;
at each iteration: sample a batch, select the elite, update the mean and standard deviation;
Difference from GA/PSO: an explicit distribution over the parameters (novelty: the "distribution-search" line
vs particle / population replacement).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from classes.base_comparator import BaseComparator
from classes.simulation_objective import SimulationObjective, weighted_hist_compare
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper


class CemOptimization(BaseComparator):
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
        batch_size: int = 12,
        elite_frac: float = 0.25,
        init_std_frac: float = 0.25,
        min_std: float = 1e-3,
        rng_seed: int = 42,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._batch = max(4, int(batch_size))
        self._elite_frac = float(min(0.5, max(0.05, elite_frac)))
        self._init_std = float(init_std_frac) * (self._high - self._low)
        self._min_std = float(min_std)
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
        if n_trials < 8:
            raise ValueError("n_trials must be >= 8 for CEM")

        self.total_trials = int(n_trials)
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.trials_log = []
        self._eval_counter = 0

        dim = 3
        span = self._high - self._low
        mean = np.full(dim, 0.5 * (self._low + self._high))
        std = np.full(dim, max(self._init_std, self._min_std))

        t0 = time.perf_counter()

        while self._eval_counter < self.total_trials and not self.sleeper.is_exit:
            n_batch = min(self._batch, self.total_trials - self._eval_counter)
            if n_batch <= 0:
                break

            raw = self._rng.normal(loc=mean, scale=std, size=(n_batch, dim))
            batch = np.clip(raw, self._low, self._high)

            losses = np.empty(n_batch, dtype=float)
            for i in range(n_batch):
                if self._eval_counter >= self.total_trials or self.sleeper.is_exit:
                    break
                losses[i] = self._eval(batch[i])

            if self._eval_counter >= self.total_trials or self.sleeper.is_exit:
                break

            n_elite = max(2, int(np.ceil(n_batch * self._elite_frac)))
            elite_idx = np.argsort(losses)[:n_elite]
            elite = batch[elite_idx]
            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-6
            std = np.maximum(std, self._min_std)
            std = np.minimum(std, 0.5 * span)

        self.optimization_wall_time_sec = time.perf_counter() - t0
        if self.best_params:
            print(f"CEM finished. best_loss={self.best_loss:.4f} evals={self._eval_counter}")