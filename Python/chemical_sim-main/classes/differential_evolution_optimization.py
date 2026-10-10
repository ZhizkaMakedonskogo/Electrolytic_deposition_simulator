"""
Differential evolution with a fixed evaluation budget.
``n_trials`` calls to SimulationObjective.evaluate (for a fair comparison with other optimizers).
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from classes.base_comparator import BaseComparator
from classes.simulation_objective import SimulationObjective, weighted_hist_compare
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper


class DifferentialEvolutionOptimization(BaseComparator):
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
        rng_seed: int = 42,
        population_size: int = 0,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._rng_seed = rng_seed
        self._pop_override = int(population_size) if int(population_size) > 0 else 0
        self._sim = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)

        self.current_trial = 0
        self.total_trials = 0
        self.best_loss = float("inf")
        self.best_params: Optional[Dict[str, float]] = None
        self.optimization_wall_time_sec = 0.0

    @staticmethod
    def hist_compare(theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)

    def optimize(self, n_trials: int = 50) -> None:
        budget = int(n_trials)
        if budget < 4:
            raise ValueError("DE requires n_trials >= 4")

        rng = np.random.default_rng(self._rng_seed)
        if self._pop_override > 0:
            pop_size = max(2, min(self._pop_override, budget - 1))
        else:
            raw_pop = int(min(12, max(4, budget // 4)))
            pop_size = min(raw_pop, max(2, budget // 3))
            pop_size = max(2, min(pop_size, budget - 1))

        dim = 3
        pop = rng.uniform(self._low, self._high, size=(pop_size, dim))
        fitness = np.zeros(pop_size, dtype=float)

        self.total_trials = budget
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None

        eval_idx = 0

        def eval_vec(z: np.ndarray) -> float:
            nonlocal eval_idx
            if eval_idx >= budget:
                return float("inf")
            eval_idx += 1
            self.current_trial = eval_idx
            loss, hist = self._sim.evaluate(
                float(z[0]),
                float(z[1]),
                float(z[2]),
                eval_index=eval_idx,
                n_repeats=self._n_repeats,
            )
            if loss < self.best_loss:
                self.best_loss = float(loss)
                self.best_params = {
                    "add": float(z[0]),
                    "transit": float(z[1]),
                    "merge": float(z[2]),
                }
                if hist:
                    self._best_experimental_bar = dict(hist)
            return float(loss)

        t0 = time.perf_counter()
        for j in range(pop_size):
            fitness[j] = eval_vec(pop[j])

        F = 0.8
        Cr = 0.9

        while eval_idx < budget and not self.sleeper.is_exit:
            target = int(rng.integers(0, pop_size))
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), self._low, self._high)
            trial = pop[target].copy()
            cross = rng.random(dim) < Cr
            if not np.any(cross):
                cross[rng.integers(0, dim)] = True
            trial[cross] = mutant[cross]
            fit_trial = eval_vec(trial)
            if fit_trial <= fitness[target]:
                pop[target] = trial
                fitness[target] = fit_trial

        self.optimization_wall_time_sec = time.perf_counter() - t0
        if self.best_params:
            print(f"DE finished. best_loss={self.best_loss:.4f} evals={eval_idx}")