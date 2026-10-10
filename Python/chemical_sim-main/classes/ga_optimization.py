from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from classes.base_comparator import BaseComparator
from classes.simulation_objective import SimulationObjective, weighted_hist_compare
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper


def _sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
    low: float,
    high: float,
    eta: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover, bounded to [low, high]"""
    dim = len(p1)
    u = rng.random(dim)
    beta = np.empty(dim)
    mask = u <= 0.5
    beta[mask] = (2.0 * u[mask]) ** (1.0 / (eta + 1.0))
    beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (eta + 1.0))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, low, high), np.clip(c2, low, high)


def _poly_mutation(
    x: np.ndarray,
    rng: np.random.Generator,
    low: float,
    high: float,
    eta: float = 20.0,
    p_mutate_dim: float = 0.25,
) -> np.ndarray:
    """Polynomial mutation for real-valued chromosomes (per-dimension probability)"""
    y = x.copy()
    span = high - low
    for j in range(len(y)):
        if rng.random() > p_mutate_dim:
            continue
        u = rng.random()
        if u < 0.5:
            dq = (2 * u) ** (1.0 / (eta + 1)) - 1.0
        else:
            dq = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1))
        y[j] = y[j] + dq * span
    return np.clip(y, low, high)


class GaOptimization(BaseComparator):
    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        population_size: int = 16,
        tournament_k: int = 3,
        n_repeats: int = 3,
        param_low: float = 0.0,
        param_high: float = 1.0,
        sbx_eta: float = 15.0,
        mut_eta: float = 20.0,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._pop = max(4, int(population_size))
        self._k = max(2, min(int(tournament_k), self._pop))
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._sbx_eta = float(sbx_eta)
        self._mut_eta = float(mut_eta)

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

    def _tournament_idx(self, fitness: np.ndarray, rng: np.random.Generator) -> int:
        idx = rng.choice(self._pop, size=self._k, replace=False)
        sub = fitness[idx]
        return int(idx[np.argmin(sub)])

    def optimize(self, n_trials: int = 50) -> None:
        if n_trials < 4:
            raise ValueError("n_trials must be >= 4")

        self.total_trials = int(n_trials)
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.trials_log = []

        rng = np.random.default_rng(self._rng_seed)
        dim = 3

        pop = rng.uniform(self._low, self._high, size=(self._pop, dim))
        fitness = np.empty(self._pop, dtype=float)

        t0 = time.perf_counter()
        eval_idx = 0

        for i in range(self._pop):
            if eval_idx >= self.total_trials or self.sleeper.is_exit:
                break
            eval_idx += 1
            self.current_trial = eval_idx
            fitness[i] = self._eval(pop[i], eval_idx)

        while eval_idx < self.total_trials and not self.sleeper.is_exit:
            ia = self._tournament_idx(fitness, rng)
            ib = self._tournament_idx(fitness, rng)
            c1, _c2 = _sbx_crossover(pop[ia], pop[ib], rng, self._low, self._high, self._sbx_eta)
            child = _poly_mutation(c1, rng, self._low, self._high, self._mut_eta)

            eval_idx += 1
            self.current_trial = eval_idx
            f_child = self._eval(child, eval_idx)

            worst = int(np.argmax(fitness))
            if f_child < fitness[worst]:
                pop[worst] = child
                fitness[worst] = f_child

        self.optimization_wall_time_sec = time.perf_counter() - t0
        if self.best_params:
            print(f"GA (steady-state) finished. best_loss={self.best_loss:.4f} evals={eval_idx}")