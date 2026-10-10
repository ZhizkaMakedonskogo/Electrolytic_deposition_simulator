"""
Exponential Natural Evolution Strategies (xNES)
Search region [0.05, 0.95]^3, stochastic objective function via SimulationObjective (mean loss over seeds).
Budget: one unit = one call to black-box evaluate() (like one Optuna trial).
"""
from __future__ import annotations

import datetime
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.linalg import expm

from classes.base_comparator import BaseComparator
from classes.board import Board
from classes.simulation_objective import SimulationObjective, weighted_hist_compare
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper
from resources.constants import Mode
from resources.utilities import optimization_result_file_path


XNES_PARAM_LOW = 0.05
XNES_PARAM_HIGH = 0.95


def _rank_utility_weights(n: int) -> np.ndarray:
    """Fitness for xNES is defined as the largest weight for the best rank (rank 1 = best, 1-based in the formula)"""
    if n <= 1:
        return np.zeros(max(1, n), dtype=float)
    a = float(np.log(n / 2.0 + 1.0))
    u = np.array([max(0.0, a - np.log(r)) for r in range(1, n + 1)], dtype=float)
    s = u.sum()
    if s <= 0:
        return np.zeros(n, dtype=float)
    u /= s
    u -= 1.0 / n
    return u


class XnesOptimization(BaseComparator):
    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        population_size: int = 12,
        sigma0: float = 0.175,
        eta_mu: float = 1.0,
        eta_sigma: float = 0.25,
        eta_b: float = 0.1,
        n_repeats: int = 4,
        param_low: float = XNES_PARAM_LOW,
        param_high: float = XNES_PARAM_HIGH,
        ui_update_every_gen: int = 2,
        convergence_patience: int = 5,
        convergence_tol: float = 1e-4,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._pop_size = max(2, int(population_size))
        self._sigma0 = float(sigma0)
        self._eta_mu = float(eta_mu)
        self._eta_sigma = float(eta_sigma)
        self._eta_b = float(eta_b)
        self._n_repeats = int(n_repeats)
        self._low = float(param_low)
        self._high = float(param_high)
        self._ui_every = max(1, int(ui_update_every_gen))
        self._conv_patience = int(convergence_patience)
        self._conv_tol = float(convergence_tol)

        self._sim_objective = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)

        self.study = None
        self.trials_log: List[dict] = []
        self.current_trial = 0
        self.total_trials = 0
        self.best_loss = float("inf")
        self.best_params: Optional[Dict[str, float]] = None
        self._best_hist: Dict[Any, int] = {}
        self.optimization_wall_time_sec = 0.0
        self._generation = 0

    @staticmethod
    def hist_compare(theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)

    def create_bar(self) -> dict:
        return dict(self._best_hist) if self._best_hist else {}

    def _schedule_ui(
        self,
        *,
        force: bool,
        loss: float,
        add: float,
        transit: float,
        merge: float,
    ) -> None:
        if not hasattr(self.drawer, "window") or self.drawer.window is None:
            return
        if not force and self._generation % self._ui_every != 0:
            return

        def _upd(
            ct=self.current_trial,
            tt=self.total_trials,
            bl=self.best_loss,
            l=loss,
            a=add,
            tr=transit,
            m=merge,
        ):
            try:
                self.drawer.stat_1.config(text=f"Trial: {ct}/{tt}")
                self.drawer.stat_2.config(
                    text=f"Loss: {l:.4f} (best: {bl:.4f})"
                )
                self.drawer.stat_3.config(text=f"add={a:.3f} | trans={tr:.3f} | merge={m:.3f}")
                self.drawer.window.update_idletasks()
            except Exception as e:
                print(f"xNES UI update: {e}")

        try:
            self.drawer.window.after(0, _upd)
        except Exception:
            pass

    def optimize(self, n_trials: int = 50) -> None:
        if n_trials <= 0:
            raise ValueError("Budget n_trials must be > 0")

        self.total_trials = int(n_trials)
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_hist = {}
        self.trials_log = []
        self.study = {"name": "xNES", "budget": self.total_trials}
        self._generation = 0

        d = 3
        mu = np.full(d, 0.5, dtype=float)
        sigma = self._sigma0
        bmat = np.eye(d, dtype=float)
        eval_index = 0

        t0 = time.perf_counter()
        print(
            f"xNES: budget={self.total_trials} evaluations, lambda={self._pop_size}, "
            f"sigma0={sigma:.3f}, bounds=[{self._low}; {self._high}]"
        )
        print("-" * 50)

        flat_streak = 0
        prev_best_vec: Optional[np.ndarray] = None

        try:
            while self.current_trial < self.total_trials and not self.sleeper.is_exit:
                self._generation += 1
                batch = min(self._pop_size, self.total_trials - self.current_trial)
                if batch <= 0:
                    break

                s_noise = np.random.randn(batch, d)
                z = mu + sigma * (s_noise @ bmat)
                z_eval = np.clip(z, self._low, self._high)

                fitness = np.zeros(batch, dtype=float)
                for i in range(batch):
                    if self.sleeper.is_exit:
                        return
                    add, tr, mg = (float(z_eval[i, 0]), float(z_eval[i, 1]), float(z_eval[i, 2]))
                    loss, hist = self._sim_objective.evaluate(
                        add,
                        tr,
                        mg,
                        eval_index=eval_index,
                        n_repeats=self._n_repeats,
                    )
                    eval_index += 1
                    self.current_trial += 1
                    fitness[i] = loss

                    self.trials_log.append(
                        {
                            "number": len(self.trials_log),
                            "value": float(loss),
                            "params": {"add": add, "transit": tr, "merge": mg},
                            "hist": hist,
                        }
                    )

                    if loss < self.best_loss:
                        self.best_loss = float(loss)
                        self.best_params = {"add": add, "transit": tr, "merge": mg}
                        self._best_hist = dict(hist) if hist else {}
                        self._schedule_ui(force=True, loss=loss, add=add, transit=tr, merge=mg)
                    else:
                        self._schedule_ui(force=False, loss=loss, add=add, transit=tr, merge=mg)

                order = np.argsort(fitness)
                s_sort = s_noise[order]

                u = _rank_utility_weights(batch)
                dj_delta = u @ s_sort
                dj_mmat = s_sort.T @ (s_sort * u[:, np.newaxis]) - float(np.sum(u)) * np.eye(d)
                dj_sigma = float(np.trace(dj_mmat)) / d
                dj_bmat = dj_mmat - dj_sigma * np.eye(d)

                if batch >= 2 and np.all(np.isfinite(dj_mmat)):
                    mu = mu + self._eta_mu * sigma * (bmat @ dj_delta)
                    sigma = float(sigma * np.exp(0.5 * self._eta_sigma * dj_sigma))
                    sigma = float(np.clip(sigma, 1e-6, 2.0))
                    try:
                        bmat = bmat @ expm(0.5 * self._eta_b * dj_bmat)
                    except Exception as e:
                        print(f"xNES: expm failed ({e}), skipping B update")
                    mu = np.clip(mu, self._low, self._high)

                if self.best_params:
                    vec = np.array(
                        [self.best_params["add"], self.best_params["transit"], self.best_params["merge"]],
                        dtype=float,
                    )
                    if prev_best_vec is not None:
                        delta = float(np.max(np.abs(vec - prev_best_vec)))
                        if delta < self._conv_tol:
                            flat_streak += 1
                        else:
                            flat_streak = 0
                    prev_best_vec = vec.copy()
                    if flat_streak >= self._conv_patience:
                        print(
                            f"xNES: early stop - best params stable for {self._conv_patience} generations "
                            f"(tol={self._conv_tol})."
                        )
                        break
        finally:
            self.optimization_wall_time_sec = time.perf_counter() - t0

        if self.best_params:
            print("\nxNES optimization finished.")
            print(
                f"Best params: add={self.best_params['add']:.4f}, "
                f"transit={self.best_params['transit']:.4f}, "
                f"merge={self.best_params['merge']:.4f}"
            )
            print(f"Best loss: {self.best_loss:.4f}")
            print(f"Evaluations used: {self.current_trial}/{self.total_trials}")
            print(f"Wall time: {self.optimization_wall_time_sec:.2f} s")
        else:
            print("xNES: no valid result")
            self.best_params = {"add": 0.5, "transit": 0.5, "merge": 0.5}

    def result(self) -> Dict[str, Any]:
        _time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_path = optimization_result_file_path("XNESOptimization", _time)

        payload = {
            "metadata": {
                "optimizer": "xNES",
                "budget": self.total_trials,
                "evaluations_used": self.current_trial,
                "best_loss": self.best_loss,
                "best_params": self.best_params,
                "wall_time_sec": self.optimization_wall_time_sec,
                "n_repeats": self._n_repeats,
                "population_size": self._pop_size,
                "bounds": [self._low, self._high],
                "timestamp": _time,
            },
            "trials": self.trials_log,
        }

        try:
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
            print(f"\nxNES results saved to: {result_path}")
        except Exception as e:
            print(f"xNES: failed to save results: {e}")
            result_path = None

        if self.best_params and not self.sleeper.is_exit:
            print("\nFinal simulation with best parameters (500 steps)...")
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
                        if hasattr(self.drawer, "window") and self.drawer.window is not None:
                            self.drawer.window.update()
                    except Exception as e:
                        print(f"xNES final grid draw: {e}")
                    time.sleep(0.05)

            final_bar = final_board.create_bar()
            self.board = final_board
            self._best_hist = dict(final_bar) if final_bar else self._best_hist
            try:
                self.drawer.draw_bar(final_bar)
                if hasattr(self.drawer, "window") and self.drawer.window is not None:
                    self.drawer.window.update()
            except Exception as e:
                print(f"xNES: histogram draw error: {e}")

            time.sleep(1.5)

        return {
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "result_file": result_path,
            "wall_time_sec": self.optimization_wall_time_sec,
        }