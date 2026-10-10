"""
Gaussian process with Expected Improvement (sequential Bayesian optimization).
Latin Hypercube; surrogate: Matern + white noise. SimulationObjective, seed-averaged, search region [param_low, param_high]^3.
"""
from __future__ import annotations

import datetime
import json
import time
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from classes.base_comparator import BaseComparator
from classes.board import Board
from classes.simulation_objective import (
    DEFAULT_LHS_SEED,
    DEFAULT_N_REPEATS,
    PARAM_HIGH,
    PARAM_LOW,
    STEPS_PER_TRIAL,
    SimulationObjective,
    default_n_init,
    latin_hypercube_initial,
    numpy_losses_for_surrogate_fit,
    weighted_hist_compare,
)
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper
from resources.constants import Mode
from resources.utilities import optimization_result_file_path


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """EI for minimization; y_best is the minimum observed (same scale as mu)."""
    sigma = np.maximum(sigma, 1e-12)
    imp = y_best - mu - xi
    z = imp / sigma
    ei = imp * scipy_stats.norm.cdf(z) + sigma * scipy_stats.norm.pdf(z)
    return np.maximum(ei, 0.0)


class GPBayesianOptimization(BaseComparator):
    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        n_repeats: int = DEFAULT_N_REPEATS,
        lhs_seed: int = DEFAULT_LHS_SEED,
        gpr_random_state: int = 42,
        param_low: float = PARAM_LOW,
        param_high: float = PARAM_HIGH,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n_repeats = int(n_repeats)
        self._lhs_seed = int(lhs_seed)
        self._gpr_random_state = int(gpr_random_state)
        self._param_low = float(param_low)
        self._param_high = float(param_high)

        self._sim_obj = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)
        self.hists: dict = {}
        self._trials: list = []
        self.current_trial = 0
        self.total_trials = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._eval_index = 0
        self._rng = np.random.default_rng(self._lhs_seed)
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        self._gpr: Optional[GaussianProcessRegressor] = None
        self._use_x_scaling = True

    def hist_compare(self, theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)

    def _evaluate(self, add: float, transit: float, merge: float) -> Tuple[float, Dict]:
        idx = self._eval_index
        self._eval_index += 1
        loss, hist = self._sim_obj.evaluate(
            add,
            transit,
            merge,
            eval_index=idx,
            n_repeats=self._n_repeats,
            steps=STEPS_PER_TRIAL,
        )
        return loss, hist

    def _make_gpr(self) -> GaussianProcessRegressor:
        kernel = (
            C(1.0, (1e-2, 1e2))
            * Matern(
                length_scale=np.ones(3),
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=8,
            random_state=self._gpr_random_state,
            alpha=1e-10,
        )

    def _fit_surrogate(self, x_list: np.ndarray, y_list: np.ndarray) -> Tuple[GaussianProcessRegressor, np.ndarray]:
        y_list = numpy_losses_for_surrogate_fit(y_list)

        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()

        self._use_x_scaling = len(x_list) >= 2
        if self._use_x_scaling:
            xn = self._x_scaler.fit_transform(x_list)
        else:
            xn = np.asarray(x_list, dtype=float)

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
        xn = self._x_scaler.transform(x) if self._use_x_scaling else x
        mu, sigma = gpr.predict(xn, return_std=True)
        ei = _expected_improvement(mu, sigma, y_best_scaled, xi=0.01)
        return -float(ei[0])

    def _maximize_acquisition(
        self,
        gpr: GaussianProcessRegressor,
        y_best_scaled: float,
        n_restarts: int = 32,
    ) -> Tuple[np.ndarray, float]:
        bounds = [(self._param_low, self._param_high)] * 3
        best_x = None
        best_ei = -1.0

        for _ in range(n_restarts):
            x0 = np.array([self._rng.uniform(lo, hi) for lo, hi in bounds])
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
            best_x = np.array(
                [self._rng.uniform(self._param_low, self._param_high) for _ in range(3)]
            )
            best_ei = 0.0

        return np.clip(best_x, self._param_low, self._param_high), best_ei

    def _update_ui_stats(self, trial_num: int, loss: float, add: float, transit: float, merge: float):
        try:
            self.drawer.stat_1.config(text=f"Trial: {trial_num}/{self.total_trials}")
            self.drawer.stat_2.config(text=f"Loss: {loss:.4f} (best: {self.best_loss:.4f})")
            self.drawer.stat_3.config(text=f"add={add:.3f} | trans={transit:.3f} | merge={merge:.3f}")
            self.drawer.window.update_idletasks()
        except Exception as e:
            print(f"UI update error: {e}")

    def optimize(self, n_trials: int = 50, n_init: Optional[int] = None):
        if n_trials <= 0:
            raise ValueError("The number of trials must be > 0")

        self.total_trials = n_trials
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.hists = {}
        self._trials = []
        self._gpr = None
        self._eval_index = 0

        if n_init is None:
            n_init = default_n_init(n_trials)
        n_init = min(n_init, n_trials)

        x_accum = []
        y_accum = []

        print(f"GP (Bayesian optimization): {n_trials} trials, LHS: {n_init}, n_repeats={self._n_repeats}")
        print("-" * 50)

        x0 = latin_hypercube_initial(n_init, self._lhs_seed, self._param_low, self._param_high)
        for i in range(n_init):
            if self.sleeper.is_exit:
                break
            self.current_trial = i + 1
            add, transit, merge = float(x0[i, 0]), float(x0[i, 1]), float(x0[i, 2])
            loss, hist = self._evaluate(add, transit, merge)
            self.hists[i] = hist
            x_accum.append([add, transit, merge])
            y_accum.append(loss)
            self._trials.append(
                {
                    "number": i,
                    "value": loss,
                    "params": {"add": add, "transit": transit, "merge": merge},
                    "hist": hist,
                }
            )

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = {"add": add, "transit": transit, "merge": merge}
                if hist:
                    self._best_experimental_bar = dict(hist)

            self.drawer.window.after(
                0,
                lambda t=i + 1, l=loss, a=add, tr=transit, m=merge: self._update_ui_stats(t, l, a, tr, m),
            )

        x_arr = np.asarray(x_accum, dtype=float)
        y_arr = np.asarray(y_accum, dtype=float)

        plateau = 0
        plateau_limit = min(8, max(3, n_trials // 5))
        ei_stop = 1e-6

        t = n_init
        while t < n_trials and not self.sleeper.is_exit:
            self.current_trial = t + 1

            gpr, yn = self._fit_surrogate(x_arr, y_arr)
            self._gpr = gpr
            y_best_scaled = float(np.min(yn))

            x_next, ei_max = self._maximize_acquisition(gpr, y_best_scaled)
            add, transit, merge = float(x_next[0]), float(x_next[1]), float(x_next[2])

            loss, hist = self._evaluate(add, transit, merge)
            self.hists[t] = hist

            x_arr = np.vstack([x_arr, x_next.reshape(1, -1)])
            y_arr = np.append(y_arr, loss)

            self._trials.append(
                {
                    "number": t,
                    "value": loss,
                    "params": {"add": add, "transit": transit, "merge": merge},
                    "hist": hist,
                }
            )

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_params = {"add": add, "transit": transit, "merge": merge}
                if hist:
                    self._best_experimental_bar = dict(hist)
                plateau = 0
            else:
                plateau += 1

            self.drawer.window.after(
                0,
                lambda num=t + 1, l=loss, a=add, tr=transit, m=merge: self._update_ui_stats(num, l, a, tr, m),
            )

            if ei_max < ei_stop and t > n_init:
                print(f"Stop: EI < {ei_stop}")
                break
            if plateau >= plateau_limit:
                print(f"Stop: best loss has not improved for {plateau_limit} trials")
                break

            t += 1

        if self.best_params:
            print("\nOptimization finished!")
            print(
                f"Best params: add={self.best_params['add']:.4f}, "
                f"transit={self.best_params['transit']:.4f}, "
                f"merge={self.best_params['merge']:.4f}"
            )
            print(f"Minimum loss: {self.best_loss:.4f}")
        else:
            print("Optimization produced no result")
            self.best_params = {"add": 0.3, "transit": 0.2, "merge": 0.1}

    def result(self):
        _time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_path = optimization_result_file_path("GPBayesianOptimization", _time)

        try:
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "metadata": {
                            "method": "GP_Bayesian_EI",
                            "total_trials": len(self._trials),
                            "best_loss": self.best_loss,
                            "best_params": self.best_params,
                            "n_repeats": self._n_repeats,
                            "timestamp": _time,
                        },
                        "trials": self._trials,
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"\nResults saved to: {result_path}")
        except Exception as e:
            print(f"Failed to save results: {e}")
            result_path = None

        print("\n" + "=" * 60)
        print("Result: GP with Bayesian optimization (EI)")
        print("=" * 60)
        if self.best_params:
            print(f"Optimal add:     {self.best_params['add']:.4f}")
            print(f"Optimal transit: {self.best_params['transit']:.4f}")
            print(f"Optimal merge:   {self.best_params['merge']:.4f}")
            print(f"Minimum loss:    {self.best_loss:.4f}")
        else:
            print("No optimal parameters")
        print("=" * 60)

        if self.best_params and not self.sleeper.is_exit:
            print("\nStarting the final simulation with the optimal parameters")
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
                        self.drawer.window.update()
                    except Exception as e:
                        print(f"Draw error: {e}")
                    time.sleep(0.05)

            final_bar = final_board.create_bar()
            try:
                print(f"Final histogram: {len(final_bar)} clusters")
                self.drawer.draw_bar(final_bar)
                self.drawer.window.update()
                print("Histogram drawn!")
            except Exception as e:
                print(f"Histogram draw error: {e}")
                import traceback

                traceback.print_exc()

            print("The window will close in 3 seconds")
            time.sleep(3.0)
            print("Final simulation finished")

        return {
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "result_file": result_path,
        }