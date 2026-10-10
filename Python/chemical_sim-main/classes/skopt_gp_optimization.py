"""
Bayesian optimization via scikit-optimize (Gaussian Process + Expected Improvement).
Consistent with the TPE path: the same bounds, LHS warm start, seed-averaged objective function.
"""
from __future__ import annotations

import datetime
import inspect
import json
import time
from typing import Any, Dict, List, Optional

from skopt import gp_minimize
from skopt.space import Real

from classes.base_comparator import BaseComparator
from classes.board import Board
from classes.simulation_objective import (
    DEFAULT_LHS_SEED,
    DEFAULT_N_REPEATS,
    PARAM_HIGH,
    PARAM_LOW,
    SimulationObjective,
    default_n_init,
    latin_hypercube_initial,
    loss_scalar_for_surrogate_fit,
    weighted_hist_compare,
)
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper
from resources.constants import Mode
from resources.utilities import optimization_result_file_path


class StopOptimization(Exception):
    """Raised when the user interrupts optimization (skopt has no TrialPruned)"""


class SkoptGpOptimization(BaseComparator):
    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        n_repeats: int = DEFAULT_N_REPEATS,
        lhs_seed: int = DEFAULT_LHS_SEED,
        skopt_random_state: int = 42,
        param_low: float = PARAM_LOW,
        param_high: float = PARAM_HIGH,
        acq_func: str = "EI",
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._n_repeats = n_repeats
        self._lhs_seed = lhs_seed
        self._skopt_random_state = skopt_random_state
        self._param_low = param_low
        self._param_high = param_high
        self._acq_func = acq_func

        self._sim_obj = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)
        self.hists: Dict[int, dict] = {}
        self._trials_log: List[dict] = []
        self.current_trial = 0
        self.total_trials = 0
        self.best_loss = float("inf")
        self.best_params: Optional[Dict[str, float]] = None
        self._eval_index = 0
        self.optimization_wall_time_sec = 0.0
        self._skopt_result = None

    def hist_compare(self, theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)

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
            raise ValueError("n_trials must be > 0")

        self.total_trials = n_trials
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.hists = {}
        self._trials_log = []
        self._eval_index = 0
        self._skopt_result = None

        if n_init is None:
            n_init = default_n_init(n_trials)
        n_init = min(n_init, n_trials)

        dimensions = [
            Real(self._param_low, self._param_high),
            Real(self._param_low, self._param_high),
            Real(self._param_low, self._param_high),
        ]

        x0_arr = latin_hypercube_initial(n_init, self._lhs_seed, self._param_low, self._param_high)
        x0_list: List[List[float]] = [[float(r[0]), float(r[1]), float(r[2])] for r in x0_arr]

        def objective(x) -> float:
            if self.sleeper.is_exit:
                raise StopOptimization("Stopped by user")
            add, transit, merge = float(x[0]), float(x[1]), float(x[2])
            idx = self._eval_index
            self._eval_index += 1

            loss, hist = self._sim_obj.evaluate(
                add,
                transit,
                merge,
                eval_index=idx,
                n_repeats=self._n_repeats,
            )

            trial_no = len(self._trials_log) + 1
            self.current_trial = trial_no
            self.hists[trial_no - 1] = hist

            entry = {
                "number": trial_no - 1,
                "value": loss,
                "params": {"add": add, "transit": transit, "merge": merge},
                "hist": hist,
            }
            self._trials_log.append(entry)

            if loss < self.best_loss and loss < float("inf"):
                self.best_loss = loss
                self.best_params = {"add": add, "transit": transit, "merge": merge}
                if hist:
                    self._best_experimental_bar = dict(hist)

            self.drawer.window.after(
                0,
                lambda t=trial_no, l=loss, a=add, tr=transit, m=merge: self._update_ui_stats(t, l, a, tr, m),
            )
            return loss_scalar_for_surrogate_fit(loss)

        print(f"skopt GP ({self._acq_func}): {n_trials} trials, LHS init: {n_init}")
        print("-" * 50)

        t0 = time.perf_counter()
        _gm_sig = inspect.signature(gp_minimize)
        _gm_kw = {}
        if "n_initial_points" in _gm_sig.parameters:
            _gm_kw["n_initial_points"] = 0
        try:
            self._skopt_result = gp_minimize(
                objective,
                dimensions,
                n_calls=n_trials,
                x0=x0_list if x0_list else None,
                acq_func=self._acq_func,
                random_state=self._skopt_random_state,
                verbose=False,
                **_gm_kw,
            )
        except StopOptimization as e:
            print(f"\nStopped: {e}")
            self._skopt_result = None
        except Exception as e:
            print(f"skopt optimization interrupted: {e}")
            self._skopt_result = None
        finally:
            self.optimization_wall_time_sec = time.perf_counter() - t0

        if (
            self._skopt_result is not None
            and self._skopt_result.x is not None
            and self.best_params is None
        ):
            xb = self._skopt_result.x
            self.best_params = {"add": float(xb[0]), "transit": float(xb[1]), "merge": float(xb[2])}
            self.best_loss = float(self._skopt_result.fun)

        if self._trials_log:
            best_entry = min(self._trials_log, key=lambda e: e["value"])
            if best_entry.get("hist"):
                self._best_experimental_bar = dict(best_entry["hist"])

        if self.best_params:
            print("\nskopt optimization finished.")
            print(
                f"Best params: add={self.best_params['add']:.4f}, "
                f"transit={self.best_params['transit']:.4f}, "
                f"merge={self.best_params['merge']:.4f}"
            )
            print(f"Best loss: {self.best_loss:.4f}")
            print(f"Wall time: {self.optimization_wall_time_sec:.2f} s")
        else:
            print("skopt optimization did not complete cleanly")
            self.best_params = {"add": 0.3, "transit": 0.2, "merge": 0.1}

    def result(self):
        _time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_path = optimization_result_file_path("SkoptGpOptimization", _time)

        try:
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "metadata": {
                            "optimizer": "skopt_gp_minimize",
                            "acq_func": self._acq_func,
                            "total_trials": len(self._trials_log),
                            "best_loss": self.best_loss,
                            "best_params": self.best_params,
                            "wall_time_sec": self.optimization_wall_time_sec,
                            "n_repeats": self._n_repeats,
                            "lhs_seed": self._lhs_seed,
                            "skopt_random_state": self._skopt_random_state,
                            "bounds": [self._param_low, self._param_high],
                            "timestamp": _time,
                        },
                        "trials": self._trials_log,
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"\nSaved results to: {result_path}")
        except Exception as e:
            print(f"Save error: {e}")
            result_path = None

        print("\n" + "=" * 60)
        print("RESULTS: GP (skopt, EI)")
        print("=" * 60)
        if self.best_params:
            print(f"add:     {self.best_params['add']:.4f}")
            print(f"transit: {self.best_params['transit']:.4f}")
            print(f"merge:   {self.best_params['merge']:.4f}")
            print(f"loss:    {self.best_loss:.4f}")
        else:
            print("No optimization results")
        print("=" * 60)

        if self.best_params and not self.sleeper.is_exit:
            print("\nFinal visualization run")
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
                print(f"Drawing histogram ({len(final_bar)} bins)")
                self.drawer.draw_bar(final_bar)
                self.drawer.window.update()
                print("Histogram drawn.")
            except Exception as e:
                print(f"Histogram error: {e}")
                import traceback

                traceback.print_exc()

            print("Pause 3s")
            time.sleep(3.0)
            print("Done.")

        return {
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "result_file": result_path,
            "wall_time_sec": self.optimization_wall_time_sec,
        }