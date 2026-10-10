from optuna import create_study, samplers, TrialPruned
import json
import datetime
import time

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
    weighted_hist_compare,
)
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper
from resources.constants import Mode
from resources.utilities import optimization_result_file_path


class OptunaIntegration(BaseComparator):
    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        optimizer_label: str = "TPE",
        result_file_prefix: str = "OptunaOptimization",
        n_repeats: int = DEFAULT_N_REPEATS,
        lhs_seed: int = DEFAULT_LHS_SEED,
        tpe_seed: int = 42,
        param_low: float = PARAM_LOW,
        param_high: float = PARAM_HIGH,
    ):
        super().__init__(rows=rows, theory=theory, drawer=drawer, sleeper=sleeper)
        self._sampler = samplers.TPESampler(seed=tpe_seed)
        self._optimizer_label = optimizer_label
        self._result_file_prefix = result_file_prefix
        self._n_repeats = n_repeats
        self._lhs_seed = lhs_seed
        self._tpe_seed = tpe_seed
        self._param_low = param_low
        self._param_high = param_high
        self._sim_objective = SimulationObjective(rows, theory, sleeper, hist_compare_fn=weighted_hist_compare)
        self.study = None
        self.hists = {}
        self.current_trial = 0
        self.total_trials = 0
        self.best_loss = float("inf")
        self.best_params = None
        self.optimization_wall_time_sec = 0.0

    def _objective(self, trial):
        self.current_trial = trial.number + 1

        add = trial.suggest_float("add", self._param_low, self._param_high)
        transit = trial.suggest_float("transit", self._param_low, self._param_high)
        merge = trial.suggest_float("merge", self._param_low, self._param_high)

        loss, experimental_bar = self._sim_objective.evaluate(
            add,
            transit,
            merge,
            eval_index=trial.number,
            n_repeats=self._n_repeats,
        )

        self.hists[trial.number] = experimental_bar

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = {
                "add": add,
                "transit": transit,
                "merge": merge,
            }
            if experimental_bar:
                self._best_experimental_bar = dict(experimental_bar)

        self.drawer.window.after(
            0,
            lambda t=trial.number + 1, l=loss, a=add, tr=transit, m=merge: self._update_ui_stats(t, l, a, tr, m),
        )

        return loss

    def _safe_draw_grid(self, step: int, board: Board):
        """Safe grid drawing from the main Tkinter thread"""
        try:
            if not hasattr(self.drawer, "window") or self.drawer.window is None:
                return
            try:
                if not self.drawer.window.winfo_exists():
                    return
            except Exception:
                return
            if hasattr(self.drawer, "no_ui") and self.drawer.no_ui.get():
                return
            self.drawer.prepare_draw()
            board.draw(self.drawer)
            self.drawer.complete_draw()
        except Exception as e:
            print(f"Error in _safe_draw_grid (step {step}): {e}")

    def _update_ui_stats(self, trial_num: int, loss: float, add: float, transit: float, merge: float):
        """Update statistics in the main Tkinter thread"""
        try:
            self.drawer.stat_1.config(text=f"Trial: {trial_num}/{self.total_trials}")
            self.drawer.stat_2.config(text=f"Loss: {loss:.4f} (best: {self.best_loss:.4f})")
            self.drawer.stat_3.config(text=f"add={add:.3f} | trans={transit:.3f} | merge={merge:.3f}")
            self.drawer.window.update_idletasks()
        except Exception as e:
            print(f"UI update error: {e}")

    def optimize(self, n_trials: int = 50, n_init=None):
        """Start optimization with a correct stopping rule"""
        if n_trials <= 0:
            raise ValueError("The number of trials must be > 0")

        self.total_trials = n_trials
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.hists = {}

        if n_init is None:
            n_init = default_n_init(n_trials)
        n_init = min(n_init, n_trials)

        self.study = create_study(direction="minimize", sampler=self._sampler)

        lhs = latin_hypercube_initial(n_init, self._lhs_seed, self._param_low, self._param_high)
        for i in range(len(lhs)):
            self.study.enqueue_trial(
                {
                    "add": float(lhs[i][0]),
                    "transit": float(lhs[i][1]),
                    "merge": float(lhs[i][2]),
                }
            )

        print(f"Starting {self._optimizer_label} optimization: {n_trials} trials, LHS: {n_init}")
        print("-" * 50)

        t0 = time.perf_counter()
        try:
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                callbacks=[self._stop_callback],
                show_progress_bar=False,
            )
        except Exception as e:
            print(f"Optimization interrupted: {e}")
        finally:
            self.optimization_wall_time_sec = time.perf_counter() - t0

        if self.best_params:
            print("\nOptimization finished.")
            print(
                f"Best params: add={self.best_params['add']:.4f}, "
                f"transit={self.best_params['transit']:.4f}, "
                f"merge={self.best_params['merge']:.4f}"
            )
            print(f"Minimum loss: {self.best_loss:.4f}")
            print(f"Optimization time: {self.optimization_wall_time_sec:.2f} s")
        else:
            print("Optimization did not finish cleanly")
            self.best_params = {"add": 0.3, "transit": 0.2, "merge": 0.1}

    def _stop_callback(self, study, trial):
        """Callback to check the stopping condition"""
        if self.sleeper.is_exit:
            print(f"\nOptimization stopped by the user at trial {trial.number + 1}")
            raise TrialPruned("Forced stop by the user")

    def result(self):
        """Save and print the optimization results"""
        _time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_path = optimization_result_file_path(self._result_file_prefix, _time)

        results = []
        for trial in self.study.get_trials():
            if trial.state.name == "COMPLETE":
                results.append(
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                        "hist": self.hists.get(trial.number, {}),
                    }
                )

        try:
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "metadata": {
                            "optimizer": self._optimizer_label,
                            "total_trials": len(results),
                            "best_trial": self.study.best_trial.number if self.study.best_trial else None,
                            "best_loss": self.study.best_value if self.study.best_value else None,
                            "best_params": self.best_params,
                            "wall_time_sec": self.optimization_wall_time_sec,
                            "n_repeats": self._n_repeats,
                            "lhs_seed": self._lhs_seed,
                            "tpe_seed": self._tpe_seed,
                            "bounds": [self._param_low, self._param_high],
                            "timestamp": _time,
                        },
                        "trials": results,
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
        print(f"OPTIMIZATION RESULTS {self._optimizer_label}")
        print("=" * 60)
        if self.best_params:
            print(f"Creation probability:   {self.best_params['add']:.4f}")
            print(f"Transition probability: {self.best_params['transit']:.4f}")
            print(f"Merge probability:      {self.best_params['merge']:.4f}")
            print(f"Achieved loss:          {self.best_loss:.4f}")
        else:
            print("No optimization results")
        print("=" * 60)

        if self.best_params and not self.sleeper.is_exit:
            print("\nStarting the final simulation with the optimal parameters...")
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
                print(f"Drawing histogram: {len(final_bar)} values")
                self.drawer.draw_bar(final_bar)
                self.drawer.window.update()
                print("Histogram drawn.")
            except Exception as e:
                print(f"Histogram draw error: {e}")
                import traceback

                traceback.print_exc()

            print("Window open for 3 seconds for viewing")
            time.sleep(3.0)
            print("Final simulation finished.")

        return {
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "result_file": result_path,
            "wall_time_sec": self.optimization_wall_time_sec,
        }

    def hist_compare(self, theory: dict, experimental: dict) -> float:
        return weighted_hist_compare(theory, experimental)