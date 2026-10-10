"""
Optuna TPE + HyperbandPruner — sequential halving over the stochastic repeat index.
After each repeat: trial.report(mean_loss_so_far); bad configurations are pruned without full repeats.
"""
from __future__ import annotations

import datetime
import json
import time
from typing import Optional

import numpy as np
from optuna import TrialPruned, create_study, pruners

from classes.optuna_integration import OptunaIntegration
from classes.simulation_objective import DEFAULT_N_REPEATS, default_n_init, latin_hypercube_initial
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper
from resources.utilities import optimization_result_file_path


class OptunaHyperbandIntegration(OptunaIntegration):
    """TPE sampler + HyperbandPruner; resource = repeat index (1..max_r)."""

    def __init__(
        self,
        rows: int,
        theory: dict,
        drawer: Drawer,
        sleeper: Sleeper,
        *,
        n_repeats: int = DEFAULT_N_REPEATS,
        lhs_seed: int = 42,
        tpe_seed: int = 42,
        hyperband_reduction_factor: int = 2,
    ):
        super().__init__(
            rows=rows,
            theory=theory,
            drawer=drawer,
            sleeper=sleeper,
            optimizer_label="Hyperband+TPE",
            result_file_prefix="OptunaHyperband",
            n_repeats=n_repeats,
            lhs_seed=lhs_seed,
            tpe_seed=tpe_seed,
        )
        self._hb_eta = max(2, int(hyperband_reduction_factor))

    def _objective(self, trial):
        self.current_trial = trial.number + 1

        add = trial.suggest_float("add", self._param_low, self._param_high)
        transit = trial.suggest_float("transit", self._param_low, self._param_high)
        merge = trial.suggest_float("merge", self._param_low, self._param_high)

        losses: list[float] = []
        hists: list[dict] = []
        max_r = max(2, int(self._n_repeats))

        for rep in range(max_r):
            if self.sleeper.is_exit:
                raise TrialPruned()

            loss_one, bar_one = self._sim_objective.run_one_repeat(
                add,
                transit,
                merge,
                eval_index=trial.number,
                rep=rep,
            )
            if self.sleeper.is_exit:
                raise TrialPruned()

            losses.append(loss_one)
            hists.append(bar_one)
            mean_so_far = float(sum(losses) / len(losses))
            trial.report(mean_so_far, step=rep + 1)

            if rep + 1 < max_r and trial.should_prune():
                raise TrialPruned()

        mean_loss = float(sum(losses) / len(losses))
        i_best = int(np.argmin(np.asarray(losses, dtype=np.float64)))
        best_bar = hists[i_best] if hists else {}

        self.hists[trial.number] = dict(best_bar) if best_bar else {}

        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            self.best_params = {"add": add, "transit": transit, "merge": merge}
            if best_bar:
                self._best_experimental_bar = dict(best_bar)

        self.drawer.window.after(
            0,
            lambda t=trial.number + 1, l=mean_loss, a=add, tr=transit, m=merge: self._update_ui_stats(
                t, l, a, tr, m
            ),
        )

        return mean_loss

    def optimize(self, n_trials: int = 50, n_init=None):
        if n_trials <= 0:
            raise ValueError("n_trials must be > 0")

        self.total_trials = n_trials
        self.current_trial = 0
        self.best_loss = float("inf")
        self.best_params = None
        self._best_experimental_bar = None
        self.hists = {}

        if n_init is None:
            n_init = default_n_init(n_trials)
        n_init = min(n_init, n_trials)

        max_r = max(2, int(self._n_repeats))
        hb = pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max_r,
            reduction_factor=self._hb_eta,
        )
        self.study = create_study(direction="minimize", sampler=self._sampler, pruner=hb)

        lhs = latin_hypercube_initial(n_init, self._lhs_seed, self._param_low, self._param_high)
        for i in range(len(lhs)):
            self.study.enqueue_trial(
                {
                    "add": float(lhs[i, 0]),
                    "transit": float(lhs[i, 1]),
                    "merge": float(lhs[i, 2]),
                }
            )

        print(
            f"Start {self._optimizer_label}: n_trials={n_trials}, LHS={n_init}, "
            f"Hyperband max_resource={max_r} (repeats), reduction_factor={self._hb_eta}"
        )
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

        complete = [t for t in self.study.get_trials() if t.state.name == "COMPLETE"]
        pruned = len([t for t in self.study.get_trials() if t.state.name == "PRUNED"])
        print(f"\nTrials COMPLETE: {len(complete)}, PRUNED (Hyperband): {pruned}")

        if self.best_params:
            print("\nOptimization done.")
            print(
                f"Best params: add={self.best_params['add']:.4f}, "
                f"transit={self.best_params['transit']:.4f}, "
                f"merge={self.best_params['merge']:.4f}"
            )
            print(f"Best loss (full repeats): {self.best_loss:.4f}")
            print(f"Wall time: {self.optimization_wall_time_sec:.2f} s")
        else:
            print("Optimization did not finish cleanly")
            self.best_params = {"add": 0.3, "transit": 0.2, "merge": 0.1}

    def result(self):
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

        best_trial_num: Optional[int] = None
        try:
            best_trial_num = int(self.study.best_trial.number)
        except (ValueError, RuntimeError, AttributeError):
            pass

        try:
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "metadata": {
                            "optimizer": self._optimizer_label,
                            "total_trials_complete": len(results),
                            "pruned_trials": len(
                                [t for t in self.study.get_trials() if t.state.name == "PRUNED"]
                            ),
                            "best_trial": best_trial_num,
                            "best_loss": self.best_loss,
                            "best_params": self.best_params,
                            "wall_time_sec": self.optimization_wall_time_sec,
                            "n_repeats": self._n_repeats,
                            "hyperband_reduction_factor": self._hb_eta,
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
            print(f"Save error: {e}")
            result_path = None

        print("\n" + "=" * 60)
        print(f"RESULTS {self._optimizer_label}")
        print("=" * 60)
        if self.best_params:
            print(f"add:    {self.best_params['add']:.4f}")
            print(f"transit:{self.best_params['transit']:.4f}")
            print(f"merge:  {self.best_params['merge']:.4f}")
            print(f"loss:   {self.best_loss:.4f}")
        else:
            print("No results")
        print("=" * 60)

        return {
            "best_params": self.best_params,
            "best_loss": self.best_loss,
            "result_file": result_path,
            "wall_time_sec": self.optimization_wall_time_sec,
        }