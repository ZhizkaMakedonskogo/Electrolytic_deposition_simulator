import datetime
import json

from classes.base_comparator import BaseComparator
from classes.simulation_objective import (
    DEFAULT_N_REPEATS,
    STEPS_PER_TRIAL,
    SimulationObjective,
    weighted_hist_compare,
)
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper
from resources.utilities import optimization_result_file_path


class CoordinateDescent(BaseComparator):
    def __init__(
        self,
        rows: int,
        addprob: float,
        transitprob: float,
        mergeprob: float,
        drawer: Drawer,
        sleeper: Sleeper,
        theory: dict,
        multiplier: float,
    ):
        super().__init__(rows=rows, drawer=drawer, sleeper=sleeper, theory=theory)
        self.probs = [addprob, transitprob, mergeprob]
        self._hist: list = []
        self.multiplier = multiplier
        self._obj = SimulationObjective(
            rows, theory, sleeper, hist_compare_fn=weighted_hist_compare
        )
        self._eval_idx = 0
        self.best_loss: float = float("inf")

    def _func(self, addprob: float, transitprob: float, mergeprob: float):
        n_repeats = int(getattr(self, "_n_repeats", DEFAULT_N_REPEATS))
        mean_loss, best_bar = self._obj.evaluate(
            addprob,
            transitprob,
            mergeprob,
            eval_index=self._eval_idx,
            n_repeats=n_repeats,
            steps=STEPS_PER_TRIAL,
        )
        self._eval_idx += 1
        return mean_loss, best_bar

    @staticmethod
    def _fixed_step(probs: list, it: int, func, step: float):
        x_ = list(probs)
        x1 = list(probs)

        if probs[it] - step > 0.0:
            x_[it] = probs[it] - step
        if probs[it] + step <= 1.0:
            x1[it] = probs[it] + step

        if x1 == probs and x_ == probs:
            return None, None

        f_, hist_ = func(x_[0], x_[1], x_[2])
        f1, hist1 = func(x1[0], x1[1], x1[2])

        if f_ > f1:
            probs[it] = x1[it]
            return f1, hist1
        else:
            probs[it] = x_[it]
            return f_, hist_

    def optimize(self):
        for i in range(len(self.probs)):
            result = self._fixed_step(
                probs=self.probs, it=i, func=self._func, step=self.multiplier
            )
            if result[0] is not None and i == 2:
                value, hist = result
                self._hist.append(
                    (self.probs[0], self.probs[1], self.probs[2], value, hist)
                )
        if self._hist:
            best = min(self._hist, key=lambda t: t[3])
            self.best_loss = float(best[3])
            if best[4]:
                self._best_experimental_bar = dict(best[4])
        
    def result(self):
        _time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result = []
        num = 0
        for trial in self._hist:
            result.append(
                {
                    "number": num,
                    "value": trial[3],
                    "params": {
                        "add": trial[0],
                        "transit": trial[1],
                        "merge": trial[2],
                    },
                    "hist": trial[4],
                }
            )
            num += 1

        path = optimization_result_file_path("CDOptimization", _time)
        with open(path, "w", encoding="utf-8") as file:
            for trial in result:
                line = json.dumps(trial)
                file.write(line + "\n")

        return result
