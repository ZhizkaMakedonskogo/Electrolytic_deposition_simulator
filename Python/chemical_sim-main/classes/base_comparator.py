from __future__ import annotations

import time
from typing import Any, Dict, Optional

from classes import Board
from classes.simulation_objective import STEPS_PER_TRIAL
from interfaces.optimization import Optimization
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper
from math import sqrt
from numpy.polynomial.polynomial import polyfit


class BaseComparator(Optimization):
    def __init__(
            self,
            rows: int,
            theory: dict,
            drawer: Drawer,
            sleeper: Sleeper,
    ):
        self.rows = rows
        self.theory = theory
        self.sleeper = sleeper
        self.drawer = drawer
        self.board = None
        self._best_experimental_bar: Optional[Dict[Any, Any]] = None

    def create_bar(self) -> Dict[Any, Any]:
        """
        Histogram for the UI: for the inverse problem — the last best by loss;
        otherwise — from the current Board (forward modeling mode).
        """
        if self._best_experimental_bar is not None and len(self._best_experimental_bar) > 0:
            return dict(self._best_experimental_bar)
        if self.board is not None:
            return self.board.create_bar()
        return {}

    def optimize(self):
        pass

    def result(self):
        pass

    def draw(self):
        self.drawer.prepare_draw()
        self.board.draw(self.drawer)
        self.drawer.draw_bar(self.board.create_bar())
        self.drawer.complete_draw()

    def _modelling(self):
        self.board.run()
        self.draw()
        self.sleeper.sleep()

    def _forward_steps_for_theory(self) -> None:
        """
        Fixed number of lattice updates,
        len(bar) <= len(theory) and 1000 iterations.
        """
        for _ in range(STEPS_PER_TRIAL):
            if getattr(self.sleeper, "is_exit", False):
                return
            self.board.run()
            if hasattr(self.sleeper, "scale_sleep"):
                delay = self.sleeper.scale_sleep.get()
                if delay and delay > 0:
                    time.sleep(float(delay))

    def modelling(self):
        if self.board is None:
            print("Warning: board is None")
            return

        if self.theory is None or len(self.theory) == 0:
            self._modelling()
            return

        # STEPS_PER_TRIAL "steps" of run(), one draw
        self._forward_steps_for_theory()
        self.draw()

    @staticmethod
    def hist_compare(theory, result) -> float:
        if not theory or not result:
            return float('inf')

        sorted_theory = sorted(theory.items(), key=lambda x: x[0])
        sorted_result = sorted(result.items(), key=lambda x: x[0])
        values_theory = list(i[1] for i in sorted_theory)
        values_result = list(i[1] for i in sorted_result)

        eps = set()
        for i in range(len(values_result) - len(values_theory) + 1):
            eps.add(
                sum((values_result[j + i] ** 2 - values_theory[j] ** 2 for j in range(len(theory))))
            )
        eps = min(eps) if eps else 0

        eps = sqrt(abs(eps))

        koeff1 = polyfit(x=list(result.keys()), y=list(result.values()), deg=1)
        koeff2 = polyfit(x=list(theory.keys()), y=list(theory.values()), deg=1)

        eps = abs(koeff1[1] - koeff2[1]) * eps

        return eps

    @staticmethod
    def _hist_compare_int(theory, result) -> float:
        """Comparison with integer keys (old algorithm)"""
        try:
            sorted_theory = sorted(theory.items(), key=lambda x: x[0])
            sorted_result = sorted(result.items(), key=lambda x: x[0])
            values_theory = list(i[1] for i in sorted_theory)
            values_result = list(i[1] for i in sorted_result)

            eps = set()
            for i in range(len(values_result) - len(values_theory) + 1):
                eps.add(
                    sum((values_result[j + i] ** 2 - values_theory[j] ** 2 for j in range(len(theory))))
                )
            eps = min(eps) if eps else 0

            eps = sqrt(abs(eps))

            koeff1 = polyfit(x=list(result.keys()), y=list(result.values()), deg=1)
            koeff2 = polyfit(x=list(theory.keys()), y=list(theory.values()), deg=1)

            eps = abs(koeff1[1] - koeff2[1]) * eps

            return eps
        except Exception as e:
            print(f"Error in _hist_compare_int: {e}")
            return float('inf')

    @staticmethod
    def _hist_compare_float(theory, result) -> float:
        """
        Comparison with float keys via interpolation
        """
        try:
            from numpy import interp

            # Sort theory
            sorted_theory = sorted(theory.items(), key=lambda x: x[0])
            theory_keys = [k for k, v in sorted_theory]
            theory_values = [v for k, v in sorted_theory]

            # Sort result
            sorted_result = sorted(result.items(), key=lambda x: x[0])
            result_keys = [k for k, v in sorted_result]
            result_values = [v for k, v in sorted_result]

            if not theory_keys or not result_keys:
                return float('inf')

            # Normalize the values
            theory_sum = sum(theory_values)
            result_sum = sum(result_values)

            if theory_sum == 0 or result_sum == 0:
                return float('inf')

            theory_norm = [v / theory_sum for v in theory_values]
            result_norm = [v / result_sum for v in result_values]

            # Interpolate theory onto the result keys
            interpolated_theory = []
            for r_key in result_keys:
                if r_key <= theory_keys[0]:
                    interpolated_theory.append(theory_norm[0])
                elif r_key >= theory_keys[-1]:
                    interpolated_theory.append(theory_norm[-1])
                else:
                    # Linear interpolation
                    idx = 0
                    while idx < len(theory_keys) - 1 and theory_keys[idx + 1] < r_key:
                        idx += 1

                    if idx >= len(theory_keys) - 1:
                        interpolated_theory.append(theory_norm[-1])
                    else:
                        x0, x1 = theory_keys[idx], theory_keys[idx + 1]
                        y0, y1 = theory_norm[idx], theory_norm[idx + 1]
                        interpolated = y0 + (y1 - y0) * (r_key - x0) / (x1 - x0)
                        interpolated_theory.append(interpolated)

            # Compute the error
            eps = 0.0
            for i in range(min(len(interpolated_theory), len(result_norm))):
                eps += (interpolated_theory[i] - result_norm[i]) ** 2

            eps = sqrt(eps)

            # Trend slopes
            try:
                koeff1 = polyfit(x=result_keys, y=result_norm, deg=1)
                koeff2 = polyfit(x=result_keys, y=interpolated_theory, deg=1)
                eps = abs(koeff1[1] - koeff2[1]) * eps
            except Exception:
                pass

            return eps
        except Exception as e:
            print(f"Error in _hist_compare_float: {e}")
            return float('inf')