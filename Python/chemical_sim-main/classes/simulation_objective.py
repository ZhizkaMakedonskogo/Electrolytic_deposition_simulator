"""
Shared black-box function: maps (add, transit, merge) to a scalar loss via a Board simulation.
Uses averaging over several seeds to reduce stochastic noise
(the same loss definition as in the Optuna TPE path).

Forward model: simulator parameters theta = (p_add, p_transit, p_merge) in [0, 1]^3
are run for ``steps`` lattice updates; the resulting cluster-size histogram H_hat
is compared with the **target** histogram H* (the ``theory`` dictionary: support sizes -> counts
or weights, normalized inside the loss function).

Inverse problem (project formulation)
Given fixed H*, find argmin_theta E[L(H*, H_hat(theta, xi))],
where xi is the simulation stochasticity (random seeds in the lattice dynamics). The expectation
is approximated by averaging ``n_repeats`` independent runs with deterministic seeds
obtained from ``deterministic_seed(eval_index, rep)``.
``weighted_hist_compare`` is a weighted mean-squared error over normalized
bin probabilities, with a larger weight for large clusters, plus a small penalty for
a difference in the number of non-empty bins (see the implementation below). The factor 100 scales the estimate
for readability.

"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, Final, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from classes.board import Board
from interfaces.sleeper import Sleeper
from resources.constants import Mode

PARAM_LOW = 0.0
PARAM_HIGH = 1.0

STEPS_PER_TRIAL = 200
DEFAULT_N_REPEATS = 3
DEFAULT_LHS_SEED = 42

METRIC_WEIGHTED_L = "weighted_L"
METRIC_COMBINED_L_NEW = "combined_L_new"
METRIC_RSH_NOISE_ROBUST = "rsh_noise_robust"
METRIC_CHOICES = (METRIC_WEIGHTED_L, METRIC_COMBINED_L_NEW, METRIC_RSH_NOISE_ROBUST)
DEFAULT_METRIC_NAME = METRIC_WEIGHTED_L
_ACTIVE_METRIC_NAME = DEFAULT_METRIC_NAME

# Upper bound when fitting surrogates (sklearn / skopt)
SURROGATE_FIT_LOSS_MAX = 1.0e8


def loss_scalar_for_surrogate_fit(loss: float) -> float:
    """
    Scalar loss for GPR, RF, skopt fit — the same semantic quantity, but finite.
    The actual loss values in the logs are still weighted_hist_compare.
    """
    if loss is None:
        return float(SURROGATE_FIT_LOSS_MAX)
    x = float(loss)
    if not np.isfinite(x) or x > SURROGATE_FIT_LOSS_MAX:
        return float(SURROGATE_FIT_LOSS_MAX)
    return max(0.0, x)


def numpy_losses_for_surrogate_fit(y) -> np.ndarray:
    """Vector y before rf.fit / gpr.fit when finite and infinite losses are mixed"""
    a = np.asarray(y, dtype=np.float64)
    a = np.nan_to_num(a, posinf=SURROGATE_FIT_LOSS_MAX, neginf=SURROGATE_FIT_LOSS_MAX, nan=SURROGATE_FIT_LOSS_MAX)
    return np.clip(a, 0.0, float(SURROGATE_FIT_LOSS_MAX))


def weighted_hist_compare(theory: dict, experimental: dict) -> float:
    """The same weighted distribution loss function as in OptunaIntegration"""
    if not theory or not experimental:
        return float("inf")

    total_theory = sum(theory.values())
    total_exp = sum(experimental.values())

    if total_theory == 0 or total_exp == 0:
        return float("inf")

    all_sizes = set(theory.keys()) | set(experimental.keys())
    max_size = max(all_sizes) if all_sizes else 1

    loss = 0.0
    weight_sum = 0.0

    for size in all_sizes:
        t_val = theory.get(size, 0) / total_theory
        e_val = experimental.get(size, 0) / total_exp
        weight = size / max_size if max_size > 0 else 1.0
        diff = (t_val - e_val) ** 2
        loss += weight * diff
        weight_sum += weight

    if weight_sum > 0:
        loss /= weight_sum

    cluster_count_diff = abs(len(theory) - len(experimental)) / max(len(theory), len(experimental), 1)
    loss += 0.1 * cluster_count_diff

    return loss * 100


def _normalized_prob_vectors(theory: dict, experimental: dict) -> Tuple[np.ndarray, np.ndarray]:
    if not theory or not experimental:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    total_theory = float(sum(theory.values()))
    total_exp = float(sum(experimental.values()))
    if total_theory <= 0.0 or total_exp <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    all_sizes = sorted(set(theory.keys()) | set(experimental.keys()), key=lambda x: float(x))
    p_t = np.asarray([float(theory.get(k, 0.0)) / total_theory for k in all_sizes], dtype=np.float64)
    p_e = np.asarray([float(experimental.get(k, 0.0)) / total_exp for k in all_sizes], dtype=np.float64)
    p_t = np.clip(p_t, 0.0, 1.0)
    p_e = np.clip(p_e, 0.0, 1.0)
    st = float(np.sum(p_t))
    se = float(np.sum(p_e))
    if st <= 0.0 or se <= 0.0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return p_t / st, p_e / se


def sqrt_js_distance(theory: dict, experimental: dict, eps: float = 1e-12) -> float:
    p_t, p_e = _normalized_prob_vectors(theory, experimental)
    if p_t.size == 0:
        return float("inf")
    m = 0.5 * (p_t + p_e)
    kl_tm = np.sum(p_t * np.log((p_t + eps) / (m + eps)))
    kl_em = np.sum(p_e * np.log((p_e + eps) / (m + eps)))
    js = max(0.0, 0.5 * float(kl_tm + kl_em))
    return float(np.sqrt(js))


def wasserstein1_norm(theory: dict, experimental: dict) -> float:
    p_t, p_e = _normalized_prob_vectors(theory, experimental)
    if p_t.size == 0:
        return float("inf")
    if p_t.size == 1:
        return 0.0
    c_t = np.cumsum(p_t)
    c_e = np.cumsum(p_e)
    w1 = float(np.sum(np.abs(c_t - c_e)))
    return max(0.0, w1 / float(p_t.size - 1))


def combined_hist_compare(theory: dict, experimental: dict) -> float:
    d_js = sqrt_js_distance(theory, experimental)
    d_w1 = wasserstein1_norm(theory, experimental)
    if not np.isfinite(d_js) or not np.isfinite(d_w1):
        return float("inf")
    # L_new in [0, 100], lower is better; smoother than a weighted-MSE-style loss.
    return float((0.6 * d_js + 0.4 * d_w1) * 100.0)


def rsh_noise_robust_hist_compare(
    theory: dict,
    experimental: dict,
    *,
    alpha: float = 5e-3,
    mix_w1: float = 0.6,
) -> float:
    p_t, p_e = _normalized_prob_vectors(theory, experimental)
    if p_t.size == 0:
        return float("inf")
    k = float(p_t.size)
    # Pseudo-count smoothing to suppress stochastic outliers in the tail
    p_t = (p_t + alpha) / (1.0 + alpha * k)
    p_e = (p_e + alpha) / (1.0 + alpha * k)

    if p_t.size == 1:
        w1_norm = 0.0
    else:
        c_t = np.cumsum(p_t)
        c_e = np.cumsum(p_e)
        w1_norm = float(np.sum(np.abs(c_t - c_e)) / float(p_t.size - 1))

    hellinger_sq = 0.5 * float(np.sum((np.sqrt(p_t) - np.sqrt(p_e)) ** 2))
    mix = float(np.clip(mix_w1, 0.0, 1.0))
    val = mix * w1_norm + (1.0 - mix) * hellinger_sq
    return float(max(0.0, val) * 100.0)


_METRIC_FUNCTIONS: Final[Dict[str, Callable[[dict, dict], float]]] = {
    METRIC_COMBINED_L_NEW: combined_hist_compare,
    METRIC_RSH_NOISE_ROBUST: rsh_noise_robust_hist_compare,
}


def metric_function_by_name(name: Optional[str]) -> Callable[[dict, dict], float]:
    if name is None:
        return weighted_hist_compare
    return _METRIC_FUNCTIONS.get(name, weighted_hist_compare)


def set_active_metric(name: str) -> str:
    global _ACTIVE_METRIC_NAME
    _ACTIVE_METRIC_NAME = name if name in METRIC_CHOICES else DEFAULT_METRIC_NAME
    return _ACTIVE_METRIC_NAME


def get_active_metric() -> str:
    return _ACTIVE_METRIC_NAME


def get_active_metric_function() -> Callable[[dict, dict], float]:
    return metric_function_by_name(_ACTIVE_METRIC_NAME)


def deterministic_seed(eval_index: int, rep: int) -> int:
    """Stable, collision-resistant seed from (trial index, repeat index)."""
    return int(eval_index) * 7919 + int(rep) * 104729 + 17


def latin_hypercube_initial(n_points: int, seed: int, low: float = PARAM_LOW, high: float = PARAM_HIGH) -> np.ndarray:
    """Initial space-filling LHS."""
    if n_points <= 0:
        return np.zeros((0, 3), dtype=float)
    d = 3
    lhs = qmc.LatinHypercube(d=d, seed=int(seed))
    u = lhs.random(n_points)
    lo = np.full(d, low)
    hi = np.full(d, high)
    return qmc.scale(u, lo, hi)


def default_n_init(n_trials: int) -> int:
    """The same heuristic as in GPBayesianOptimization (bounded LHS size, not above the budget)"""
    if n_trials <= 0:
        return 0
    n = max(3, min(8, max(3, n_trials // 2)))
    return min(n, n_trials)


class SimulationObjective:
    """Runs Board for STEPS_PER_TRIAL steps; averages the loss over n_repeats RNG seeds"""

    def __init__(
        self,
        rows: int,
        theory: dict,
        sleeper: Sleeper,
        hist_compare_fn: Optional[Callable[[dict, dict], float]] = None,
    ):
        self.rows = rows
        self.theory = theory
        self.sleeper = sleeper
        if hist_compare_fn is None or (
            hist_compare_fn is weighted_hist_compare and get_active_metric() != METRIC_WEIGHTED_L
        ):
            self.hist_compare = get_active_metric_function()
        else:
            self.hist_compare = hist_compare_fn

    def run_one_repeat(
        self,
        add: float,
        transit: float,
        merge: float,
        *,
        eval_index: int,
        rep: int,
        steps: int = STEPS_PER_TRIAL,
    ) -> Tuple[float, Dict[Any, int]]:
        """One stochastic repeat for Hyperband"""
        if steps <= 0:
            raise ValueError("steps must be >= 1")
        if self.sleeper.is_exit:
            return float("inf"), {}

        random.seed(deterministic_seed(eval_index, rep))
        board = Board(
            rows=self.rows,
            addprob=add,
            transitprob=transit,
            mergeprob=merge,
            current_mode=Mode.CONST,
        )
        for _ in range(steps):
            if self.sleeper.is_exit:
                return float("inf"), {}
            board.run()
            if hasattr(self.sleeper, "scale_sleep"):
                delay = self.sleeper.scale_sleep.get()
                if delay > 0:
                    time.sleep(delay)

        experimental_bar = board.create_bar()
        loss = self.hist_compare(self.theory, experimental_bar)
        return float(loss), experimental_bar

    def evaluate(
        self,
        add: float,
        transit: float,
        merge: float,
        *,
        eval_index: int,
        n_repeats: int = DEFAULT_N_REPEATS,
        steps: int = STEPS_PER_TRIAL,
    ) -> Tuple[float, Dict[Any, int]]:
        if n_repeats <= 0:
            raise ValueError("n_repeats must be >= 1")
        if steps <= 0:
            raise ValueError("steps must be >= 1")

        losses: list[float] = []
        hists: list[Dict[Any, int]] = []

        for rep in range(n_repeats):
            loss, experimental_bar = self.run_one_repeat(
                add, transit, merge, eval_index=eval_index, rep=rep, steps=steps
            )
            if self.sleeper.is_exit:
                return float("inf"), {}
            losses.append(loss)
            hists.append(experimental_bar)

        mean_loss = float(sum(losses) / len(losses))
        i_best = int(np.argmin(np.asarray(losses, dtype=np.float64)))
        return mean_loss, hists[i_best] if hists else {}