from enum import Enum


class Algorithm(str, Enum):
    DEFAULT = "Normal run"
    DOWNHILL = "Coordinate descent"
    TPE = "TPE"
    XNES = "xNES"
    CMA_ES = "CMA-ES (Optuna)"
    GP_BAYES = "GP (Bayesian optimization)"
    SKOPT_GP = "GP (skopt, EI)"
    MULTI_FIDELITY = "Multi-fidelity optimization (MFO)"
    PSO = "PSO (particle swarm)"
    GA = "GA (genetic algorithm)"
    DE = "Differential evolution (DE)"
    TRUST_REGION_BO = "Trust-region BO"
    TURBO_MT = "TuRBO-M (multiple trust regions)"
    HYPERBAND_TPE = "Hyperband + TPE (Optuna)"
    LHS_NM = "LHS + Nelder–Mead"
    RF_EI = "RF + EI"
    CEM = "CEM (cross-entropy)"
    SA = "Simulated annealing (SA)"


# Algorithms for fitting (add, transit, merge) to a loaded target histogram 
ALGORITHMS_INVERSE_PROBLEM: tuple[str, ...] = (
    Algorithm.TPE.value,
    Algorithm.XNES.value,
    Algorithm.CMA_ES.value,
    Algorithm.GP_BAYES.value,
    Algorithm.SKOPT_GP.value,
    Algorithm.MULTI_FIDELITY.value,
    Algorithm.PSO.value,
    Algorithm.GA.value,
    Algorithm.DE.value,
    Algorithm.TRUST_REGION_BO.value,
    Algorithm.TURBO_MT.value,
    Algorithm.HYPERBAND_TPE.value,
    Algorithm.LHS_NM.value,
    Algorithm.RF_EI.value,
    Algorithm.CEM.value,
    Algorithm.SA.value,
)

# Comparison order: coordinate descent + all "inverse-problem" heuristics
ALGORITHMS_MODE_COMPARE_BUNDLE: tuple[str, ...] = (Algorithm.DOWNHILL.value,) + ALGORITHMS_INVERSE_PROBLEM

# Series Q: only the quartet for a large budget
ALGORITHMS_SERIES_Q_BUNDLE: tuple[str, ...] = (
    Algorithm.TPE.value,
    Algorithm.PSO.value,
    Algorithm.GA.value,
    Algorithm.DE.value,
)
SERIES_Q_BENCHMARK_KEYS: tuple[str, ...] = ("tpe", "pso", "ga", "de")


# Require the "Number of optimizations" field / budget (including coordinate descent)
ALGORITHMS_NEED_OPTI_OR_BUDGET: tuple[str, ...] = ALGORITHMS_INVERSE_PROBLEM + (Algorithm.DOWNHILL.value,)

# Do not reset the loaded histogram when switching the combobox between these modes
ALGORITHMS_USE_LOADED_HISTOGRAM: tuple[str, ...] = ALGORITHMS_INVERSE_PROBLEM + (Algorithm.DOWNHILL.value,)


# Base layout 1280×720; global window and panel scaling ×1.5
class UISize(int, Enum):
    WIN_W = 1920
    WIN_H = 1080
    SIDEBAR_W = 300
    SIDEBAR_H = 1080
    GRAPHBAR_W = 600
    GRAPHBAR_H = 540
    STATBAR_W = 600
    STATBAR_H = 540
    CANVAS_W = 1020
    CANVAS_H = 1080


RESIZE = 3


class Mode(str, Enum):
    CONST = "Constant-current"
    VAR = "Alternating-current"


class Status(str, Enum):
    ON_SURFACE = "On the surface"
    LEAVING = "Leaving the surface"
    UP_ALONG_SURFACE = "Moving up along the surface"
    DOWN_ALONG_SURFACE = "Moving down along the surface"
    MERGING = "Undergoing (has undergone) merging with another cluster"
    OFF_SURFACE = "Located off the surface"