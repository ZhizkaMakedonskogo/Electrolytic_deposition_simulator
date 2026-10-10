"""
CMA-ES via Optuna(sampler factory + search bounds)

Used together with CMA_ES_2 — CMA_ES_4.
"""
from __future__ import annotations

from dataclasses import dataclass

from optuna.samplers import CmaEsSampler

from classes.simulation_objective import PARAM_HIGH, PARAM_LOW

# The same bounds as in OptunaIntegration 
ADD_LOW = PARAM_LOW
ADD_HIGH = PARAM_HIGH
TRANSIT_LOW = PARAM_LOW
TRANSIT_HIGH = PARAM_HIGH
MERGE_LOW = PARAM_LOW
MERGE_HIGH = PARAM_HIGH

DEFAULT_SEED = 42
DEFAULT_N_STARTUP_TRIALS = 4
DEFAULT_SIGMA0 = 0.2


@dataclass(frozen=True)
class CmaEsSearchBounds:
    add: tuple[float, float] = (ADD_LOW, ADD_HIGH)
    transit: tuple[float, float] = (TRANSIT_LOW, TRANSIT_HIGH)
    merge: tuple[float, float] = (MERGE_LOW, MERGE_HIGH)


def create_cma_es_sampler(
    *,
    seed: int = DEFAULT_SEED,
    n_startup_trials: int = DEFAULT_N_STARTUP_TRIALS,
    sigma0: float = DEFAULT_SIGMA0,
) -> CmaEsSampler:
    return CmaEsSampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        sigma0=sigma0,
    )