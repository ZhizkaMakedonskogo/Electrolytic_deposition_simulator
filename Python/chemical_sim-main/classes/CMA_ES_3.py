"""
The same objective function as TPE
"""
from __future__ import annotations

from classes.CMA_ES_1 import create_cma_es_sampler
from classes.optuna_integration import OptunaIntegration
from interfaces.drawer import Drawer
from interfaces.sleeper import Sleeper


class CmaEsOptunaIntegration(OptunaIntegration):
    """Optuna sampler — CmaEsSampler"""

    def __init__(self, rows: int, theory: dict, drawer: Drawer, sleeper: Sleeper):
        super().__init__(
            rows=rows,
            theory=theory,
            drawer=drawer,
            sleeper=sleeper,
            optimizer_label="CMA-ES",
            result_file_prefix="CmaEsOptimization",
        )
        self._sampler = create_cma_es_sampler()