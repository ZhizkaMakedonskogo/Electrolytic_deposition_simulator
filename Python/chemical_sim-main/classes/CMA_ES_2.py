from __future__ import annotations

# Optuna trial parameter names (must match suggest_* in objective)
PARAM_ADD = "add"
PARAM_TRANSIT = "transit"
PARAM_MERGE = "merge"
STEPS_PER_TRIAL = 200


def param_keys() -> tuple[str, str, str]:
    return PARAM_ADD, PARAM_TRANSIT, PARAM_MERGE