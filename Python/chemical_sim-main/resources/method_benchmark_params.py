"""
Schema of tunable optimizer parameters
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# type: "int", "float"
ParamSpec = Dict[str, Any]

METHOD_PARAM_SCHEMA: Dict[str, List[ParamSpec]] = {
    "pso": [
        {
            "key": "swarm_size",
            "label": "Swarm size (0 = auto)",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 128,
        },
        {
            "key": "inertia",
            "label": "Inertia w",
            "type": "float",
            "default": 0.73,
            "min": 0.1,
            "max": 1.0,
        },
        {
            "key": "cognitive",
            "label": "Cognitive c1",
            "type": "float",
            "default": 1.496,
            "min": 0.0,
            "max": 4.0,
        },
        {
            "key": "social",
            "label": "Social c2",
            "type": "float",
            "default": 1.496,
            "min": 0.0,
            "max": 4.0,
        },
    ],
    "ga": [
        {
            "key": "population_size",
            "label": "Population (0 = auto)",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 128,
        },
        {
            "key": "tournament_k",
            "label": "Tournament size k",
            "type": "int",
            "default": 3,
            "min": 2,
            "max": 16,
        },
    ],
    "de": [
        {
            "key": "population_size",
            "label": "Population (0 = auto)",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 64,
        },
    ],
    "mfo": [
        {
            "key": "mfo_budget",
            "label": "MFO budget (0 = n_trials)",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 2000.0,
        },
    ],
}

SEED_PRESETS: Tuple[int, ...] = (42, 100, 300, 777)


def methods_with_params() -> List[str]:
    return list(METHOD_PARAM_SCHEMA.keys())


def parse_seed_list(
    preset_seeds: List[int],
    extra_text: str = "",
    *,
    fallback: int = 42,
) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for s in list(preset_seeds):
        si = int(s)
        if si not in seen:
            seen.add(si)
            out.append(si)
    text = str(extra_text or "").strip()
    if text:
        for part in re.split(r"[,;\s]+", text):
            part = part.strip()
            if not part:
                continue
            try:
                si = int(part)
            except ValueError:
                continue
            if si not in seen:
                seen.add(si)
                out.append(si)
    return out if out else [int(fallback)]


def default_method_options(method_key: str) -> Dict[str, Any]:
    opts: Dict[str, Any] = {}
    for spec in METHOD_PARAM_SCHEMA.get(method_key, []):
        opts[spec["key"]] = spec["default"]
    return opts


def coerce_method_options(method_key: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for spec in METHOD_PARAM_SCHEMA.get(method_key, []):
        key = spec["key"]
        if key not in raw:
            continue
        val = raw[key]
        if spec["type"] == "int":
            out[key] = int(val)
        else:
            out[key] = float(val)
    return out