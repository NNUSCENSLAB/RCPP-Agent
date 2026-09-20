"""Dependency-light, versioned AHP policy used by offline CLI runs."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from configs.config import MCDA_DIMENSIONS, PlanningPhase

SCENARIO_WEIGHTS: Dict[str, Dict[str, float]] = {
    "efficiency_oriented": {
        "technical": 0.28, "economic": 0.27, "social": 0.10,
        "traffic": 0.25, "policy": 0.10,
    },
    "equity_oriented": {
        "technical": 0.18, "economic": 0.12, "social": 0.32,
        "traffic": 0.16, "policy": 0.22,
    },
    "balance_oriented": {
        "technical": 0.22, "economic": 0.20, "social": 0.20,
        "traffic": 0.20, "policy": 0.18,
    },
}


def _normalize(values: Mapping[str, float]) -> Dict[str, float]:
    selected = {key: float(values.get(key, 0.0)) for key in MCDA_DIMENSIONS}
    total = sum(selected.values())
    if total <= 0:
        return {key: 1.0 / len(MCDA_DIMENSIONS) for key in MCDA_DIMENSIONS}
    return {key: value / total for key, value in selected.items()}


def _phase(value: PlanningPhase | str) -> PlanningPhase:
    if isinstance(value, PlanningPhase):
        return value
    try:
        return PlanningPhase[value]
    except KeyError:
        for member in PlanningPhase:
            if value == member.value:
                return member
    raise ValueError(f"Unknown planning phase: {value!r}")


def cached_ahp_weights(
    kind: str,
    scenario: str,
    scenario_weights: Mapping[str, float] | None = None,
    phase: PlanningPhase | str | None = None,
) -> Dict[str, Any]:
    if kind == "scenario":
        weights = SCENARIO_WEIGHTS.get(scenario, SCENARIO_WEIGHTS["balance_oriented"])
    elif kind == "phase":
        if phase is None:
            raise ValueError("phase is required for cached phase weights")
        base = _normalize(scenario_weights or {})
        boost = {
            PlanningPhase.INITIAL_EXPLORATION: {"technical": 1.20, "economic": 1.10},
            PlanningPhase.RAPID_DEVELOPMENT: {"traffic": 1.20, "economic": 1.10},
            PlanningPhase.MATURITY_OPTIMIZATION: {"social": 1.15, "policy": 1.15},
        }[_phase(phase)]
        weights = _normalize({key: value * boost.get(key, 1.0) for key, value in base.items()})
    else:
        raise ValueError(f"Unknown AHP kind: {kind!r}")
    return {
        "weights": dict(weights),
        "pairwise_matrix": None,
        "consistency_ratio": None,
        "reasoning": "Versioned cached AHP policy; no external model call.",
        "source": "cached",
    }
