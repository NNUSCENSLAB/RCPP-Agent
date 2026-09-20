# -*- coding: utf-8 -*-
"""Deterministic fact derivation for semantic memory.

The language model is deliberately kept out of threshold calculations.  It may
turn these facts into concise prose, but it must not change the labels or
source values produced here.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Optional


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_statistical_summary(text: str) -> Dict[str, Any]:
    """Extract the first JSON object following ``Statistical Summary``."""
    marker = "Statistical Summary"
    start_at = text.find(marker)
    candidate = text[start_at + len(marker) :] if start_at >= 0 else text
    start = candidate.find("{")
    if start < 0:
        raise ValueError("Statistical Summary JSON was not found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(candidate[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("Statistical Summary must be a JSON object")
                return value
    raise ValueError("Statistical Summary JSON is incomplete")


def _sensitive_count(counts: Mapping[str, Any]) -> float:
    if "sensitive" in counts:
        return _number(counts.get("sensitive"))
    aliases = (
        "medical",
        "education",
        "government_public",
        "public",
        "public_admin",
    )
    return sum(_number(counts.get(key)) for key in aliases)


def derive_semantic_facts(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Calculate canonical labels and retain the exact evidence values."""
    poi = summary.get("poi_counts_1km") or {}
    sensitive = summary.get("poi_counts_sensitive_0.1km") or {}
    traffic = summary.get("traffic_flow_stats") or {}
    infrastructure = summary.get("infrastructure_dist") or {}

    residential = _number(poi.get("residential"))
    commercial = _number(poi.get("commercial"))
    inflow = _number(traffic.get("total_inflow"))
    outflow = _number(traffic.get("total_outflow"))
    total_flow = inflow + outflow
    sensitive_total = _sensitive_count(sensitive)
    grid_km = _number(infrastructure.get("grid_distance_km"), -1.0)
    grid_m: Optional[float] = None if grid_km < 0 else grid_km * 1000.0

    if residential > 1.5 * commercial and residential >= 5:
        zone = "residential_oriented"
    elif commercial > 1.5 * residential and commercial >= 5:
        zone = "commercial_oriented"
    elif residential >= 5 and commercial >= 5:
        zone = "mixed_functional"
    else:
        zone = "low_density"

    if total_flow > 100_000:
        flow = "high"
    elif total_flow >= 50_000:
        flow = "medium"
    else:
        flow = "low"

    if sensitive_total >= 2:
        risk = "high"
    elif sensitive_total > 0:
        risk = "low"
    else:
        risk = "none"

    if grid_m is None:
        grid = "unknown"
    elif grid_m < 500:
        grid = "excellent"
    elif grid_m < 1000:
        grid = "good"
    elif grid_m < 2000:
        grid = "moderate"
    elif grid_m < 5000:
        grid = "poor"
    else:
        grid = "very_poor"

    return {
        "rps_id": str(summary.get("rps_id") or ""),
        "labels": {
            "functional_zone": zone,
            "commuting_flow": flow,
            "sensitive_risk": risk,
            "grid_accessibility": grid,
        },
        "evidence": {
            "residential_count_1km": residential,
            "commercial_count_1km": commercial,
            "total_inflow": inflow,
            "total_outflow": outflow,
            "total_flow": total_flow,
            "sensitive_count_100m": sensitive_total,
            "grid_distance_m": grid_m,
        },
    }


def deterministic_semantic_memory(
    summary: Mapping[str, Any], *, bsv_image: Optional[str] = None
) -> Dict[str, Any]:
    """Build a safe fallback that remains compatible with existing consumers."""
    facts = derive_semantic_facts(summary)
    labels = facts["labels"]
    ev = facts["evidence"]
    image = bsv_image if bsv_image is not None else summary.get("bsv_image")
    zone_text = {
        "residential_oriented": "Residential-Oriented",
        "commercial_oriented": "Commercial-Oriented",
        "mixed_functional": "Mixed Functional",
        "low_density": "Low Density",
    }[labels["functional_zone"]]
    flow_text = {"high": "High Traffic", "medium": "Medium Traffic", "low": "Low Traffic"}[
        labels["commuting_flow"]
    ]
    risk_text = {"high": "High Risk", "low": "Low Risk", "none": "No Risk"}[
        labels["sensitive_risk"]
    ]
    grid_text = {
        "excellent": "Excellent",
        "good": "Good",
        "moderate": "General",
        "poor": "Poor",
        "very_poor": "Very Poor",
        "unknown": "Unknown",
    }[labels["grid_accessibility"]]
    distance_text = "unknown" if ev["grid_distance_m"] is None else f"{ev['grid_distance_m']:g}m"
    return {
        "rps_id": facts["rps_id"],
        "bsv_image": image,
        "functional_zone_type": (
            f"{zone_text}: $P_{{res}}={ev['residential_count_1km']:g}, "
            f"$P_{{comm}}={ev['commercial_count_1km']:g}."
        ),
        "commuting_flow": (
            f"{flow_text}: Total: {ev['total_flow']:g}, "
            f"Inflow: {ev['total_inflow']:g}, Outflow: {ev['total_outflow']:g}."
        ),
        "sensitive_constraints": (
            f"{risk_text}: Sensitive Count: {ev['sensitive_count_100m']:g}."
        ),
        "grid_accessibility": (
            f"{grid_text}: Dist={distance_text}."
        ),
        "rule_facts": facts,
        "evidence_fields": list(ev.keys()),
    }


def validate_semantic_against_rules(
    memory: Mapping[str, Any], summary: Mapping[str, Any]
) -> list[str]:
    """Return contradictions between model prose/labels and deterministic facts."""
    facts = derive_semantic_facts(summary)
    labels = facts["labels"]
    expected = {
        "functional_zone_type": labels["functional_zone"],
        "commuting_flow": labels["commuting_flow"],
        "sensitive_constraints": labels["sensitive_risk"],
        "grid_accessibility": labels["grid_accessibility"],
    }
    errors: list[str] = []
    normalized = lambda value: re.sub(r"[\s-]+", "_", str(value).lower())
    aliases = {
        ("sensitive_constraints", "none"): ("none", "no_risk"),
        ("grid_accessibility", "moderate"): ("moderate", "general"),
    }
    for field, label in expected.items():
        accepted = aliases.get((field, label), (label,))
        if not any(name in normalized(memory.get(field, "")) for name in accepted):
            errors.append(f"{field} does not contain canonical label {label!r}")
    supplied = memory.get("rule_facts")
    if supplied is not None and supplied != facts:
        errors.append("rule_facts differs from deterministic calculation")
    allowed_numbers = {
        float(value)
        for value in facts["evidence"].values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    # Buffer radii are part of the input schema names and may be cited in prose.
    allowed_numbers.update({0.1, 1.0, 100.0, 1000.0})
    for field in expected:
        for token in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", str(memory.get(field, ""))):
            number = float(token)
            if not any(abs(number - allowed) < 1e-9 for allowed in allowed_numbers):
                errors.append(f"{field} contains ungrounded number {token}")
    return errors
