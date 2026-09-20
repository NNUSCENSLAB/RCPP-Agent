# -*- coding: utf-8 -*-
"""Validation and version metadata for long-term memory records."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

MEMORY_SCHEMA_VERSION = "2.0"

SCENE_REQUIRED_FIELDS = {
    "has_existing_RCP",
    "is_functional_zone",
    "functional_zone_type",
    "has_ground_obstacle",
    "ground_obstacle_types",
    "ground_obstacle_count",
    "visual_distractors_noted",
    "clearance_visual_assessment",
    "scene_reasoning",
    "confidence_score",
}
SEMANTIC_REQUIRED_FIELDS = {
    "rps_id",
    "functional_zone_type",
    "commuting_flow",
    "sensitive_constraints",
    "grid_accessibility",
}


def validate_scene_memory(memory: Mapping[str, Any]) -> list[str]:
    errors = [f"missing field: {name}" for name in sorted(SCENE_REQUIRED_FIELDS - memory.keys())]
    count = memory.get("ground_obstacle_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        errors.append("ground_obstacle_count must be a non-negative integer")
    score = memory.get("confidence_score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not 1 <= score <= 5:
        errors.append("confidence_score must be between 1 and 5")
    obstacle = memory.get("has_ground_obstacle")
    if obstacle is False and count not in (0, None):
        errors.append("has_ground_obstacle=false requires ground_obstacle_count=0")
    if obstacle is True and count == 0:
        errors.append("has_ground_obstacle=true requires ground_obstacle_count>0")
    if memory.get("ground_obstacle_types") == "none" and count not in (0, None):
        errors.append("ground_obstacle_types=none requires ground_obstacle_count=0")
    return errors


def validate_semantic_memory(memory: Mapping[str, Any]) -> list[str]:
    errors = [f"missing field: {name}" for name in sorted(SEMANTIC_REQUIRED_FIELDS - memory.keys())]
    for name in SEMANTIC_REQUIRED_FIELDS - {"rps_id"}:
        value = memory.get(name)
        if not isinstance(value, str) or not value.strip() or value.strip().lower() == "null":
            errors.append(f"{name} must be a non-empty, non-null string")
    return errors


def quality_score(kind: str, errors: list[str], memory: Mapping[str, Any]) -> float:
    if errors:
        return max(0.0, round(1.0 - 0.2 * len(errors), 3))
    if kind == "scene":
        confidence = float(memory.get("confidence_score", 3))
        return round(min(1.0, max(0.0, confidence / 5.0)), 3)
    return 1.0


def build_memory_metadata(
    *,
    base_model: str,
    adapter_version: str,
    quality: float,
    evidence_source: str,
    generated_at: Optional[str] = None,
    status: str = "active",
    expires_at: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "base_model": base_model,
        "adapter_version": adapter_version,
        "schema_version": MEMORY_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "quality_score": round(float(quality), 3),
        "evidence_source": evidence_source,
        "status": status,
        "expires_at": expires_at,
    }
