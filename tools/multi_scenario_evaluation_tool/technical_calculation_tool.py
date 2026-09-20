# -*- coding: utf-8 -*-
"""
Technical dimension: grid accessibility.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.multi_scenario_evaluation_tool.mcda_text_parsing import extract_grid_distance
from tools.registry import register_tool


def evaluate(
    rps_id: str,
    coordinates: List[float],
    scene_memory: Dict[str, Any],
    semantic_memory: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Score grid reach from semantic memory.
    """
    del scene_memory, context, coordinates, rps_id
    grid_accessibility_text = semantic_memory.get("grid_accessibility", "")
    evidence = ((semantic_memory.get("rule_facts") or {}).get("evidence") or {})
    raw_distance = evidence.get("grid_distance_m")
    distance = float(raw_distance) if raw_distance is not None else extract_grid_distance(grid_accessibility_text)
    capacity_score = max(0.0, 1.0 - distance / 1000.0)
    details: Dict[str, Any] = {
        "grid_distance_m": distance,
        "grid_accessibility_text": grid_accessibility_text[:200],
        "assessment": _assess_grid_accessibility(distance),
    }
    return capacity_score, details


def _assess_grid_accessibility(distance: float) -> str:
    """
    Map distance in meters to a short text label for technical score details.
    """
    if distance < 500:
        return "Excellent: close distance, low access cost"
    if distance < 1000:
        return "Good: moderate distance, feasible access"
    if distance < 2000:
        return "Average: far distance, need to evaluate cost"
    if distance < 5000:
        return "Poor: far distance, high access cost"
    return "Very poor: far distance, access difficult"


@register_tool
class TechnicalCalculationTool(BaseTool):
    name = "technical_calculation"
    category = ToolCategory.MCDA_INDICATORS

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        rps_id: str,
        coordinates: List[float],
        scene_memory: Dict[str, Any],
        semantic_memory: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> float:
        """
        Return technical MCDA score and optionally save score details on the tool context.
        """
        score, det = evaluate(rps_id, coordinates, scene_memory, semantic_memory, context)
        if ctx is not None:
            ctx.set("technical_score", score)
            ctx.set("technical_details", det)
        return float(score)
