# -*- coding: utf-8 -*-
"""
Economic dimension: cost / benefit proxy.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.multi_scenario_evaluation_tool.mcda_text_parsing import (
    determine_zone_type_and_rcp_prob,
    extract_grid_distance,
    extract_poi_counts,
)
from tools.registry import register_tool


def evaluate(
    rps_id: str,
    coordinates: List[float],
    scene_memory: Dict[str, Any],
    semantic_memory: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Score cost and benefit proxy from semantic memory.
    """
    del scene_memory, context, coordinates, rps_id
    functional_zone_text = semantic_memory.get("functional_zone_type", "")
    evidence = ((semantic_memory.get("rule_facts") or {}).get("evidence") or {})
    parsed_res, parsed_comm = extract_poi_counts(functional_zone_text)
    p_res = int(evidence.get("residential_count_1km", parsed_res))
    p_comm = int(evidence.get("commercial_count_1km", parsed_comm))
    zone_type, _ = determine_zone_type_and_rcp_prob(p_res, p_comm)
    grid_accessibility_text = semantic_memory.get("grid_accessibility", "")
    raw_distance = evidence.get("grid_distance_m")
    grid_distance = float(raw_distance) if raw_distance is not None else extract_grid_distance(grid_accessibility_text)
    zone_cost_factors = {"residential": 0.7, "commercial": 0.9, "mixed": 0.8}
    zone_factor = zone_cost_factors.get(zone_type, 0.8)
    if grid_distance < 1000:
        grid_factor = 1.0
    elif grid_distance < 3000:
        grid_factor = 0.8
    elif grid_distance < 5000:
        grid_factor = 0.6
    else:
        grid_factor = 0.4
    cost_benefit_score = zone_factor * 0.6 + grid_factor * 0.4
    details: Dict[str, Any] = {
        "zone_type": zone_type,
        "p_res": p_res,
        "p_comm": p_comm,
        "zone_cost_factor": zone_factor,
        "grid_distance_m": grid_distance,
        "grid_cost_factor": grid_factor,
        "cost_benefit_score": round(cost_benefit_score, 4),
    }
    return cost_benefit_score, details


@register_tool
class EconomicCalculationTool(BaseTool):
    name = "economic_calculation"
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
        Return economic MCDA score and optionally save score details on the tool context.
        """
        score, det = evaluate(rps_id, coordinates, scene_memory, semantic_memory, context)
        if ctx is not None:
            ctx.set("economic_score", score)
            ctx.set("economic_details", det)
        return float(score)
