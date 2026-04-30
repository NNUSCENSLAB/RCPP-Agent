# -*- coding: utf-8 -*-
"""
Social dimension: charging demand.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.multi_scenario_evaluation_tool.mcda_text_parsing import (
    determine_zone_type_and_rcp_prob,
    extract_poi_counts,
    extract_traffic_total,
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
    Score charging demand proxy from semantic memory.
    """
    del scene_memory, context, coordinates, rps_id
    ev_penetration_rate = 0.5407
    avg_charging_energy_kwh = 48.0
    daily_charge_probability = 0.5
    max_demand_for_norm = 10000.0
    commuting_flow_text = semantic_memory.get("commuting_flow", "")
    functional_zone_text = semantic_memory.get("functional_zone_type", "")
    flow_total = extract_traffic_total(commuting_flow_text)
    p_res, p_comm = extract_poi_counts(functional_zone_text)
    zone_type, p_use_rcp = determine_zone_type_and_rcp_prob(p_res, p_comm)
    charging_demand_kwh = (
        float(flow_total)
        * float(ev_penetration_rate)
        * float(daily_charge_probability)
        * float(p_use_rcp)
        * float(avg_charging_energy_kwh)
    )
    if max_demand_for_norm > 0:
        demand_score = min(charging_demand_kwh / max_demand_for_norm, 1.0)
    else:
        demand_score = 0.0
    details: Dict[str, Any] = {
        "method": "text_parsing_od_proxy",
        "flow_total": flow_total,
        "p_res": p_res,
        "p_comm": p_comm,
        "zone_type": zone_type,
        "p_use_rcp": p_use_rcp,
        "charging_demand_kwh": round(charging_demand_kwh, 2),
        "ev_penetration_rate": ev_penetration_rate,
        "daily_charge_probability": daily_charge_probability,
        "avg_charging_energy_kwh": avg_charging_energy_kwh,
        "max_demand_for_norm": max_demand_for_norm,
    }
    return demand_score, details


@register_tool
class SocialCalculationTool(BaseTool):
    name = "social_calculation"
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
        Return social MCDA score and optionally save score details on the tool context.
        """
        score, det = evaluate(rps_id, coordinates, scene_memory, semantic_memory, context)
        if ctx is not None:
            ctx.set("social_score", score)
            ctx.set("social_details", det)
        return float(score)
