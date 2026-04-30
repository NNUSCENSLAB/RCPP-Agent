# -*- coding: utf-8 -*-
"""
text parsers and scene-semantic memory flattening for MCDA tools.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.registry import register_tool


def extract_traffic_total(commuting_flow_text: str) -> float:
    """
    Read the total flow value from commuting flow text for social demand scoring.
    """
    match = re.search(r"Total[:\s]+(\d+)", commuting_flow_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def extract_traffic_outflow(commuting_flow_text: str) -> float:
    """
    Read outflow from text or derive it as total minus inflow when labels exist.
    """
    match = re.search(r"Outflow[:\s]+(\d+)", commuting_flow_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    total_match = re.search(r"Total[:\s]+(\d+)", commuting_flow_text, re.IGNORECASE)
    inflow_match = re.search(r"Inflow[:\s]+(\d+)", commuting_flow_text, re.IGNORECASE)
    if total_match and inflow_match:
        total = float(total_match.group(1))
        inflow = float(inflow_match.group(1))
        return total - inflow
    return 0.0


def extract_poi_counts(functional_zone_text: str) -> Tuple[int, int]:
    """
    Read residential and commercial POI counts from zone text for zone type rules.
    """
    p_res = 0
    p_comm = 0
    res_match = re.search(r"\$P_\{res\}=(\d+)", functional_zone_text)
    comm_match = re.search(r"\$P_\{comm\}=(\d+)", functional_zone_text)
    if res_match:
        p_res = int(res_match.group(1))
    if comm_match:
        p_comm = int(comm_match.group(1))
    return p_res, p_comm


def determine_zone_type_and_rcp_prob(p_res: int, p_comm: int) -> Tuple[str, float]:
    """
    Decide residential commercial or mixed from two counts and return a simple use probability.
    """
    if p_res > p_comm * 1.5:
        return "residential", 0.25
    if p_comm > p_res * 1.5:
        return "commercial", 0.55
    return "mixed", 0.40


def extract_grid_distance(grid_accessibility_text: str) -> float:
    """
    Read grid distance in meters from grid accessibility text for technical and economic scores.
    """
    match = re.search(r"Dist=(\d+(?:\.\d+)?)\s*m", grid_accessibility_text)
    if match:
        return float(match.group(1))
    return 0.0


def extract_sensitive_risk(sensitive_constraints_text: str) -> Tuple[str, int]:
    """
    Read risk label and sensitive facility count from constraint text for policy scoring.
    """
    risk_level = "unknown"
    sensitive_count = 0
    if "No Risk" in sensitive_constraints_text or "No risk" in sensitive_constraints_text:
        risk_level = "no_risk"
    elif "Low Risk" in sensitive_constraints_text:
        risk_level = "low_risk"
    elif "Risk Warning" in sensitive_constraints_text:
        risk_level = "high_risk"
    count_match = re.search(r"Sensitive Count[:\s]+(\d+)", sensitive_constraints_text, re.IGNORECASE)
    if count_match:
        sensitive_count = int(count_match.group(1))
    return risk_level, sensitive_count


def split_scene_semantic_from_rps_record(mem: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    From Environment Perception Agent output, return scene_memory.
    """
    node = mem.get("memory_node", mem)
    if not isinstance(node, dict):
        return {}, {}
    scene = dict(node.get("scene_memory") or {})
    sem_raw = node.get("semantic_memory")
    if not isinstance(sem_raw, dict):
        sem_raw = {}
    semantic_for_eval: Dict[str, Any] = dict(sem_raw)
    sr = semantic_for_eval.get("semantic_reasoning")
    if isinstance(sr, dict):
        for key in ("grid_accessibility", "functional_zone_type", "commuting_flow", "sensitive_constraints"):
            if key in sr:
                semantic_for_eval.setdefault(key, sr[key])
    for key in ("grid_accessibility", "functional_zone_type", "commuting_flow", "sensitive_constraints"):
        v = semantic_for_eval.get(key, "")
        if v is None:
            semantic_for_eval[key] = ""
        elif not isinstance(v, str):
            semantic_for_eval[key] = str(v)
    return scene, semantic_for_eval


@register_tool
class SceneSemanticSplitTool(BaseTool):
    name = "scene_semantic_split"
    category = ToolCategory.MCDA_INDICATORS

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        rps_record: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Return scene and semantic dicts and optionally store them on the tool context.
        """
        scene, semantic = split_scene_semantic_from_rps_record(rps_record)
        if ctx is not None:
            ctx.set("split_scene_memory", scene)
            ctx.set("split_semantic_memory", semantic)
        return {"scene_memory": scene, "semantic_memory": semantic}
