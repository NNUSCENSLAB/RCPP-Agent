# -*- coding: utf-8 -*-
"""
Policy dimension: compliance with recommendatory clauses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.multi_scenario_evaluation_tool.mcda_text_parsing import extract_sensitive_risk
from tools.registry import register_tool


def evaluate(
    rps_id: str,
    coordinates: List[float],
    scene_memory: Dict[str, Any],
    semantic_memory: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Score policy and sensitive facilities risk from semantic memory.
    """
    del scene_memory, context, coordinates, rps_id
    sensitive_constraints_text = semantic_memory.get("sensitive_constraints", "")
    risk_level, sensitive_count = extract_sensitive_risk(sensitive_constraints_text)
    if risk_level == "no_risk" and sensitive_count == 0:
        policy_constraint_score = 1.0
        policy_assessment = "Fully compliant: no sensitive facilities"
    elif risk_level == "low_risk" and sensitive_count <= 1:
        policy_constraint_score = 0.8
        policy_assessment = "Basic compliant: low risk, few sensitive facilities"
    elif risk_level == "high_risk" or sensitive_count > 1:
        policy_constraint_score = 0.5
        policy_assessment = "Not compliant: high risk, many sensitive facilities"
    else:
        policy_constraint_score = 0.6
        policy_assessment = "Moderate compliant: need to further evaluate"
    policy_score = policy_constraint_score
    details: Dict[str, Any] = {
        "regulation_4_2_5_policy": {
            "score": policy_constraint_score,
            "risk_level": risk_level,
            "sensitive_count": sensitive_count,
            "assessment": policy_assessment,
            "text": sensitive_constraints_text[:200],
        },
        "overall_policy_score": policy_score,
    }
    return policy_score, details


@register_tool
class PolicyCalculationTool(BaseTool):
    name = "policy_calculation"
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
        Return policy MCDA score and optionally save score details on the tool context.
        """
        score, det = evaluate(rps_id, coordinates, scene_memory, semantic_memory, context)
        if ctx is not None:
            ctx.set("policy_score", score)
            ctx.set("policy_details", det)
        return float(score)
