# -*- coding: utf-8 -*-
"""Review Agent metrics and data loading BaseTool registration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.registry import register_tool
from tools.review_tool.common import scored_candidates_to_planning_records
from tools.review_tool.data_loader import ReviewDataLoader
from tools.review_tool.dcr_metric_tool import DemandCoverageRateCalculator
from tools.review_tool.eb_metric_tool import EconomicBenefitsCalculator
from tools.review_tool.nle_metric_tool import NetworkLossEfficiencyCalculator
from tools.review_tool.scr_metric_tool import ServiceCoverageRateCalculator


def _review_inputs(
    scored_candidates: List[Dict[str, Any]],
    scene_semantic_memory: Dict[str, Any],
    poi_dir: str,
) -> Tuple[Any, ...]:
    """
    Build planning records pass through scene_semantic_memory and load POI frames for review tools.
    """
    planning = scored_candidates_to_planning_records(scored_candidates)
    loader = ReviewDataLoader(Path(poi_dir))
    poi_gdf = loader.load_poi_data()
    grid_gdf = loader.load_grid_points(poi_gdf)
    return planning, scene_semantic_memory, poi_gdf, grid_gdf


@register_tool
class ReviewSCRTool(BaseTool):
    name = "review_scr"
    category = ToolCategory.REVIEW

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        scored_candidates: List[Dict[str, Any]],
        scene_semantic_memory: Dict[str, Any],
        poi_dir: str,
        **kwargs: Any,
    ) -> float:
        """
        Compute Service Coverage Rate (SCR) for Review Agent.
        """
        planning, scene_semantic_memory, poi_gdf, grid_gdf = _review_inputs(
            scored_candidates, scene_semantic_memory, poi_dir
        )
        score = float(
            ServiceCoverageRateCalculator().calculate(
                planning, scene_semantic_memory, poi_gdf, grid_gdf
            )
        )
        if ctx is not None:
            ctx.set("scr", score)
        return score


@register_tool
class ReviewDCRTool(BaseTool):
    name = "review_dcr"
    category = ToolCategory.REVIEW

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        scored_candidates: List[Dict[str, Any]],
        scene_semantic_memory: Dict[str, Any],
        poi_dir: str,
        **kwargs: Any,
    ) -> float:
        """
        Compute Demand Coverage Rate (DCR) for Review Agent.
        """
        planning, scene_semantic_memory, poi_gdf, grid_gdf = _review_inputs(
            scored_candidates, scene_semantic_memory, poi_dir
        )
        score = float(
            DemandCoverageRateCalculator().calculate(
                planning, scene_semantic_memory, poi_gdf, grid_gdf
            )
        )
        if ctx is not None:
            ctx.set("dcr", score)
        return score


@register_tool
class ReviewNLETool(BaseTool):
    name = "review_nle"
    category = ToolCategory.REVIEW

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        scored_candidates: List[Dict[str, Any]],
        scene_semantic_memory: Dict[str, Any],
        poi_dir: str,
        existing_rcp_path: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        """
        Compute Network Loss Efficiency (NLE) for Review Agent.
        """
        planning, scene_semantic_memory, poi_gdf, grid_gdf = _review_inputs(
            scored_candidates, scene_semantic_memory, poi_dir
        )
        er = Path(existing_rcp_path) if existing_rcp_path else None
        score = float(
            NetworkLossEfficiencyCalculator(existing_rcp_path=er).calculate(
                planning, scene_semantic_memory, poi_gdf, grid_gdf
            )
        )
        if ctx is not None:
            ctx.set("nle", score)
        return score


@register_tool
class ReviewEBTool(BaseTool):
    name = "review_eb"
    category = ToolCategory.REVIEW

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        scored_candidates: List[Dict[str, Any]],
        scene_semantic_memory: Dict[str, Any],
        poi_dir: str,
        **kwargs: Any,
    ) -> float:
        """
        Compute Economic Benefits (EB) for Review Agent.
        """
        planning, scene_semantic_memory, poi_gdf, grid_gdf = _review_inputs(
            scored_candidates, scene_semantic_memory, poi_dir
        )
        score = float(
            EconomicBenefitsCalculator().calculate(
                planning, scene_semantic_memory, poi_gdf, grid_gdf
            )
        )
        if ctx is not None:
            ctx.set("eb", score)
        return score
