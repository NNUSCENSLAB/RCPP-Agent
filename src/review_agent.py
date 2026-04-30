"""
Review Agent: scores preliminary planning with review metrics and either accepts the result
or readjusts multi-scenario weights when quality bars are missed.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import geopandas as gpd
from dataclasses import dataclass
from datetime import datetime

from tools.review_tool.common import PlanningRecord, scored_candidates_to_planning_records
from tools.review_tool.data_loader import ReviewDataLoader
from tools.review_tool.dcr_metric_tool import DemandCoverageRateCalculator
from tools.review_tool.eb_metric_tool import EconomicBenefitsCalculator
from tools.review_tool.nle_metric_tool import NetworkLossEfficiencyCalculator
from tools.review_tool.scr_metric_tool import ServiceCoverageRateCalculator

from configs.data_config import (
    RCPP_AGENT_MAX_WORKFLOW_ITERATIONS,
    REVIEW_SCORE_THRESHOLD_DEFAULT,
)

logger = logging.getLogger(__name__)


def _pic_basename_to_rps_id(scene_semantic_memory: Dict[str, Any]) -> Dict[str, str]:
    """Map first street-view image basename to memory RPS id (same join as phased shp export)."""
    out: Dict[str, str] = {}
    for rid, entry in scene_semantic_memory.items():
        if not isinstance(entry, dict):
            continue
        imgs = entry.get("street_view_images") or []
        if imgs:
            out[Path(str(imgs[0])).name.strip()] = str(rid)
    return out


def load_merged_phased_planning_records(
    phased_dir: Path,
    scenario: str,
    area: str,
    scene_semantic_memory: Dict[str, Any],
) -> List[PlanningRecord]:
    """
    Union of phase1, phase2, phase3 planning rps from phased decision shp exports.
    """
    from suitability_assessment_agent import guess_rps_id_column

    merged: Dict[str, PlanningRecord] = {}
    for tag in ("phase1", "phase2", "phase3"):
        shp = phased_dir / f"{scenario}_{tag}_{area}.shp"
        if not shp.is_file():
            logger.warning("Review: phased planning shp not found: %s", shp)
            continue
        try:
            gdf = gpd.read_file(shp)
        except Exception as exc:
            logger.warning("Review: failed to read %s: %s", shp, exc)
            continue
        id_col = guess_rps_id_column(gdf)
        if not id_col:
            logger.warning("Review: cannot guess RPS id column on %s", shp)
            continue
        pic_map = (
            _pic_basename_to_rps_id(scene_semantic_memory)
            if id_col in ("Pic", "pic")
            else None
        )
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            try:
                coords = [float(geom.x), float(geom.y)]
            except Exception:
                continue
            rid: Optional[str] = None
            if id_col in ("Pic", "pic") and pic_map:
                raw = row.get(id_col)
                if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                    pk = ""
                else:
                    pk = str(raw).strip()
                rid = pic_map.get(pk) if pk else None
            else:
                raw = row.get(id_col)
                if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                    rid = None
                else:
                    rid = str(raw).strip() or None
            if not rid or rid not in scene_semantic_memory:
                continue
            merged[rid] = PlanningRecord(rps_id=rid, coordinates=coords)
    if merged:
        logger.info(
            "Review: merged phased shp planning (%s unique RPS) from %s",
            len(merged),
            phased_dir,
        )
    return list(merged.values())


@dataclass
class ReviewMetrics:
    scr: float  # Service Coverage Rate
    dcr: float  # Demand Coverage Rate
    nle: float  # Network Loss Efficiency
    eb: float   # Economic Benefits
    scenario: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scr": self.scr,
            "dcr": self.dcr,
            "nle": self.nle,
            "eb": self.eb,
            "scenario": self.scenario
        }
        
    def calculate_score(self) -> float:
        weights = {
            "efficiency_oriented": {"scr": 0.15, "dcr": 0.15, "nle": 0.30, "eb": 0.40},
            "equity_oriented": {"scr": 0.40, "dcr": 0.40, "nle": 0.10, "eb": 0.10},
            "balance_oriented": {"scr": 0.25, "dcr": 0.25, "nle": 0.25, "eb": 0.25},
        }
        w = weights.get(self.scenario, weights["equity_oriented"])
        return self.scr * w["scr"] + self.dcr * w["dcr"] + self.nle * w["nle"] + self.eb * w["eb"]


class ReviewAgent:
    """
    Review Agent: evaluates planning results and decides whether to pass or rethink.
    """

    def __init__(self, threshold_mu: float = REVIEW_SCORE_THRESHOLD_DEFAULT):
        self.threshold_mu = threshold_mu
        
    def _calculate_metrics(
        self,
        phased_strategy: Dict[str, Any],
        planning: List[PlanningRecord],
        scene_semantic_memory: Dict[str, Any],
        poi_dir: Optional[Path] = None,
        existing_rcp: Optional[Path] = None,
    ) -> ReviewMetrics:
        scenario = phased_strategy.get("scenario", "equity_oriented")
        loader = ReviewDataLoader(Path(poi_dir) if poi_dir else None)
        poi_gdf = loader.load_poi_data()
        grid_gdf = loader.load_grid_points(poi_gdf)
        scr = ServiceCoverageRateCalculator().calculate(
            planning, scene_semantic_memory, poi_gdf, grid_gdf
        )
        dcr = DemandCoverageRateCalculator().calculate(
            planning, scene_semantic_memory, poi_gdf, grid_gdf
        )
        nle = NetworkLossEfficiencyCalculator(existing_rcp_path=existing_rcp).calculate(
            planning, scene_semantic_memory, poi_gdf, grid_gdf
        )
        eb = EconomicBenefitsCalculator().calculate(
            planning, scene_semantic_memory, poi_gdf, grid_gdf
        )
        return ReviewMetrics(
            scr=float(scr),
            dcr=float(dcr),
            nle=float(nle),
            eb=float(eb),
            scenario=scenario,
        )

    def evaluate_outcomes(
        self,
        phased_strategy: Dict[str, Any],
        scored_candidates: List[Dict[str, Any]],
        *,
        scene_semantic_memory: Optional[Dict[str, Any]] = None,
        poi_dir: Optional[Path] = None,
        existing_rcp: Optional[Path] = None,
        phased_planning_dir: Optional[Path] = None,
        phased_planning_area: Optional[str] = None,
        review_attempt: int = 0,
        max_readjust_rounds: int = RCPP_AGENT_MAX_WORKFLOW_ITERATIONS,
    ) -> Dict[str, Any]:

        print("Review Agent running...", flush=True)
        logger.info("Review Agent starts evaluating planning results...")

        mem = scene_semantic_memory or {}
        scenario = str(phased_strategy.get("scenario", "equity_oriented"))
        area_slug = (phased_planning_area or "").strip().lower()

        planning: List[PlanningRecord] = []
        if phased_planning_dir is not None and area_slug:
            planning = load_merged_phased_planning_records(
                Path(phased_planning_dir),
                scenario,
                area_slug,
                mem,
            )

        if not planning:
            instr = phased_strategy.get("scored_candidates_instruction_phase")
            if instr:
                candidates = instr
            else:
                candidates = scored_candidates or []
            if phased_strategy.get("instruction_phase_selected_rps_ids") is not None:
                selected = phased_strategy.get("instruction_phase_selected_rps_ids")
            else:
                selected = phased_strategy.get("all_selected_rps_ids")
            if selected and candidates:
                sel_set = {str(x) for x in selected}
                candidates = [c for c in candidates if str(c.get("rps_id", "")) in sel_set]
            planning = scored_candidates_to_planning_records(candidates)
            if not planning and phased_planning_dir and area_slug:
                logger.warning(
                    "Review: no planning rows from phased shp and no fallback candidates; metrics may be zero.",
                )

        metrics = self._calculate_metrics(
            phased_strategy,
            planning,
            mem,
            poi_dir=poi_dir,
            existing_rcp=existing_rcp,
        )
        comprehensive_score = metrics.calculate_score()
        
        logger.info(f"Comprehensive score calculation result: {comprehensive_score:.4f} (threshold: {self.threshold_mu})")
        
        passed = comprehensive_score >= self.threshold_mu
        
        readjust_scheduled = (not passed) and (review_attempt < max_readjust_rounds)
        round_display = review_attempt + 1

        if passed:
            logger.info("Planning results passed the review!")
            feedback = "No readjustment needed."
        elif readjust_scheduled:
            logger.info(
                "Score %.4f below threshold %.4f; automatically returning to "
                "Multi-Scenario Evaluation Agent to readjust LLM-AHP weights (round %s/%s).",
                comprehensive_score,
                self.threshold_mu,
                round_display,
                max_readjust_rounds,
            )
            feedback = (
                f"Comprehensive score {comprehensive_score:.4f} is below threshold "
                f"{self.threshold_mu:.4f}. The workflow automatically routes back to "
                "Multi-Scenario Evaluation Agent to recompute LLM-AHP weights "
                f"(review round {round_display}/{max_readjust_rounds})."
            )
        else:
            logger.warning(
                "Score %.4f below threshold %.4f; max readjust rounds (%s) exhausted.",
                comprehensive_score,
                self.threshold_mu,
                max_readjust_rounds,
            )
            feedback = (
                f"Comprehensive score {comprehensive_score:.4f} is below threshold "
                f"{self.threshold_mu:.4f}. No further automatic weight readjustment "
                f"(limit {max_readjust_rounds} rounds reached)."
            )

        review_result = {
            "metrics": metrics.to_dict(),
            "comprehensive_score": comprehensive_score,
            "threshold": self.threshold_mu,
            "passed": passed,
            "feedback": feedback,
            "readjust_scheduled": readjust_scheduled,
            "review_round": round_display,
            "max_readjust_rounds": max_readjust_rounds,
            "timestamp": datetime.now().isoformat(),
        }
        
        return review_result


def write_review_metrics_json(path: Union[str, Path], review_result: Dict[str, Any]) -> None:
    """
    Output review metrics to a JSON file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(review_result, f, ensure_ascii=False, indent=2, default=str)
