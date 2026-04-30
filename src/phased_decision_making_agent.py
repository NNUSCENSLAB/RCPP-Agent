"""
Phased Decision-Making Agent: applies scenario weights to score suitable RPSs,
selects quotas, and outputs a phased RCP deployment strategy.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Any, Sequence, Tuple, Union, cast
from dataclasses import dataclass, field
from datetime import datetime

import geopandas as gpd

from tools.multi_scenario_evaluation_tool import (
    evaluate_economic,
    evaluate_policy,
    evaluate_social,
    evaluate_technical,
    evaluate_traffic,
    split_scene_semantic_from_rps_record,
)
from configs.config import PLANNING_PHASE, PlanningPhase, SCENARIO_RPS_PLANNING_RATIO
from tools.llm_ahp_tool import LLMAHPTool
from tools.phased_decision_making_tool import PhasedAllocationTool
from suitability_assessment_agent import guess_rps_id_column

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """
    Save the score and details for a specific evaluation dimension for a single RPS.
    """
    dimension: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorityEvaluationResult:
    """
    Save the overall priority score and the scores for each evaluation dimension for a single RPS.
    """
    rps_id: str
    coordinates: List[float]
    overall_priority_score: float
    dimension_scores: Dict[str, DimensionScore]
    scenario: str
    phase: str
    year: int
    weights: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rps_id": self.rps_id,
            "coordinates": self.coordinates,
            "overall_priority_score": self.overall_priority_score,
            "dimension_scores": {
                k: {
                    "dimension": v.dimension,
                    "score": v.score,
                    "details": v.details,
                }
                for k, v in self.dimension_scores.items()
            },
            "scenario": self.scenario,
            "phase": self.phase,
            "year": self.year,
            "weights": self.weights,
            "timestamp": self.timestamp.isoformat(),
        }


def _mcda_context(
    spatialite_context: Optional[Dict[str, Any]],
    scenario: str,
    phase_label: str,
    year: int,
) -> Dict[str, Any]:
    """
    Shared context passed into each dimension evaluator for this scenario, phase, and year.
    """
    ctx: Dict[str, Any] = {
        "scenario": scenario,
        "phase": phase_label,
        "year": year,
    }
    if spatialite_context:
        path = spatialite_context.get("road_network_shp")
        if path:
            ctx["road_network_path"] = path
    return ctx


class DimensionScoringEngine:
    """
    Runs the MCDA indicator tools per RPS and returns scores.
    """

    _EVALUATORS = (
        ("technical", evaluate_technical),
        ("economic", evaluate_economic),
        ("social", evaluate_social),
        ("traffic", evaluate_traffic),
        ("policy", evaluate_policy),
    )

    def score_rps(
        self,
        rps_id: str,
        coordinates: List[float],
        mem: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, DimensionScore]:
        """
        Score one RPS on technical, economic, social, traffic, and policy dimensions.
        """
        scene, sem = split_scene_semantic_from_rps_record(mem)
        out: Dict[str, DimensionScore] = {}
        for dim, fn in self._EVALUATORS:
            score, details = fn(rps_id, coordinates, scene, sem, context)
            out[dim] = DimensionScore(dimension=dim, score=float(score), details=details)
        return out


class PhasedDecisionMakingAgent:
    """
    Produces phase-adjusted weights, full ranked scores, and a strategy summary for the graph.
    """

    def __init__(self) -> None:
        """Wire the dimension scoring engine; LLM-AHP runs through a tool instance created on first access."""
        self.scoring_engine = DimensionScoringEngine()
        self._llm_ahp_tool: Optional[LLMAHPTool] = None

    @property
    def llm_ahp_tool(self) -> LLMAHPTool:
        """
        Lazy LLM-AHP tool for kind phase so scenario weights are blended with the deployment phase.
        """
        if self._llm_ahp_tool is None:
            self._llm_ahp_tool = LLMAHPTool()
        return self._llm_ahp_tool

    def _score_candidates(
        self,
        suitable_rps: List[Dict[str, Any]],
        memories: Dict[str, Any],
        weights: Dict[str, float],
        scenario: str,
        phase_label: str,
        year: int,
        spatialite_context: Optional[Dict[str, Any]] = None,
    ) -> List[PriorityEvaluationResult]:
        """
        Apply MCDA weights to all candidate RPSs.
        """
        ctx = _mcda_context(spatialite_context, scenario, phase_label, year)
        results: List[PriorityEvaluationResult] = []
        for rps in suitable_rps:
            rps_id = rps["id"]
            coords = rps["coordinates"]
            mem = memories.get(rps_id, {})
            if isinstance(mem, dict):
                mem = dict(mem)
                if not mem.get("coordinates") and coords is not None:
                    mem["coordinates"] = coords

            dim_objs = self.scoring_engine.score_rps(rps_id, coords, mem, ctx)
            dim_scores = {d: o.score for d, o in dim_objs.items()}
            overall_score = sum(
                dim_scores[d] * weights.get(d, 0.2) for d in dim_scores
            )

            results.append(
                PriorityEvaluationResult(
                    rps_id=rps_id,
                    coordinates=coords,
                    overall_priority_score=overall_score,
                    dimension_scores=dim_objs,
                    scenario=scenario,
                    phase=phase_label,
                    year=year,
                    weights=weights,
                )
            )

        results.sort(key=lambda x: x.overall_priority_score, reverse=True)
        return results

    def formulate_phased_strategy(
        self,
        suitable_rps: List[Dict[str, Any]],
        memories: Dict[str, Any],
        scenario_mcda_weights: Dict[str, float],
        scenario: str,
        phase: PlanningPhase,
        year: int,
        time_horizon: Union[List[int], tuple, None] = None,
        total_budget_or_quota: int = 100,
        spatialite_context: Optional[Dict[str, Any]] = None,
        weight_revision: int = 0,
    ) -> Dict[str, Any]:
        """
        Refine weights for this deployment phase, score and rank all suitable RPSs,
        return strategy dict, full scored list, and final five-dimension weights for review.
        """
        print("Phased Decision-Making Agent running...", flush=True)
        phase_key = phase.value
        phase_display = PLANNING_PHASE.get(phase_key, phase_key)
        logger.info(
            "Phased Decision-Making Agent formulating phased planning (phase=%s, year=%s)...",
            phase_display,
            year,
        )

        phase_out = self.llm_ahp_tool.run(
            kind="phase",
            scenario=scenario,
            scenario_weights=scenario_mcda_weights,
            time_horizon=time_horizon,
            phase=phase,
            weight_revision=str(weight_revision) if weight_revision else "",
        )
        final_weights = cast(Dict[str, float], phase_out["weights"])
        weight_reasoning = str(phase_out.get("reasoning", "")).strip()

        scored_results = self._score_candidates(
            suitable_rps,
            memories,
            final_weights,
            scenario,
            phase_display,
            year,
            spatialite_context,
        )

        n_pool = len(scored_results)
        ratio = float(SCENARIO_RPS_PLANNING_RATIO.get(scenario, 1.0))
        ratio = max(0.0, min(1.0, ratio))
        cap_by_scenario = 0
        if n_pool > 0:
            cap_by_scenario = min(n_pool, math.ceil(n_pool * ratio))
        effective_quota = min(total_budget_or_quota, cap_by_scenario)

        selected_candidates = scored_results[:effective_quota]

        strategy = {
            "phase": phase_display,
            "year": year,
            "scenario": scenario,
            "requested_quota": total_budget_or_quota,
            "scenario_rps_planning_ratio": ratio,
            "rps_cap_by_scenario_ratio": cap_by_scenario,
            "quota": effective_quota,
            "selected_count": len(selected_candidates),
            "selected_rps_ids": [r.rps_id for r in selected_candidates],
            "average_score": sum(r.overall_priority_score for r in selected_candidates)
            / max(1, len(selected_candidates)),
            "weight_reasoning": weight_reasoning,
        }

        logger.info(
            "Phased deployment planning completed (%s). Selected %s RPSs for this phase.",
            phase_display,
            len(selected_candidates),
        )

        return {
            "phased_strategy": strategy,
            "scored_candidates": [r.to_dict() for r in scored_results],
            "mcda_weights": final_weights,
            "llm_ahp_phase": {
                "weights": phase_out.get("weights"),
                "pairwise_matrix": phase_out.get("pairwise_matrix"),
                "consistency_ratio": phase_out.get("consistency_ratio"),
                "reasoning": str(phase_out.get("reasoning", "")).strip(),
            },
        }

    def formulate_phase_planning(
        self,
        suitable_rps: List[Dict[str, Any]],
        memories: Dict[str, Any],
        scenario_mcda_weights: Dict[str, float],
        scenario: str,
        time_horizon: Union[List[int], tuple, None] = None,
        total_budget_or_quota: int = 100,
        spatialite_context: Optional[Dict[str, Any]] = None,
        weight_revision: int = 0,
        instruction_year: int = 2025,
    ) -> Dict[str, Any]:
        """
        Run three planning phases in order with disjoint RPS selections: each phase only
        draws from RPS not yet allocated in a previous phase.
        """
        phase_sequence: List[Tuple[PlanningPhase, int, str]] = [
            (PlanningPhase.INITIAL_EXPLORATION, 2025, "phase1"),
            (PlanningPhase.RAPID_DEVELOPMENT, 2027, "phase2"),
            (PlanningPhase.MATURITY_OPTIMIZATION, 2029, "phase3"),
        ]

        n0 = len(suitable_rps)
        ratio = float(SCENARIO_RPS_PLANNING_RATIO.get(scenario, 1.0))
        ratio = max(0.0, min(1.0, ratio))
        cap_by_scenario = min(n0, math.ceil(n0 * ratio)) if n0 > 0 else 0
        total_cap = min(total_budget_or_quota, cap_by_scenario)

        alloc_tool = PhasedAllocationTool()
        alloc_result = alloc_tool.decide_allocation(scenario, total_cap)
        ac = alloc_result.get("allocation_counts") or {}
        quotas = [
            int(ac.get("phase1", 0)),
            int(ac.get("phase2", 0)),
            int(ac.get("phase3", 0)),
        ]
        allocated: set[str] = set()
        merged_by_rps: Dict[str, Dict[str, Any]] = {}
        phase_round: List[Dict[str, Any]] = []
        last_weights: Dict[str, float] = dict(scenario_mcda_weights)

        for (phase_enum, year, tag), quota in zip(phase_sequence, quotas):
            pool = [r for r in suitable_rps if str(r.get("id", "")) not in allocated]
            if not pool or quota <= 0:
                phase_round.append(
                    {
                        "phase_tag": tag,
                        "phased_strategy": {
                            "phase": PLANNING_PHASE.get(phase_enum.value, phase_enum.value),
                            "year": year,
                            "scenario": scenario,
                            "quota": 0,
                            "selected_count": 0,
                            "selected_rps_ids": [],
                            "skipped": True,
                        },
                        "scored_candidates": [],
                        "mcda_weights": last_weights,
                        "llm_ahp_phase": None,
                    }
                )
                continue

            one = self.formulate_phased_strategy(
                pool,
                memories,
                scenario_mcda_weights,
                scenario,
                phase_enum,
                year,
                time_horizon=time_horizon,
                total_budget_or_quota=quota,
                spatialite_context=spatialite_context,
                weight_revision=weight_revision,
            )
            one["phase_tag"] = tag
            phase_round.append(one)
            last_weights = cast(Dict[str, float], one.get("mcda_weights") or last_weights)
            for rid in one["phased_strategy"].get("selected_rps_ids", []):
                allocated.add(str(rid))
            for c in one.get("scored_candidates", []):
                rid = c.get("rps_id")
                if isinstance(rid, str):
                    merged_by_rps[rid] = c

        combined_phased_strategy: Dict[str, Any] = {
            "scenario": scenario,
            "phases": [p.get("phased_strategy", {}) for p in phase_round],
            "all_selected_rps_ids": sorted(allocated),
            "total_planned_unique": len(allocated),
            "total_cap": total_cap,
            "quotas_per_phase": quotas,
            "phase_allocation_llm": {
                "allocation_scenario_slug": scenario,
                "allocation": alloc_result.get("allocation"),
                "allocation_counts": alloc_result.get("allocation_counts"),
                "reasoning": alloc_result.get("reasoning", ""),
            },
        }

        itag = instruction_phase_tag_from_year(int(instruction_year))
        p_instr = planning_phase_from_instruction_year(int(instruction_year))
        instr_sel: List[str] = []
        instr_review: List[Dict[str, Any]] = []
        for p in phase_round:
            if p.get("phase_tag") != itag:
                continue
            ps = p.get("phased_strategy", {})
            instr_sel = [str(x) for x in ps.get("selected_rps_ids", [])]
            sc_map = {
                c["rps_id"]: c
                for c in p.get("scored_candidates", [])
                if isinstance(c.get("rps_id"), str)
            }
            for rid in instr_sel:
                if rid in sc_map:
                    instr_review.append(sc_map[rid])
            break

        combined_phased_strategy["instruction_year"] = int(instruction_year)
        combined_phased_strategy["instruction_phase_tag"] = itag
        combined_phased_strategy["instruction_planning_phase"] = PLANNING_PHASE.get(
            p_instr.value, p_instr.value
        )
        combined_phased_strategy["instruction_phase_selected_rps_ids"] = sorted(instr_sel)
        combined_phased_strategy["scored_candidates_instruction_phase"] = instr_review

        return {
            "combined_phased_strategy": combined_phased_strategy,
            "scored_candidates_review": instr_review,
            "scored_candidates_memory": list(merged_by_rps.values()),
            "mcda_weights": last_weights,
            "phase_round": phase_round,
        }


def instruction_phase_tag_from_year(year: int) -> str:
    """Map instruction year to phase1/phase2/phase3 (same breakpoints as planning_phase_from_instruction_year)."""
    if year <= 2026:
        return "phase1"
    if year <= 2028:
        return "phase2"
    return "phase3"


def planning_phase_from_instruction_year(year: int) -> PlanningPhase:
    """Same rule as RCPP orchestration: year bands select INITIAL / RAPID / MATURITY."""
    if year <= 2026:
        return PlanningPhase.INITIAL_EXPLORATION
    if year <= 2028:
        return PlanningPhase.RAPID_DEVELOPMENT
    return PlanningPhase.MATURITY_OPTIMIZATION


def write_json_path(path: Union[str, Path], data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def export_phase_rps_shapefile(
    original_gdf: gpd.GeoDataFrame,
    rps_ids: Sequence[str],
    output_shp: Union[str, Path],
    *,
    id_column: Optional[str] = None,
    memories: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Subset GeoDataFrame to selected planning RPS.
    """
    out = Path(output_shp)
    out.parent.mkdir(parents=True, exist_ok=True)
    want_ids = {str(x) for x in rps_ids}
    if not want_ids:
        logger.warning("No RPS ids for export: %s", out)
        return None

    col = id_column or guess_rps_id_column(original_gdf)
    if not col:
        raise ValueError("Cannot determine RPS id column on GeoDataFrame")

    if col in ("Pic", "pic") and memories:
        pics: set[str] = set()
        for rid in want_ids:
            node = memories.get(rid)
            if not isinstance(node, dict):
                continue
            imgs = node.get("street_view_images") or []
            if imgs:
                pics.add(Path(str(imgs[0])).name.strip())
        if pics:
            sub = original_gdf[original_gdf[col].astype(str).str.strip().isin(list(pics))]
        else:
            sub = gpd.GeoDataFrame()
    else:
        sub = original_gdf[original_gdf[col].astype(str).isin(list(want_ids))]

    if sub.empty:
        logger.warning("No matching rows for phase export: %s", out)
        return None
    sub = sub.copy()
    sub.to_file(out, encoding="utf-8")
    logger.info("Phase RPS export: %s (%s features)", out, len(sub))
    return out.resolve()


__all__ = [
    "PhasedDecisionMakingAgent",
    "PlanningPhase",
    "PriorityEvaluationResult",
    "instruction_phase_tag_from_year",
    "planning_phase_from_instruction_year",
    "write_json_path",
    "export_phase_rps_shapefile",
]
