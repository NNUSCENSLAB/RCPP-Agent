"""
Suitability Assessment Agent: aligns rules with the environment
and scores each RPS for RCP deployment suitability.
"""

import logging
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import pandas as pd

from configs.data_config import DEFAULT_SPATIALITE_TABLES

logger = logging.getLogger(__name__)

_GEO_IO_ENGINE = "pyogrio"


# Metadata for one suitability rule and its evaluator method name on the RuleEnvironmentAlignment engine.
@dataclass
class Rule:
    rule_id: str
    dimension: str
    description: str
    evaluation_function: str

    # Serialize rule metadata for logging or downstream payloads.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "dimension": self.dimension,
            "description": self.description,
        }


# Outcome of applying one rule to extracted features for a single RPS.
@dataclass
class RuleEvaluationResult:
    """
    Per-rule pass or fail flag, numeric score, text rationale, and evidence dict.
    """
    rule_id: str
    satisfied: bool
    score: float
    reasoning: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Serialize one rule outcome including evidence snippets.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "satisfied": self.satisfied,
            "score": self.score,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
        }


# Assessment and scoring of environment-rule alignment for RPSs.
@dataclass
class SuitabilityEvaluationResult:
    """
    Comprehensive summary for a single RPS: applicable indicators, average score, rule mapping, and list of violations.
    """

    rps_id: str
    is_suitable: bool
    overall_score: float
    rule_results: Dict[str, RuleEvaluationResult]
    violated_rules: List[str]
    coordinates: List[float]
    street_view_images: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    # Serialize full per-RPS suitability result for checkpoints.
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rps_id": self.rps_id,
            "is_suitable": self.is_suitable,
            "overall_score": self.overall_score,
            "rule_results": {k: v.to_dict() for k, v in self.rule_results.items()},
            "violated_rules": self.violated_rules,
            "coordinates": self.coordinates,
            "street_view_images": list(self.street_view_images),
            "timestamp": self.timestamp.isoformat(),
        }


# Rule-based engine that maps scene-semantic features to scores per RPS.
class RuleEnvironmentAlignmentEngine:
    """
    Two-stage flow: extract compact features from memory, then run each rule
    evaluator and aggregate into a single SuitabilityEvaluationResult.
    """

    # Load the built-in rule table used for every evaluate call.
    def __init__(self) -> None:
        self.rules = self._initialize_rules()

    # Build the rule set.
    def _initialize_rules(self) -> Dict[str, Rule]:
        return {
            "PS-1": Rule(
                "PS-1",
                "Physical Space",
                "Ensuring sufficient width and length for equipment installation without obstructions.",
                "evaluate_physical_space",
            ),
            "SE-1": Rule(
                "SE-1",
                "Surrounding Environment",
                "Ensuring compatibility with surrounding building types (e.g., noise mitigation for residential, high density for commercial).",
                "evaluate_surrounding_environment",
            ),
            "GR-1": Rule(
                "GR-1",
                "Power Grid",
                "Ensuring reliable grid connection and voltage stability within permissible limits.",
                "evaluate_power_grid",
            ),
            "TF-1": Rule(
                "TF-1",
                "Traffic",
                "Ensuring ease of vehicle ingress/egress; equipment must not block evacuation routes.",
                "evaluate_traffic",
            ),
            "RL-1": Rule(
                "RL-1",
                "Regulation",
                "Adhering to zoning laws; strictly prohibited in restricted zones (e.g., school vicinities).",
                "evaluate_regulation",
            ),
        }

    def extract_features(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 1: Extract features from scene-semantic memory.
        """
        scene_mem = memory.get("scene_memory", {})
        semantic_mem = memory.get("semantic_memory", {})

        return {
            "scene": {
                "clearance_assessment": scene_mem.get("clearance_visual_assessment", False),
                "scene_reasoning": scene_mem.get("scene_reasoning", ""),
            },
            "semantic": {
                "functional_zone": semantic_mem.get("functional_zone_type", ""),
                "commuting_flow": semantic_mem.get("commuting_flow", ""),
                "sensitive_constraints": semantic_mem.get("sensitive_constraints", ""),
                "grid_accessibility": semantic_mem.get("grid_accessibility", ""),
            },
        }

    # PS-1: align to scene memory clearance_visual_assessment and scene_memory scene_reasoning for curbside clearance and obstruction cues.
    def evaluate_physical_space(self, features: Dict[str, Any]) -> RuleEvaluationResult:
        scene = features["scene"]
        clearance = scene.get("clearance_assessment", False)
        reasoning_src = scene.get("scene_reasoning", "")

        if isinstance(clearance, str):
            satisfied = clearance.lower() == "true"
        elif isinstance(clearance, bool):
            satisfied = clearance
        else:
            satisfied = False

        score = 1.0 if satisfied else 0.0
        evidence = {
            "clearance_assessment": clearance,
            "reasoning_extract": reasoning_src[:200] if reasoning_src else "No reasoning provided",
        }
        if satisfied:
            reasoning = "Physical space assessment: True. Sufficient physical space, meets installation requirements."
        else:
            reasoning = "Physical space assessment: False. Obstacles present or insufficient space, does not meet installation requirements."

        return RuleEvaluationResult("PS-1", satisfied, score, reasoning, evidence)

    # SE-1: align to semantic memory functional_zone_type for surrounding built environment fit.
    def evaluate_surrounding_environment(self, features: Dict[str, Any]) -> RuleEvaluationResult:
        functional_zone = features["semantic"].get("functional_zone", "")

        if "Commercial-Oriented" in functional_zone or "Residential-Dominant" in functional_zone:
            score = 1.0
            reasoning = "Surrounding environment is commercial or residential functional zone, suitable for charging pile deployment."
        elif "Mixed" in functional_zone:
            score = 0.8
            reasoning = "Surrounding environment is mixed functional zone, generally suitable for charging pile deployment."
        else:
            score = 0.7
            reasoning = "Surrounding environment function not clearly classified, but no obvious conflicts."

        evidence = {"functional_zone": functional_zone}
        return RuleEvaluationResult("SE-1", True, score, reasoning, evidence)

    # GR-1: align to semantic memory grid_accessibility for distance tokens.
    def evaluate_power_grid(self, features: Dict[str, Any]) -> RuleEvaluationResult:
        grid_acc = features["semantic"].get("grid_accessibility", "")

        distance_match = re.search(r"Dist[=:]?\s*(\d+\.?\d*)\s*m", grid_acc)
        satisfied = True
        distance: float = 0.0

        if distance_match:
            distance = float(distance_match.group(1))
            if distance > 5000:
                satisfied = False
                score = 0.0
                reasoning = f"Grid distance is {distance:.1f}m, exceeds 5000m threshold, does not meet deployment requirements."
            elif distance <= 500:
                score = 1.0
                reasoning = f"Grid distance is {distance:.1f}m, excellent grid connection conditions."
            elif distance <= 1000:
                score = 0.9
                reasoning = f"Grid distance is {distance:.1f}m, good grid connection conditions."
            elif distance <= 2000:
                score = 0.7
                reasoning = f"Grid distance is {distance:.1f}m, grid connection feasible but requires additional costs."
            else:
                score = 0.5
                reasoning = f"Grid distance is {distance:.1f}m, grid connection feasible within acceptable range."
        else:
            if "Very Poor" in grid_acc or "Extremely Difficult" in grid_acc or "Severe" in grid_acc:
                if "5km" in grid_acc or "5000" in grid_acc or "exceed" in grid_acc.lower():
                    satisfied = False
                    score = 0.0
                    reasoning = "Grid connection conditions exceed 5000m threshold, does not meet deployment requirements."
                else:
                    score = 0.3
                    reasoning = "Grid connection conditions challenging but within acceptable range."
            elif "Poor" in grid_acc or "Difficult" in grid_acc:
                score = 0.5
                reasoning = "Grid connection conditions acceptable with some engineering challenges."
            else:
                score = 0.8
                reasoning = "Grid connection conditions acceptable."

        evidence = {
            "grid_accessibility": grid_acc,
            "extracted_distance": distance if distance_match else None,
        }
        return RuleEvaluationResult("GR-1", satisfied, score, reasoning, evidence)

    # TF-1: align to semantic memory commuting_flow for flow intensity.
    def evaluate_traffic(self, features: Dict[str, Any]) -> RuleEvaluationResult:
        commuting_flow = features["semantic"].get("commuting_flow", "")
        scene_reasoning = features["scene"].get("scene_reasoning", "")

        satisfied = True
        if "High Traffic" in commuting_flow or "Heavy" in commuting_flow:
            score = 0.85
            reasoning = "High commuting flow, must ensure no impact on normal traffic."
        elif "Medium Traffic" in commuting_flow or "Moderate" in commuting_flow:
            score = 1.0
            reasoning = "Moderate commuting flow, meets deployment requirements."
        elif "Low Traffic" in commuting_flow:
            score = 0.9
            reasoning = "Low commuting flow, meets deployment requirements but usage may be limited."
        else:
            score = 0.8
            reasoning = "Traffic conditions acceptable."

        if "evacuation" in scene_reasoning.lower() or "emergency" in scene_reasoning.lower():
            satisfied = False
            score = 0.0
            reasoning = "Located in evacuation or emergency route, does not meet deployment requirements."

        evidence = {
            "commuting_flow": commuting_flow,
            "scene_check": scene_reasoning[:200] if scene_reasoning else "",
        }
        return RuleEvaluationResult("TF-1", satisfied, score, reasoning, evidence)

    # RL-1: align to semantic memory sensitive_constraints for risk tiers and proximity cues near schools, hospitals, or similar sensitive uses.
    def evaluate_regulation(self, features: Dict[str, Any]) -> RuleEvaluationResult:
        constraints = features["semantic"].get("sensitive_constraints", "")

        satisfied = True
        if "High Risk" in constraints:
            satisfied = False
            score = 0.0
            reasoning = "High-risk sensitive facilities present, deployment strictly prohibited."
        elif "Medium Risk" in constraints or "Risk Warning" in constraints:
            if "No Risk" in constraints or "Safe" in constraints:
                score = 1.0
                reasoning = "No regulatory restrictions, meets deployment requirements."
            else:
                satisfied = False
                score = 0.3
                reasoning = "Medium-risk sensitive facilities present, requires careful assessment."
        elif "No Risk" in constraints or "Safe" in constraints:
            score = 1.0
            reasoning = "No sensitive facility restrictions, meets regulatory requirements."
        else:
            score = 0.8
            reasoning = "No obvious regulatory conflicts found."

        if satisfied:
            sensitive_keywords = ["school", "hospital", "kindergarten"]
            for keyword in sensitive_keywords:
                if keyword in constraints.lower():
                    if "No" not in constraints or "far" in constraints.lower():
                        satisfied = False
                        score = 0.0
                        reasoning = f"Near sensitive facility ({keyword}), does not meet regulatory requirements."
                        break

        evidence = {"sensitive_constraints": constraints}
        return RuleEvaluationResult("RL-1", satisfied, score, reasoning, evidence)

    def align_rules_with_features(self, features: Dict[str, Any]) -> Dict[str, RuleEvaluationResult]:
        """
        Stage 2: Rule-environment alignment.
        """
        results: Dict[str, RuleEvaluationResult] = {}
        for rule_id, rule in self.rules.items():
            eval_func = getattr(self, rule.evaluation_function)
            results[rule_id] = eval_func(features)
        return results

    def evaluate(self, rps_id: str, coordinates: List[float], memory: Dict[str, Any]) -> SuitabilityEvaluationResult:
        """
        Execute suitability assessment for RPSs.
        """
        features = self.extract_features(memory)
        rule_results = self.align_rules_with_features(features)

        all_satisfied = all(res.satisfied for res in rule_results.values())
        overall_score = (
            sum(res.score for res in rule_results.values()) / len(rule_results) if rule_results else 0.0
        )
        violated_rules = [r_id for r_id, res in rule_results.items() if not res.satisfied]

        return SuitabilityEvaluationResult(
            rps_id=rps_id,
            is_suitable=all_satisfied,
            overall_score=overall_score,
            rule_results=rule_results,
            violated_rules=violated_rules,
            coordinates=coordinates,
        )


class SuitabilityAssessmentAgent:

    def __init__(self) -> None:
        self.engine = RuleEnvironmentAlignmentEngine()

    def assess_suitability(
        self, scene_semantic_memory: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[SuitabilityEvaluationResult]]:
        """
        Evaluate each RPS in the Environment Perception Agent's output.
        """
        print("Suitability Assessment Agent running...", flush=True)
        logger.info("Suitability Assessment Agent: evaluating %s RPSs.", len(scene_semantic_memory))

        suitable_rps: List[Dict[str, Any]] = []
        evaluations: List[SuitabilityEvaluationResult] = []

        for rps_id, memory_obj in scene_semantic_memory.items():
            coords = memory_obj.get("coordinates", [0.0, 0.0])
            memory_data = memory_obj.get("memory_node", {}) if "memory_node" in memory_obj else memory_obj

            result = self.engine.evaluate(rps_id, coords, memory_data)
            imgs = memory_obj.get("street_view_images")
            if isinstance(imgs, list):
                sv = [str(x) for x in imgs]
            else:
                sv = []
            result = replace(result, street_view_images=sv)
            evaluations.append(result)

            if result.is_suitable:
                suitable_rps.append(
                    {
                        "id": rps_id,
                        "coordinates": coords,
                        "suitability_score": result.overall_score,
                    }
                )

        logger.info(
            "Suitability assessment finished: %s RPS total, %s suitable for deployment.",
            len(scene_semantic_memory),
            len(suitable_rps),
        )
        return suitable_rps, evaluations


# Shapefile export
def resolve_rps_shp_path(spatialite_context: Dict[str, Any]) -> Optional[Path]:
    raw = spatialite_context.get("rps_shp_path")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    return None


def resolve_spatialite_db_path(spatialite_context: Dict[str, Any]) -> Optional[Path]:
    """Absolute path to the SpatiaLite DB; resolves relative ``db_path`` against ``workspace_dir``."""
    raw = spatialite_context.get("db_path")
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        ws = spatialite_context.get("workspace_dir")
        if ws:
            p = Path(str(ws)) / p
    return p.resolve()


def load_rps_gdf_from_spatialite(spatialite_context: Dict[str, Any]) -> gpd.GeoDataFrame:
    """Read the RPS layer from SpatiaLite (same source as import_spatialite)."""
    db_path = resolve_spatialite_db_path(spatialite_context)
    if db_path is None or not db_path.is_file():
        raise FileNotFoundError(
            f"SpatiaLite database not found: {db_path}"
        )
    tables = spatialite_context.get("tables") or {}
    layer = str(tables.get("rps") or DEFAULT_SPATIALITE_TABLES["rps"])
    gdf = gpd.read_file(db_path, layer=layer, engine=_GEO_IO_ENGINE)
    return gdf


def load_rps_gdf_for_export(spatialite_context: Dict[str, Any]) -> Optional[gpd.GeoDataFrame]:
    """
    RPS geometries for suitability / phased SHP export: use ``rps_shp_path`` when present on disk,
    otherwise the ``rps`` layer in ``spatialite_context``'s SpatiaLite DB.
    """
    shp = resolve_rps_shp_path(spatialite_context)
    if shp is not None and shp.exists():
        try:
            gdf = load_merged_rps_gdf(shp)
            logger.info("RPS geometry source: shapefile %s (%s features)", shp, len(gdf))
            return gdf
        except Exception as exc:
            logger.warning(
                "RPS shapefile load failed (%s); using SpatiaLite rps layer.", exc
            )
    try:
        gdf = load_rps_gdf_from_spatialite(spatialite_context)
        dbp = resolve_spatialite_db_path(spatialite_context)
        tables = spatialite_context.get("tables") or {}
        layer = str(tables.get("rps") or DEFAULT_SPATIALITE_TABLES["rps"])
        logger.info(
            "RPS geometry source: SpatiaLite %s layer=%s (%s features)",
            dbp,
            layer,
            len(gdf),
        )
        return gdf
    except Exception as exc:
        logger.warning("Could not load RPS GeoDataFrame from SpatiaLite: %s", exc)
        return None


def load_merged_rps_gdf(shp_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """Load one .shp file or merge all .shp in a directory."""
    p = Path(shp_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"RPS shapefile path not found: {p}")
    if p.is_file():
        if p.suffix.lower() != ".shp":
            raise ValueError(f"Not a shapefile: {p}")
        return gpd.read_file(p)
    if not p.is_dir():
        raise ValueError(f"Invalid path: {p}")
    shps = sorted(p.glob("*.shp"))
    if not shps:
        raise ValueError(f"No .shp files under directory: {p}")
    frames = [gpd.read_file(f) for f in shps]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))


def guess_rps_id_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    """Prefer ``Pic`` / ``pic`` (image filename) when present — primary join key per project data."""
    for name in ("Pic", "pic", "RPS_ID", "RPS_id", "rps_id", "RPSID"):
        if name in gdf.columns:
            return name
    return None


def _rule_field_prefix(rule_id: str) -> str:
    return rule_id.replace("-", "_")


def export_suitability_shapefiles(
    original_gdf: gpd.GeoDataFrame,
    evaluations: Sequence[SuitabilityEvaluationResult],
    output_dir: Union[str, Path],
    area_slug: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Write suitable_rps_{area}.shp and unsuitable_rps_{area}.shp.
    If the GeoDataFrame has ``Pic``, join by first street-view image basename.
    Else join by RPS id column.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    crs = original_gdf.crs if getattr(original_gdf, "crs", None) else "EPSG:4326"
    by_rps_id: Dict[str, SuitabilityEvaluationResult] = {e.rps_id: e for e in evaluations}
    pic_map: Dict[str, SuitabilityEvaluationResult] = {}
    for ev in evaluations:
        imgs = getattr(ev, "street_view_images", None) or []
        if not imgs:
            continue
        pic_map[Path(str(imgs[0])).name.strip()] = ev

    id_col = guess_rps_id_column(original_gdf)
    pic_col = (
        "Pic"
        if "Pic" in original_gdf.columns
        else ("pic" if "pic" in original_gdf.columns else None)
    )

    suitable_rows: List[Dict[str, Any]] = []
    unsuitable_rows: List[Dict[str, Any]] = []

    for _, row in original_gdf.iterrows():
        base = row.to_dict()
        ev: Optional[SuitabilityEvaluationResult] = None
        if pic_col:
            raw = row.get(pic_col)
            if raw is None:
                pk = ""
            elif isinstance(raw, float) and math.isnan(raw):
                pk = ""
            else:
                pk = str(raw).strip()
            if pk:
                ev = pic_map.get(pk)
        if ev is None and id_col and id_col not in ("Pic", "pic"):
            raw = row.get(id_col)
            if raw is None:
                k = ""
            elif isinstance(raw, float) and math.isnan(raw):
                k = ""
            else:
                k = str(raw).strip()
            if k:
                ev = by_rps_id.get(k)
        if ev is None:
            base["Matched"] = False
            base["Suitable"] = False
            base["Score"] = 0.0
            unsuitable_rows.append(base)
            continue
        base["Matched"] = True
        base["RPS_ID"] = ev.rps_id
        base["Suitable"] = ev.is_suitable
        base["Score"] = round(ev.overall_score, 6)
        for rid, rr in ev.rule_results.items():
            pfx = _rule_field_prefix(str(rid))
            base[f"{pfx}_ok"] = rr.satisfied
            base[f"{pfx}_score"] = round(rr.score, 6)
        base["Violations"] = len(ev.violated_rules)
        (suitable_rows if ev.is_suitable else unsuitable_rows).append(base)

    sp: Optional[Path] = out / f"suitable_rps_{area_slug}.shp"
    up: Optional[Path] = out / f"unsuitable_rps_{area_slug}.shp"
    if suitable_rows:
        gpd.GeoDataFrame(suitable_rows, crs=crs).to_file(sp, encoding="utf-8")
        logger.info("Wrote %s (%s features)", sp, len(suitable_rows))
    else:
        logger.warning("No suitable RPS rows to export for area=%s", area_slug)
        sp = None
    if unsuitable_rows:
        gpd.GeoDataFrame(unsuitable_rows, crs=crs).to_file(up, encoding="utf-8")
        logger.info("Wrote %s (%s features)", up, len(unsuitable_rows))
    else:
        up = None
    return sp, up
