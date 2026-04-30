# -*- coding: utf-8 -*-
"""
Shared types, text parsing, and charging-demand helpers for review metric calculators.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PlanningRecord:
    """
    One planned RPS id and map coordinates for review metrics.
    """

    rps_id: str
    coordinates: List[float]


def extract_traffic_total(commuting_flow_text: str) -> float:
    """
    Parse total commuting flow count from text for review demand helpers.
    """
    match = re.search(r"Total[:\s]+(\d+)", commuting_flow_text or "", re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def extract_traffic_outflow(commuting_flow_text: str) -> float:
    """
    Parse outflow from text or derive it from total minus inflow when both appear.
    """
    text = commuting_flow_text or ""
    match = re.search(r"Outflow[:\s]+(\d+)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    total_match = re.search(r"Total[:\s]+(\d+)", text, re.IGNORECASE)
    inflow_match = re.search(r"Inflow[:\s]+(\d+)", text, re.IGNORECASE)
    if total_match and inflow_match:
        return float(total_match.group(1)) - float(inflow_match.group(1))
    return 0.0


def extract_poi_counts(functional_zone_text: str) -> Tuple[int, int]:
    """
    Read residential and commercial POI counts from zone text for zone type rules.
    """
    text = functional_zone_text or ""
    p_res, p_comm = 0, 0
    res_match = re.search(r"\$P_\{res\}=(\d+)", text)
    comm_match = re.search(r"\$P_\{comm\}=(\d+)", text)
    if res_match:
        p_res = int(res_match.group(1))
    if comm_match:
        p_comm = int(comm_match.group(1))
    return p_res, p_comm


def determine_zone_type_and_rcp_prob(p_res: int, p_comm: int) -> Tuple[str, float]:
    """
    Return a simple zone label and use rate from two integer counts for demand scoring.
    """
    if p_res > p_comm * 1.5:
        return "residential", 0.25
    if p_comm > p_res * 1.5:
        return "commercial", 0.55
    return "mixed", 0.40


def normalize_semantic_memory(semantic_memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge features sub-dictionary to the top level, for parsing keys like commuting_flow.
    """
    if not isinstance(semantic_memory, dict):
        return {}
    out = dict(semantic_memory)
    feat = semantic_memory.get("features")
    if isinstance(feat, dict):
        for k, v in feat.items():
            out.setdefault(k, v)
    return out


def rps_entry_coordinates(rps_entry: Dict[str, Any]) -> List[float]:
    """Return lon lat as two floats from one scene_semantic_memory value for review metrics."""
    c = rps_entry["coordinates"]
    coords = list(c) if isinstance(c, (list, tuple)) else [0.0, 0.0]
    return coords[:2] if len(coords) >= 2 else [0.0, 0.0]


def scored_candidates_to_planning_records(
    candidates: List[Dict[str, Any]],
) -> List[PlanningRecord]:
    """
    Turn scored candidate dicts from phased planning into PlanningRecord rows for SCR DCR NLE and EB.
    """
    records: List[PlanningRecord] = []
    for c in candidates or []:
        rid = str(c.get("rps_id", ""))
        coords = c.get("coordinates") or [0.0, 0.0]
        if not rid:
            continue
        records.append(PlanningRecord(rps_id=rid, coordinates=list(coords)[:2]))
    return records


class ChargingDemandPredictor:
    """
    Charging demand (kWh/day) based on commuting text OD agent, for DCR/EB usage.
    """

    def __init__(
        self,
        *,
        ev_penetration_rate: float = 0.5407,
        avg_charging_energy_kwh: float = 48.0,
        daily_charge_probability: float = 0.5,
        max_demand_for_norm: float = 10000.0,
    ):
        """
        Store fixed rates and caps used when turning flow text into a demand score for review.
        """
        self.ev_penetration_rate = ev_penetration_rate
        self.avg_charging_energy_kwh = avg_charging_energy_kwh
        self.daily_charge_probability = daily_charge_probability
        self.max_demand_for_norm = max_demand_for_norm

    def evaluate(
        self,
        rps_id: str,
        coordinates: List[float],
        scene_memory: Dict[str, Any],
        semantic_memory: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Return a zero to one demand score and small detail dict from one site memory for DCR and EB.
        """
        sem = normalize_semantic_memory(semantic_memory)
        commuting_flow_text = sem.get("commuting_flow", "")
        functional_zone_text = sem.get("functional_zone_type", "")
        flow_total = extract_traffic_total(commuting_flow_text)
        p_res, p_comm = extract_poi_counts(functional_zone_text)
        _, p_use_rcp = determine_zone_type_and_rcp_prob(p_res, p_comm)
        charging_demand_kwh = (
            float(flow_total)
            * float(self.ev_penetration_rate)
            * float(self.daily_charge_probability)
            * float(p_use_rcp)
            * float(self.avg_charging_energy_kwh)
        )
        demand_score = (
            min(charging_demand_kwh / self.max_demand_for_norm, 1.0)
            if self.max_demand_for_norm > 0
            else 0.0
        )
        details = {
            "flow_total": flow_total,
            "charging_demand_kwh": round(charging_demand_kwh, 2),
            "p_use_rcp": p_use_rcp,
        }
        return demand_score, details
