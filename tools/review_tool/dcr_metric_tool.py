# -*- coding: utf-8 -*-
"""Demand Coverage Rate Calculator."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import geopandas as gpd

from configs.data_config import REVIEW_AGENT_CONFIG
from tools.review_tool.common import (
    ChargingDemandPredictor,
    PlanningRecord,
    extract_traffic_outflow,
    extract_traffic_total,
    normalize_semantic_memory,
    rps_entry_coordinates,
)

logger = logging.getLogger(__name__)


class DemandSideRCPDemandCalculator:
    def __init__(
        self,
        ev_penetration_rate: float = 0.5407,
        driving_license_ownership_ratio: float = 0.13,
        daily_charge_probability: float = 0.33,
        rcp_usage_probability: float = 0.5,
        avg_charging_energy_kwh: float = 48.0,
    ):
        """
        Initialize the DemandSideRCPDemandCalculator with the fixed rates.
        """
        self.ev_penetration_rate = ev_penetration_rate
        self.driving_license_ownership_ratio = driving_license_ownership_ratio
        self.daily_charge_probability = daily_charge_probability
        self.rcp_usage_probability = rcp_usage_probability
        self.avg_charging_energy_kwh = avg_charging_energy_kwh

    def calculate_demand_side_rcp_demand(self, flow_total: float) -> float:
        """
        Scale total traffic flow by stored rates to get demand side RCP energy in kilowatt hours.
        """
        return (
            float(flow_total)
            * self.ev_penetration_rate
            * self.driving_license_ownership_ratio
            * self.daily_charge_probability
            * self.rcp_usage_probability
            * self.avg_charging_energy_kwh
        )


class DemandCoverageRateCalculator:
    def __init__(
        self,
        E_charge: float = REVIEW_AGENT_CONFIG["E_charge"],
        n_per_rps: int = REVIEW_AGENT_CONFIG["n_per_rps"],
        mu_charges_per_day: float = REVIEW_AGENT_CONFIG["mu_charges_per_day"],
    ):
        """
        Connect flow based demand helper with per charge energy and daily use limits for DCR review.
        """
        self.demand_side_rcp_demand_calculator = DemandSideRCPDemandCalculator()
        self.demand_predictor = ChargingDemandPredictor()
        self.E_charge = E_charge
        self.n_per_rps = n_per_rps
        self.mu_charges_per_day = mu_charges_per_day

    def _extract_grid_id(self, rps_entry: Dict[str, Any]) -> Optional[str]:
        """
        Parse commuting flow text in memory to a simple grid id string or None when missing.
        """
        semantic = normalize_semantic_memory(rps_entry["memory_node"]["semantic_memory"])
        commuting_flow_text = semantic.get("commuting_flow", "")
        if not commuting_flow_text:
            return None
        flow_total = extract_traffic_total(commuting_flow_text)
        return str(int(flow_total)) if flow_total > 0 else None

    def _group_by_grid(self, scene_semantic_memory: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Map each grid id to every RPS id whose memory yields that grid id for DCR aggregation.
        """
        grid_to_rps: Dict[str, List[str]] = {}
        for rps_id, rps_entry in scene_semantic_memory.items():
            grid_id = self._extract_grid_id(rps_entry)
            if grid_id:
                grid_to_rps.setdefault(grid_id, []).append(rps_id)
        return grid_to_rps

    def _calculate_demand_from_memory(self, rps_id: str, rps_entry: Dict[str, Any]) -> float:
        """
        Return charging demand in kilowatt hours from predictor output for one RPS or zero on failure.
        """
        try:
            mn = rps_entry["memory_node"]
            _, details = self.demand_predictor.evaluate(
                rps_id=rps_id,
                coordinates=rps_entry_coordinates(rps_entry),
                scene_memory=mn["scene_memory"],
                semantic_memory=mn["semantic_memory"],
                context=None,
            )
            if isinstance(details, dict):
                return float(details.get("charging_demand_kwh", 0.0))
        except Exception as e:
            logger.warning("DCR demand from memory failed for %s: %s", rps_id, e)
        return 0.0

    def _calculate_grid_demand(
        self, grid_id: str, rps_ids: List[str], scene_semantic_memory: Dict[str, Any]
    ) -> float:
        """
        Use first RPS in the grid and outflow text to estimate total grid RCP demand in kilowatt hours.
        """
        if not rps_ids or rps_ids[0] not in scene_semantic_memory:
            return 0.0
        rps_entry = scene_semantic_memory[rps_ids[0]]
        semantic = normalize_semantic_memory(rps_entry["memory_node"]["semantic_memory"])
        commuting_flow_text = semantic.get("commuting_flow", "")
        if not commuting_flow_text:
            return 0.0
        outflow = extract_traffic_outflow(commuting_flow_text)
        if outflow <= 0:
            return 0.0
        return self.demand_side_rcp_demand_calculator.calculate_demand_side_rcp_demand(outflow)

    def _calculate_grid_supply(self, planned_rps_ids: List[str]) -> float:
        """
        Sum over planned RPS of energy per charge times piles per site times mean charges per day for DCR calculation.
        """
        supply_per_rps = self.E_charge * self.n_per_rps * self.mu_charges_per_day
        return len(planned_rps_ids) * supply_per_rps

    def calculate(
        self,
        planning_results: List[PlanningRecord],
        scene_semantic_memory: Dict[str, Any],
        poi_gdf: Optional[gpd.GeoDataFrame] = None,
        grid_points_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> float:
        """
        Weight grid level supply over demand then soften ratios above one for Review Agent DCR score.
        """
        if not planning_results:
            logger.warning("DCR: no planning results")
            return 0.0

        grid_to_rps = self._group_by_grid(scene_semantic_memory)
        if not grid_to_rps:
            logger.warning("DCR: no grid grouping from memories")
            return 0.0

        grid_to_planned_rps: Dict[str, List[str]] = {}
        for result in planning_results:
            rps_id = result.rps_id
            if rps_id not in scene_semantic_memory:
                continue
            rps_entry = scene_semantic_memory[rps_id]
            grid_id = self._extract_grid_id(rps_entry)
            if grid_id:
                grid_to_planned_rps.setdefault(grid_id, []).append(rps_id)

        total_supply_demand = 0.0
        total_demand = 0.0
        grid_stats = []

        for grid_id, rps_ids in grid_to_rps.items():
            planned_rps_ids = grid_to_planned_rps.get(grid_id, [])
            if not planned_rps_ids:
                continue
            grid_total_demand = self._calculate_grid_demand(grid_id, rps_ids, scene_semantic_memory)
            grid_supply_demand = self._calculate_grid_supply(planned_rps_ids)
            total_demand += grid_total_demand
            total_supply_demand += grid_supply_demand
            grid_stats.append(
                {
                    "grid_id": grid_id,
                    "grid_total_demand": grid_total_demand,
                    "grid_supply_demand": grid_supply_demand,
                    "grid_dcr": grid_supply_demand / grid_total_demand if grid_total_demand > 0 else 0.0,
                }
            )

        if total_demand == 0:
            logger.warning("DCR: total demand is 0")
            return 0.0

        weighted_dcr_sum = sum(s["grid_dcr"] * s["grid_total_demand"] for s in grid_stats)
        overall_dcr = weighted_dcr_sum / total_demand if total_demand > 0 else 0.0
        raw_dcr = overall_dcr
        if overall_dcr > 1.0:
            excess = overall_dcr - 1.0
            overall_dcr = max(0.5, 1.0 - 0.1 * excess)

        logger.info(
            "DCR: grids=%s total_demand=%.2f supply=%.2f raw=%.4f final=%.4f",
            len(grid_stats),
            total_demand,
            total_supply_demand,
            raw_dcr,
            overall_dcr,
        )
        return overall_dcr
