# -*- coding: utf-8 -*-
"""Economic Benefits Calculator."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import geopandas as gpd

from configs.data_config import REVIEW_AGENT_CONFIG
from tools.review_tool.common import (
    ChargingDemandPredictor,
    PlanningRecord,
    rps_entry_coordinates,
)

logger = logging.getLogger(__name__)


class EconomicBenefitsCalculator:
    def __init__(
        self,
        gas_price: float = REVIEW_AGENT_CONFIG["gas_price"],
        electricity_price: float = REVIEW_AGENT_CONFIG["electricity_price"],
        icev_consumption: float = REVIEW_AGENT_CONFIG["icev_consumption"],
        ev_efficiency: float = REVIEW_AGENT_CONFIG["ev_efficiency"],
    ):
        """
        Initialize the EconomicBenefitsCalculator with the fuel and power prices, simple ICEV and EV use rates, and demand helper.
        """
        self.gas_price = gas_price
        self.electricity_price = electricity_price
        self.icev_consumption = icev_consumption
        self.ev_efficiency = ev_efficiency
        self.delta_unit_cost = (gas_price * icev_consumption) - (electricity_price * ev_efficiency)
        self.demand_predictor = ChargingDemandPredictor()

    def _calculate_demand_from_memory(self, rps_id: str, rps_entry: Dict[str, Any]) -> float:
        """
        Return charging demand in kilowatt hours from predictor output for one RPS for EB calculation.
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
            logger.warning("EB demand for %s: %s", rps_id, e)
        return 0.0

    def calculate(
        self,
        planning_results: List[PlanningRecord],
        scene_semantic_memory: Dict[str, Any],
        poi_gdf: Optional[gpd.GeoDataFrame] = None,
        grid_points_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> float:
        """
        Compare the economic benefits of planned RPS to a reference maximum to calculate the EB score.
        """
        if not planning_results:
            logger.warning("EB: no planning results")
            return 0.0

        total_economic_benefits = 0.0
        for result in planning_results:
            rps_id = result.rps_id
            covered_demand_kwh = 0.0
            if rps_id in scene_semantic_memory:
                covered_demand_kwh = self._calculate_demand_from_memory(
                    rps_id, scene_semantic_memory[rps_id]
                )
            if covered_demand_kwh > 0:
                covered_vmt = covered_demand_kwh / self.ev_efficiency
                total_economic_benefits += covered_vmt * self.delta_unit_cost

        num_planned_rps = len(planning_results)
        all_rps_benefits = []
        for rps_id, rps_entry in scene_semantic_memory.items():
            potential_demand_kwh = self._calculate_demand_from_memory(rps_id, rps_entry)
            if potential_demand_kwh > 0:
                potential_vmt = potential_demand_kwh / self.ev_efficiency
                all_rps_benefits.append(potential_vmt * self.delta_unit_cost)

        if num_planned_rps > 0 and len(all_rps_benefits) > 0:
            all_rps_benefits.sort(reverse=True)
            max_economic_benefits = sum(all_rps_benefits[:num_planned_rps])
        elif total_economic_benefits > 0:
            max_economic_benefits = total_economic_benefits * 2.0
        else:
            max_economic_benefits = 1.0

        eb = (total_economic_benefits / max_economic_benefits) if max_economic_benefits > 0 else 0.0
        eb = max(0.0, min(1.0, eb))
        logger.info(
            "EB: total_benefit=%.2f max=%.2f eb=%.4f",
            total_economic_benefits,
            max_economic_benefits,
            eb,
        )
        return eb
