# -*- coding: utf-8 -*-
"""Network Loss Efficiency Calculator."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
from shapely.geometry import Point

from configs.data_config import REVIEW_AGENT_CONFIG
from tools.review_tool.common import (
    PlanningRecord,
    extract_traffic_outflow,
    normalize_semantic_memory,
    rps_entry_coordinates,
)

logger = logging.getLogger(__name__)

_METRIC_CRS = "EPSG:3857"


class NetworkLossEfficiencyCalculator:
    def __init__(
        self,
        P_rate: float = REVIEW_AGENT_CONFIG["P_rate"],
        V_nom: float = REVIEW_AGENT_CONFIG["V_nom"],
        rho: float = REVIEW_AGENT_CONFIG["rho"],
        baseline_ratio: float = REVIEW_AGENT_CONFIG["baseline_ratio"],
        penalty_gamma: float = 0.3,
        penalty_beta: float = 0.3,
        E_charge: float = REVIEW_AGENT_CONFIG["E_charge"],
        n_per_rps: int = REVIEW_AGENT_CONFIG["n_per_rps"],
        mu_charges_per_day: float = REVIEW_AGENT_CONFIG["mu_charges_per_day"],
        ev_penetration_rate: float = 0.5407,
        daily_charge_probability: float = 0.33,
        rcp_usage_probability: float = 0.5,
        existing_rcp_path: Optional[Path] = None,
    ):
        """Store review line loss settings optional existing RCP file path and baseline cache fields for NLE."""
        self.P_rate = P_rate
        self.V_nom = V_nom
        self.rho = rho
        self.baseline_ratio = baseline_ratio
        self.penalty_gamma = penalty_gamma
        self.penalty_beta = penalty_beta
        self.E_charge = E_charge
        self.n_per_rps = n_per_rps
        self.mu_charges_per_day = mu_charges_per_day
        self.ev_penetration_rate = ev_penetration_rate
        self.daily_charge_probability = daily_charge_probability
        self.rcp_usage_probability = rcp_usage_probability
        self.temporal_periods = REVIEW_AGENT_CONFIG["temporal_periods"]
        self._cached_base_losses: Optional[Dict[str, float]] = None
        self._cached_base_valid_count: int = 0
        self.existing_rcp_path = existing_rcp_path
        self.existing_rps_gdf: Optional[gpd.GeoDataFrame] = None
        self._baseline_rps_ids_for_log: List[str] = []
        if existing_rcp_path:
            ep = Path(existing_rcp_path)
            if ep.exists():
                self._load_existing_rcp(ep)
            else:
                logger.warning("Existing RCP data file not found: %s", existing_rcp_path)

    def _extract_outflow(self, rps_entry: Dict[str, Any]) -> float:
        """Return positive commuting outflow from one RPS semantic memory or zero when text is missing."""
        semantic = normalize_semantic_memory(rps_entry["memory_node"]["semantic_memory"])
        commuting_flow_text = semantic.get("commuting_flow", "")
        if not commuting_flow_text:
            return 0.0
        outflow = extract_traffic_outflow(commuting_flow_text)
        return outflow if outflow > 0 else 0.0

    def _calculate_daily_demand(self, outflow: float) -> float:
        """Scale outflow by EV share and charging habit to daily demand in kilowatt hours for NLE."""
        return (
            float(outflow)
            * self.ev_penetration_rate
            * self.daily_charge_probability
            * self.rcp_usage_probability
            * self.E_charge
        )

    def _calculate_temporal_demand(self, daily_demand: float) -> Dict[str, float]:
        """Split daily demand across named time windows using each window factor and hour length."""
        total_weight = 0.0
        period_weights: Dict[str, float] = {}
        for period_name, period_info in self.temporal_periods.items():
            weight = period_info["factor"] * period_info["duration_hours"]
            period_weights[period_name] = weight
            total_weight += weight
        return {
            pn: daily_demand * (w / total_weight) if total_weight > 0 else 0.0
            for pn, w in period_weights.items()
        }

    def _calculate_charging_piles_period(self, period_demand: float, period_duration: float) -> float:
        """Pick active pile count for one window from demand power limit and per RPS service cap."""
        theoretical_piles = period_demand / (self.P_rate * period_duration) if period_duration > 0 else 0.0
        service_capacity = self.n_per_rps * (self.mu_charges_per_day * period_duration / 24.0)
        actual_piles = min(theoretical_piles, service_capacity)
        return max(0.0, actual_piles)

    def _grid_distances_km(
        self, coords_list: List[List[float]], grid_points_gdf: gpd.GeoDataFrame
    ) -> List[float]:
        """Vectorised distance from each coord pair to the nearest grid point (km)."""
        if not coords_list:
            return []
        grid_metric = grid_points_gdf.to_crs(_METRIC_CRS)
        points = [Point(c[0], c[1]) for c in coords_list if len(c) >= 2]
        query_gdf = gpd.GeoDataFrame(
            [{"geometry": p} for p in points], crs="EPSG:4326"
        ).to_crs(_METRIC_CRS)
        distances: List[float] = []
        for geom in query_gdf.geometry:
            min_dist = grid_metric.geometry.distance(geom).min()
            distances.append(min_dist / 1000.0)
        return distances

    def _calculate_power_loss(self, distance_km: float, power_kw: float) -> float:
        """Approximate line loss from distance in kilometers and active power in kilowatts for NLE."""
        r_ij = self.rho * distance_km
        return r_ij * (power_kw**2) / (self.V_nom**2)

    def _load_existing_rcp(self, path: Path) -> None:
        """Load existing RCP points from a shapefile path or folder into GeoDataFrame for NLE baseline."""
        try:
            if path.is_dir():
                shp_files = list(path.glob("*.shp"))
                if not shp_files:
                    logger.warning("No .shp file found in directory: %s", path)
                    return
                path = shp_files[0]
            elif not path.exists():
                if path.suffix != ".shp":
                    shp_path = path.with_suffix(".shp")
                    path = shp_path if shp_path.exists() else path
                if not path.exists():
                    logger.warning("Existing RCP data file not found: %s", path)
                    return
            gdf = gpd.read_file(str(path))
            if "wgs84_lon" not in gdf.columns or "wgs84_lat" not in gdf.columns:
                logger.warning("Existing RCP data missing wgs84_lon / wgs84_lat columns")
                return
            if "geometry" not in gdf.columns or gdf.geometry.isna().all():
                gdf["geometry"] = gpd.points_from_xy(gdf["wgs84_lon"], gdf["wgs84_lat"], crs="EPSG:4326")
            elif gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            self.existing_rps_gdf = gdf
            logger.info("Loaded %s existing RCP points", len(gdf))
        except Exception as e:
            logger.error("Failed to load existing RCP data: %s", e)
            self.existing_rps_gdf = None

    def _nearest_outflows_for_coords(
        self,
        coords_list: List[List[float]],
        scene_semantic_memory: Dict[str, Any],
    ) -> List[float]:
        """Vectorised nearest-neighbour lookup: for each coord return the outflow of the nearest RPS memory."""
        if not scene_semantic_memory or not coords_list:
            return [0.0] * len(coords_list)

        mem_coords = []
        mem_outflows = []
        for m in scene_semantic_memory.values():
            coords = rps_entry_coordinates(m)
            if len(coords) < 2:
                continue
            mem_coords.append(coords)
            mem_outflows.append(self._extract_outflow(m))
        if not mem_coords:
            return [0.0] * len(coords_list)

        mem_gdf = gpd.GeoDataFrame(
            [{"geometry": Point(c[0], c[1])} for c in mem_coords], crs="EPSG:4326"
        ).to_crs(_METRIC_CRS)

        query_gdf = gpd.GeoDataFrame(
            [{"geometry": Point(c[0], c[1])} for c in coords_list if len(c) >= 2],
            crs="EPSG:4326",
        ).to_crs(_METRIC_CRS)

        result: List[float] = []
        for geom in query_gdf.geometry:
            distances = mem_gdf.geometry.distance(geom)
            nearest_pos = int(distances.to_numpy().argmin())
            result.append(mem_outflows[nearest_pos])
        return result

    def _accumulate_losses(
        self,
        coords_list: List[List[float]],
        outflows: List[float],
        grid_points_gdf: gpd.GeoDataFrame,
    ):
        """Return (losses_by_period, valid_count) for a batch of (coord, outflow) pairs."""
        losses_by_period: Dict[str, float] = {pn: 0.0 for pn in self.temporal_periods}
        valid_count = 0

        valid_coords = [c for c, o in zip(coords_list, outflows) if o > 0 and len(c) >= 2]
        valid_outflows = [o for o in outflows if o > 0]
        if not valid_coords:
            return losses_by_period, valid_count

        distances_km = self._grid_distances_km(valid_coords, grid_points_gdf)

        for outflow, dist_km in zip(valid_outflows, distances_km):
            if dist_km <= 0:
                continue
            valid_count += 1
            daily_demand = self._calculate_daily_demand(outflow)
            temporal_demands = self._calculate_temporal_demand(daily_demand)
            for period_name, period_demand in temporal_demands.items():
                period_duration = self.temporal_periods[period_name]["duration_hours"]
                charging_piles = self._calculate_charging_piles_period(period_demand, period_duration)
                period_power = self.P_rate * charging_piles
                losses_by_period[period_name] += self._calculate_power_loss(dist_km, period_power)

        return losses_by_period, valid_count

    def calculate(
        self,
        planning_results: List[PlanningRecord],
        scene_semantic_memory: Dict[str, Any],
        poi_gdf: Optional[gpd.GeoDataFrame] = None,
        grid_points_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> float:
        """Compare planned grid tied losses to a baseline case and return NLE from zero to one for Review Agent."""
        self._baseline_rps_ids_for_log = []
        if not planning_results or grid_points_gdf is None or len(grid_points_gdf) == 0:
            logger.warning("NLE: missing planning results or grid points")
            return 0.0

        plan_coords = []
        plan_outflows = []
        for result in planning_results:
            if result.rps_id not in scene_semantic_memory:
                continue
            rps_entry = scene_semantic_memory[result.rps_id]
            outflow = self._extract_outflow(rps_entry)
            plan_coords.append(result.coordinates)
            plan_outflows.append(outflow)

        plan_losses_by_period, plan_valid_count = self._accumulate_losses(
            plan_coords, plan_outflows, grid_points_gdf
        )

        if self._cached_base_losses is not None:
            base_losses_by_period = self._cached_base_losses.copy()
            base_valid_count = self._cached_base_valid_count
            logger.info("Using cached baseline scenario loss data")
        else:
            base_losses_by_period = {pn: 0.0 for pn in self.temporal_periods}
            base_valid_count = 0

        if self._cached_base_losses is None:
            if self.existing_rps_gdf is not None and len(self.existing_rps_gdf) > 0:
                base_coords = [
                    [float(row["wgs84_lon"]), float(row["wgs84_lat"])]
                    for _, row in self.existing_rps_gdf.iterrows()
                    if row.get("wgs84_lon") is not None and row.get("wgs84_lat") is not None
                ]
                base_outflows = self._nearest_outflows_for_coords(base_coords, scene_semantic_memory)
                base_losses_by_period, base_valid_count = self._accumulate_losses(
                    base_coords, base_outflows, grid_points_gdf
                )
            else:
                logger.info("No existing RCP data provided; using baseline_ratio to construct reference scenario")
                all_rps_ids = list(scene_semantic_memory.keys())
                rps_demand_scores = []
                for rps_id in all_rps_ids:
                    rps_entry = scene_semantic_memory[rps_id]
                    outflow = self._extract_outflow(rps_entry)
                    rps_demand_scores.append((rps_id, self._calculate_daily_demand(outflow) if outflow > 0 else 0.0))
                rps_demand_scores.sort(key=lambda x: x[1], reverse=True)
                num_baseline_rps = max(1, int(len(all_rps_ids) * self.baseline_ratio))
                self._baseline_rps_ids_for_log = [r for r, _ in rps_demand_scores[:num_baseline_rps]]

                base_coords = [
                    rps_entry_coordinates(scene_semantic_memory[rid])
                    for rid in self._baseline_rps_ids_for_log
                    if len(rps_entry_coordinates(scene_semantic_memory[rid])) >= 2
                ]
                base_outflows = [
                    self._extract_outflow(scene_semantic_memory[rid])
                    for rid in self._baseline_rps_ids_for_log
                    if len(rps_entry_coordinates(scene_semantic_memory[rid])) >= 2
                ]
                base_losses_by_period, base_valid_count = self._accumulate_losses(
                    base_coords, base_outflows, grid_points_gdf
                )

            self._cached_base_losses = base_losses_by_period.copy()
            self._cached_base_valid_count = base_valid_count

        total_plan_loss = sum(plan_losses_by_period.values())
        total_base_loss = sum(base_losses_by_period.values())
        lbar_plan = total_plan_loss / plan_valid_count if plan_valid_count > 0 else 0.0
        lbar_base = total_base_loss / base_valid_count if base_valid_count > 0 else 0.0
        if (lbar_base + lbar_plan) > 0:
            nle_match = lbar_base / (lbar_base + lbar_plan)
        else:
            nle_match = 1.0
        s = plan_valid_count / max(1, base_valid_count)
        penalty = 1.0 / (1.0 + (s**self.penalty_gamma)) if s > 0 else 1.0
        nle = nle_match * (penalty**self.penalty_beta)
        nle = max(0.0, min(1.0, nle))
        logger.info(
            "NLE: plan_loss=%.4f base_loss=%.4f plan_n=%s base_n=%s nle=%.4f",
            total_plan_loss,
            total_base_loss,
            plan_valid_count,
            base_valid_count,
            nle,
        )
        if nle <= 0.01:
            n_base = (
                len(self._baseline_rps_ids_for_log)
                if self._baseline_rps_ids_for_log
                else (len(self.existing_rps_gdf) if self.existing_rps_gdf is not None else 0)
            )
            logger.warning(
                "NLE near 0: plan_loss=%.4f base_loss=%.4f plan_rps=%s base_ref=%s",
                total_plan_loss,
                total_base_loss,
                len(planning_results),
                n_base,
            )
        return nle
