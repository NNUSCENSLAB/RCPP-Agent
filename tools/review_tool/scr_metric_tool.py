# -*- coding: utf-8 -*-
"""Service Coverage Rate Calculator."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import geopandas as gpd
from shapely.geometry import Point

from configs.data_config import REVIEW_AGENT_CONFIG
from tools.review_tool.common import PlanningRecord

logger = logging.getLogger(__name__)


class ServiceCoverageRateCalculator:
    def __init__(self, service_radius: float = REVIEW_AGENT_CONFIG["service_radius_r"]):
        """
        Initialize the ServiceCoverageRateCalculator with the service radius.
        """
        self.service_radius = service_radius

    def calculate(
        self,
        planning_results: List[PlanningRecord],
        scene_semantic_memory: Dict[str, Any],
        poi_gdf: Optional[gpd.GeoDataFrame] = None,
        grid_points_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> float:
        """
        Calculate the Service Coverage Rate (SCR) score from zero to one.
        """
        if not planning_results or poi_gdf is None or len(poi_gdf) == 0:
            logger.warning("SCR: missing planning results or POI data")
            return 0.0

        metric_crs = "EPSG:3857"
        poi_metric = poi_gdf.to_crs(metric_crs)
        covered_pois_set = set()

        for result in planning_results:
            rps_id = result.rps_id
            if rps_id not in scene_semantic_memory:
                continue
            coords = result.coordinates
            if len(coords) < 2:
                continue
            rps_point = Point(coords[0], coords[1])
            rps_gdf = gpd.GeoDataFrame([{"geometry": rps_point}], crs="EPSG:4326")
            rps_metric = rps_gdf.to_crs(metric_crs)
            buffer = rps_metric.geometry.iloc[0].buffer(self.service_radius)
            poi_in_buffer = poi_metric[poi_metric.geometry.within(buffer)]
            for idx in poi_in_buffer.index:
                covered_pois_set.add(idx)

        total_pois = len(poi_gdf)
        covered_pois_count = len(covered_pois_set)
        scr = (covered_pois_count / total_pois) * 100.0 if total_pois > 0 else 0.0
        scr_normalized = scr / 100.0
        logger.info(
            "SCR: covered_pois=%s total_pois=%s scr=%.4f",
            covered_pois_count,
            total_pois,
            scr_normalized,
        )
        return scr_normalized
