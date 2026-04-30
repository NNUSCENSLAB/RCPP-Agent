# -*- coding: utf-8 -*-
"""
Traffic dimension: distance to nearest intersection.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
from shapely.geometry import Point

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.registry import register_tool

logger = logging.getLogger(__name__)


class IntersectionExtractor:
    """
    Extract intersections from road network shapefile and calculate distance to nearest intersection (meters).
    """

    def __init__(
        self,
        road_network_path: Optional[str] = None,
        coordinate_tolerance: float = 0.00001,
    ) -> None:
        self.road_network_path = road_network_path
        self.coordinate_tolerance = coordinate_tolerance
        self._intersections_cache: Optional[List[Point]] = None
        self._projected_intersections_cache: Optional[List[Point]] = None
        self._cached_road_path: Optional[str] = None
        self.projected_crs = "EPSG:3857"

    def extract_intersections(self, road_network_path: Optional[str] = None) -> List[Point]:
        """
        Build a list of crossing points where many line ends meet after rounding coordinates.
        """
        path = road_network_path or self.road_network_path
        if path is None:
            raise ValueError("Road network path not provided, please provide in __init__ or extract_intersections")
        if self._intersections_cache is not None and self._cached_road_path == path:
            return self._intersections_cache
        logger.info("Starting to extract intersections from road network: %s", path)
        road_gdf = gpd.read_file(path)
        endpoints: List[Point] = []
        for _, row in road_gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == "LineString":
                coords = list(geom.coords)
                if len(coords) >= 2:
                    endpoints.append(Point(coords[0]))
                    endpoints.append(Point(coords[-1]))
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        endpoints.append(Point(coords[0]))
                        endpoints.append(Point(coords[-1]))
        endpoint_coords = [
            (
                round(p.x / self.coordinate_tolerance) * self.coordinate_tolerance,
                round(p.y / self.coordinate_tolerance) * self.coordinate_tolerance,
            )
            for p in endpoints
        ]
        coord_counts = Counter(endpoint_coords)
        intersection_coords = {c: n for c, n in coord_counts.items() if n >= 2}
        coord_to_point: Dict[tuple, Point] = {}
        for p in endpoints:
            rounded_coord = (
                round(p.x / self.coordinate_tolerance) * self.coordinate_tolerance,
                round(p.y / self.coordinate_tolerance) * self.coordinate_tolerance,
            )
            if rounded_coord in intersection_coords and rounded_coord not in coord_to_point:
                coord_to_point[rounded_coord] = p
        intersections = list(coord_to_point.values())
        self._intersections_cache = intersections
        self._cached_road_path = path
        return intersections

    def get_projected_intersections(self, road_network_path: Optional[str] = None) -> List[Point]:
        """
        Return crossing points in a flat map so distance math uses meters.
        """
        path = road_network_path or self.road_network_path
        if self._projected_intersections_cache is not None and self._cached_road_path == path:
            return self._projected_intersections_cache
        intersections_wgs84 = self.extract_intersections(road_network_path)
        if not intersections_wgs84:
            self._projected_intersections_cache = []
            return []
        gdf = gpd.GeoDataFrame(geometry=intersections_wgs84, crs="EPSG:4326")
        gdf_projected = gdf.to_crs(self.projected_crs)
        projected = [Point(p.x, p.y) for p in gdf_projected.geometry]
        self._projected_intersections_cache = projected
        return projected

    def calculate_distance_to_nearest_intersection(
        self,
        point: Point,
        road_network_path: Optional[str] = None,
    ) -> float:
        """
        Measure shortest map distance in meters from a site point to any projected crossing.
        """
        if isinstance(point, list):
            if len(point) >= 2:
                point = Point(point[0], point[1])
            else:
                raise ValueError("point coordinate list length is less than 2")
        elif not isinstance(point, Point):
            raise ValueError("point must be a Point object or coordinate list")
        projected_intersections = self.get_projected_intersections(road_network_path)
        if not projected_intersections:
            logger.warning("No intersections found, returning default distance 1000 meters")
            return 1000.0
        gdf_point = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
        gdf_point_projected = gdf_point.to_crs(self.projected_crs)
        projected_point = gdf_point_projected.geometry.iloc[0]
        min_distance = float("inf")
        for intersection in projected_intersections:
            d = projected_point.distance(intersection)
            if d < min_distance:
                min_distance = d
        return min_distance if min_distance != float("inf") else 1000.0


def _calculate_traffic_score(distance_to_intersection: float) -> Tuple[float, str]:
    """
    Map distance to nearest crossing into a smooth traffic score and short reason text.
    """
    if distance_to_intersection < 50:
        return (
            0.3,
            "Not suitable: close to intersection, high traffic congestion risk"
        )
    if distance_to_intersection < 100:
        return 0.6, "Not ideal: close to intersection, high traffic congestion risk"
    if distance_to_intersection < 200:
        return 0.8, "Basic compliant: close to intersection, meets basic traffic flow requirements"
    return 1.0, "Fully compliant: far from intersection, located on a smooth traffic segment"


def evaluate(
    rps_id: str,
    coordinates: List[float],
    scene_memory: Dict[str, Any],
    semantic_memory: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Score traffic smoothness from distance to nearest road crossing when a road file path exists in context.
    """
    del scene_memory, semantic_memory
    traffic_smooth_score = 1.0
    traffic_distance: Optional[float] = None
    traffic_evaluation = "Not evaluated: road network data not provided"
    road_network_path: Optional[str] = None
    if context:
        road_network_path = context.get("road_network_path") or context.get("road_network_shp")
    if road_network_path:
        try:
            ext = IntersectionExtractor(road_network_path=road_network_path)
            rps_point = Point(coordinates[0], coordinates[1])
            traffic_distance = ext.calculate_distance_to_nearest_intersection(
                rps_point, road_network_path
            )
            traffic_smooth_score, traffic_evaluation = _calculate_traffic_score(traffic_distance)
        except Exception as e:
            logger.warning("Traffic evaluation failed (RPS %s): %s", rps_id, e)
            traffic_smooth_score = 1.0
            traffic_evaluation = f"Evaluation failed: {e}"
    details: Dict[str, Any] = {
        "regulation_4_2_1_traffic": {
            "score": traffic_smooth_score,
            "distance_to_intersection_m": traffic_distance,
            "evaluation": traffic_evaluation,
            "status": "implemented" if road_network_path else "no_road_network",
        }
    }
    return traffic_smooth_score, details


@register_tool
class TrafficCalculationTool(BaseTool):
    name = "traffic_calculation"
    category = ToolCategory.MCDA_INDICATORS

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        road_network_path: str,
        point_lon: float,
        point_lat: float,
        coordinate_tolerance: float = 0.00001,
        **kwargs: Any,
    ) -> float:
        """
        Return traffic MCDA score and store helper values on the tool context.
        """
        _ = coordinate_tolerance  # BaseTool API; extractor uses default tolerance
        score, det = evaluate(
            "tool",
            [point_lon, point_lat],
            {},
            {},
            {"road_network_path": road_network_path},
        )
        traffic = det.get("regulation_4_2_1_traffic", {})
        d = traffic.get("distance_to_intersection_m")
        if ctx is not None:
            if d is not None:
                ctx.set("intersection_distance_m", d)
            ctx.set("traffic_score", score)
            ctx.set("traffic_details", det)
        return float(score)
