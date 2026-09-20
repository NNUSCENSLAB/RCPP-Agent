"""
Generate semantic memory inference input JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from rcpp_core.semantic_rules import deterministic_semantic_memory, derive_semantic_facts
from tools.base import BaseTool, ToolCategory, ToolContext
from tools.memory_construction_tool.od_flow_tool import load_od_flow_data, load_od_flow_data_from_gdfs
from tools.memory_construction_tool.poi_statistics_tool import POIStatisticsTool
from tools.registry import register_tool


def calculate_grid_distance(
    parking_point: Point, grid_pois: gpd.GeoDataFrame, metric_crs: str = "EPSG:3857"
) -> float:
    if grid_pois.empty:
        return -1.0
    parking_metric = gpd.GeoDataFrame([1], geometry=[parking_point], crs="EPSG:4326").to_crs(metric_crs)
    grid_metric = grid_pois.to_crs(metric_crs)
    distances = grid_metric.geometry.distance(parking_metric.geometry.iloc[0])
    min_distance_m = distances.min()
    return float(min_distance_m / 1000.0)


def get_grid_accessibility_level(grid_distance_km: float) -> str:
    if grid_distance_km < 0:
        return "unknown"
    distance_m = grid_distance_km * 1000
    if distance_m < 500:
        return "excellent"
    if distance_m < 1000:
        return "good"
    if distance_m < 2000:
        return "moderate"
    if distance_m < 5000:
        return "poor"
    return "very_poor"


def _build_semantic_records(poi_tool: POIStatisticsTool, flow_summary: pd.DataFrame) -> List[Dict]:
    parking_latlon = poi_tool.load_parking()
    poi_latlon = poi_tool.load_pois()
    grid_pois = cast(
        gpd.GeoDataFrame,
        poi_latlon[poi_latlon["category"] == "grid"].copy(),
    )
    flow_summary_gdf = gpd.GeoDataFrame(flow_summary, geometry="geometry", crs="EPSG:4326")
    parking_metric = parking_latlon.to_crs(poi_tool.metric_crs)

    flow_for_join = cast(
        gpd.GeoDataFrame,
        flow_summary_gdf[["locations", "inflow", "outflow", "geometry"]],
    )
    parking_with_grid = gpd.sjoin(
        parking_latlon,
        flow_for_join,
        how="left",
        predicate="within",
    )
    parking_with_grid = parking_with_grid[~parking_with_grid.index.duplicated(keep="first")]

    poi_records = poi_tool.build_intermediate_records()
    poi_dict = {record["rps_id"]: record for record in poi_records}

    training_records: List[Dict] = []

    for idx, _parking_row in parking_metric.iterrows():
        original_idx = parking_latlon.index[idx]
        rps_id = parking_latlon.iloc[idx]["RPS_id"]

        poi_record = poi_dict.get(rps_id, {})
        poi_counts_1km = poi_record.get("poi_counts_1km", {"residential": 0, "commercial": 0})
        poi_counts_sensitive = poi_record.get("poi_counts_sensitive_0.1km", {"sensitive": 0})

        if original_idx in parking_with_grid.index:
            total_inflow = (
                float(parking_with_grid.loc[original_idx, "inflow"])
                if pd.notna(parking_with_grid.loc[original_idx, "inflow"])
                else 0.0
            )
            total_outflow = (
                float(parking_with_grid.loc[original_idx, "outflow"])
                if pd.notna(parking_with_grid.loc[original_idx, "outflow"])
                else 0.0
            )
        else:
            total_inflow = 0.0
            total_outflow = 0.0

        original_point = parking_latlon.iloc[idx].geometry
        grid_distance = calculate_grid_distance(original_point, grid_pois, poi_tool.metric_crs)
        coordinates = (float(parking_latlon.iloc[idx].geometry.x), float(parking_latlon.iloc[idx].geometry.y))

        angle = None
        if "angle" in parking_latlon.columns:
            try:
                angle = float(parking_latlon.iloc[idx]["angle"])
            except (ValueError, TypeError):
                angle = None
        elif "angel" in parking_latlon.columns:
            try:
                angle = float(parking_latlon.iloc[idx]["angel"])
            except (ValueError, TypeError):
                angle = None
        if angle is None:
            angle = 0.0

        bsv_image = None
        if "Pic" in parking_latlon.columns:
            pic_value = parking_latlon.iloc[idx]["Pic"]
            if pd.notna(pic_value):
                bsv_image = str(pic_value)

        statistical_summary = {
            "rps_id": rps_id,
            "coordinates": {"lng": coordinates[0], "lat": coordinates[1], "angle": angle},
            "poi_counts_1km": poi_counts_1km,
            "poi_counts_sensitive_0.1km": poi_counts_sensitive,
            "traffic_flow_stats": {"total_inflow": int(total_inflow), "total_outflow": int(total_outflow)},
            "infrastructure_dist": {
                "grid_distance_km": round(grid_distance, 3) if grid_distance >= 0 else -1,
                "grid_accessibility_level": get_grid_accessibility_level(grid_distance),
            },
        }
        statistical_summary["angle"] = angle
        if bsv_image:
            statistical_summary["bsv_image"] = bsv_image
        rule_facts = derive_semantic_facts(statistical_summary)
        gold_fallback = deterministic_semantic_memory(statistical_summary)

        training_record = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"RPS ID: {rps_id}\n\n"
                        f"【Statistical Summary】\n{json.dumps(statistical_summary, ensure_ascii=False, indent=2)}\n\n"
                        f"【Task Requirements】\nGenerate semantic_reasoning as a JSON object including: "
                        f"functional_zone_type (brief functional zone definition based on 1km buffer POI counts for residential and commercial), "
                        f"commuting_flow (traffic flow analysis and charging impact), "
                        f"sensitive_constraints (compliance risk identification based on 0.1km buffer sensitive POI counts), "
                        f"grid_accessibility (grid accessibility assessment).\n\n"
                        f"Note: This uses a layered buffer strategy - 1km for functional zone identification (residential/commercial), "
                        f"0.1km for sensitive facility compliance assessment.\n\n"
                        f"【Rule Facts - copy exactly】\n{json.dumps(rule_facts, ensure_ascii=False, indent=2)}\n\n"
                        f"The semantic_reasoning will be used to build a knowledge graph where:\n"
                        f"- RPS node (id={rps_id}) connects to POI entities via spatial relationships\n"
                        f"- RPS node connects to grid infrastructure via distance relationships\n"
                        f"- RPS node connects to traffic flow regions via flow relationships\n"
                        f"- Semantic attributes enrich the RPS node properties for graph-based reasoning\n\n"
                        f"Do not recalculate labels or numbers. /no_think"
                    ),
                },
                {
                    "role": "assistant",
                    # Deterministic, valid target/fallback. Human-authored prose from the
                    # audited legacy dataset may replace these strings for QLoRA training.
                    "content": json.dumps(gold_fallback, ensure_ascii=False),
                },
            ]
        }
        training_records.append(training_record)

    return training_records


def process_parking_spots(
    parking_path: Path,
    poi_dir: Path,
    regions_path: Path,
    od_flows_path: Path,
    output_path: Path,
) -> List[Dict]:
    poi_tool = POIStatisticsTool(
        parking_path=parking_path,
        poi_dir=poi_dir,
        residential_commercial_buffer_m=1000.0,
        sensitive_buffer_m=100.0,
    )
    flow_summary = load_od_flow_data(regions_path, od_flows_path)
    return _build_semantic_records(poi_tool, flow_summary)


def generate_semantic_inference_from_spatialite(
    db_path: Path,
    output_path: Path,
    area: str,
    tables: Optional[Dict[str, str]] = None,
) -> int:
    tables = tables or {}
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    rps = gpd.read_file(db_path, layer=tables.get("rps", "rps"))
    poi = gpd.read_file(db_path, layer=tables.get("poi", "poi"))
    regions = gpd.read_file(db_path, layer=tables.get("od_region", "od_region"))
    od_flow = gpd.read_file(db_path, layer=tables.get("od_flow", "od_flow"))

    if area and "area" in rps.columns:
        rps = rps[rps["area"].astype(str) == str(area)]
        if rps.empty:
            raise ValueError(f"After filtering by area={area}, RPS is empty")

    for col in ("name", "category"):
        if col not in poi.columns:
            alt = col.capitalize()
            if alt in poi.columns:
                poi = poi.rename(columns={alt: col})
    if "name" not in poi.columns or "category" not in poi.columns:
        raise ValueError("POI layer must contain name and category columns")

    flow_summary = load_od_flow_data_from_gdfs(regions, od_flow)
    poi_tool = POIStatisticsTool(
        parking_path=Path("."),
        poi_dir=Path("."),
        residential_commercial_buffer_m=1000.0,
        sensitive_buffer_m=100.0,
        parking_gdf=cast(gpd.GeoDataFrame, rps),
        poi_gdf=cast(gpd.GeoDataFrame, poi),
    )
    records = _build_semantic_records(poi_tool, flow_summary)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Generated semantic inference input: {output_path}, contains {len(records)} records")
    return len(records)


@register_tool
class SemanticInferenceInputTool(BaseTool):
    name = "semantic_inference_input"
    category = ToolCategory.MEMORY_CONSTRUCTION

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        db_path: Optional[Path] = None,
        output_json: Optional[Path] = None,
        area: str = "gulou",
        tables: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> int:
        if db_path is None or output_json is None:
            raise ValueError("db_path and output_json are required")
        n = generate_semantic_inference_from_spatialite(db_path, output_json, area, tables)
        if ctx is not None:
            ctx.set("semantic_inference_input_path", str(Path(output_json).resolve()))
        return n
