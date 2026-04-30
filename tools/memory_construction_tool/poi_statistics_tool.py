from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from configs.data_config import POI_FILENAME_MAPPING
from tools.base import BaseTool, ToolCategory, ToolContext
from tools.registry import register_tool


@dataclass
class POIStatisticsTool:
    """
    Layered buffer POI statistics for RPSs and intermediate records for semantic memory inference.
    """

    parking_path: Path
    poi_dir: Path
    radius_m: float = 1000.0
    metric_crs: str = "EPSG:3857"
    latlon_crs: str = "EPSG:4326"

    residential_commercial_buffer_m: float = 1000.0
    sensitive_buffer_m: float = 100.0
    parking_gdf: Optional[gpd.GeoDataFrame] = field(default=None)
    poi_gdf: Optional[gpd.GeoDataFrame] = field(default=None)

    def load_parking(self) -> gpd.GeoDataFrame:
        if self.parking_gdf is not None:
            parking_gdf = self.parking_gdf.copy()
        else:
            parking_gdf = gpd.read_file(self.parking_path)
        if parking_gdf.crs is None:
            parking_gdf.set_crs(self.latlon_crs, inplace=True)
        else:
            parking_gdf = parking_gdf.to_crs(self.latlon_crs)
        if "ID" not in parking_gdf.columns:
            raise ValueError("Parking spot data must include an ID column")
        parking_gdf["RPS_id"] = parking_gdf["ID"].astype(str)
        return parking_gdf

    def load_pois(self) -> gpd.GeoDataFrame:
        if self.poi_gdf is not None:
            poi_gdf = self.poi_gdf.copy()
            if poi_gdf.crs is None:
                poi_gdf.set_crs(self.latlon_crs, inplace=True)
            else:
                poi_gdf = poi_gdf.to_crs(self.latlon_crs)
            required = {"name", "category", "geometry"}
            if not required.issubset(set(poi_gdf.columns)):
                raise ValueError(f"POI GeoDataFrame must contain columns: {required}")
            return poi_gdf

        records: List[pd.DataFrame] = []

        for csv_path in self.poi_dir.glob("*.csv"):
            key_part = csv_path.stem.rsplit("_", 1)[-1].lower()
            category_label = POI_FILENAME_MAPPING.get(key_part)

            if not category_label:
                continue

            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
            df["category"] = category_label
            df["name"] = df["name"].astype(str)
            df["geometry"] = [Point(lon, lat) for lon, lat in zip(df["lng"], df["lat"])]
            records.append(
                pd.DataFrame(
                    {"name": df["name"], "category": df["category"], "geometry": df["geometry"]}
                )
            )

        if not records:
            raise ValueError(f"No valid POI CSV files found under {self.poi_dir}")

        poi_df = pd.concat(records, ignore_index=True)
        return gpd.GeoDataFrame(poi_df, geometry="geometry", crs=self.latlon_crs)

    def build_intermediate_records(self) -> List[Dict[str, object]]:
        parking_latlon = self.load_parking()
        poi_latlon = self.load_pois()

        parking_metric = parking_latlon.to_crs(self.metric_crs)
        poi_metric = poi_latlon.to_crs(self.metric_crs)

        residential_pois = poi_metric[poi_metric["category"] == "residential"].copy()
        commercial_pois = poi_metric[poi_metric["category"] == "commercial"].copy()
        sensitive_pois = poi_metric[
            poi_metric["category"].isin(["medical", "education", "public_admin"])
        ].copy()
        other_pois = poi_metric[
            ~poi_metric["category"].isin(
                [
                    "residential",
                    "commercial",
                    "medical",
                    "education",
                    "public_admin",
                    "grid",
                ]
            )
        ].copy()

        results: List[Dict[str, object]] = []
        for idx, parking_row in parking_metric.iterrows():
            buffer_1km = parking_row.geometry.buffer(self.residential_commercial_buffer_m)
            residential_mask = residential_pois.geometry.within(buffer_1km)
            commercial_mask = commercial_pois.geometry.within(buffer_1km)

            residential_count = residential_mask.sum()
            commercial_count = commercial_mask.sum()
            residential_names = (
                residential_pois[residential_mask]["name"].tolist() if residential_count > 0 else []
            )
            commercial_names = (
                commercial_pois[commercial_mask]["name"].tolist() if commercial_count > 0 else []
            )

            buffer_100m = parking_row.geometry.buffer(self.sensitive_buffer_m)
            sensitive_mask = sensitive_pois.geometry.within(buffer_100m)

            sensitive_count = sensitive_mask.sum()
            sensitive_names = (
                sensitive_pois[sensitive_mask]["name"].tolist() if sensitive_count > 0 else []
            )

            other_mask = other_pois.geometry.within(buffer_1km)
            other_subset = cast(gpd.GeoDataFrame, other_pois.loc[other_mask].copy())

            counts_1km = {
                "residential": int(residential_count),
                "commercial": int(commercial_count),
            }
            names_1km = {
                "residential": residential_names,
                "commercial": commercial_names,
            }

            other_counts: Dict[str, int] = {}
            other_names: Dict[str, List[Any]] = {}
            if not other_subset.empty:
                for category, group_df in other_subset.groupby("category"):
                    if category not in ["grid"]:
                        other_counts[str(category)] = len(group_df)
                        other_names[str(category)] = group_df["name"].tolist()

            sensitive_counts = {"sensitive": int(sensitive_count)}
            sensitive_names_dict = {"sensitive": sensitive_names}

            original_row = parking_latlon.iloc[idx]
            results.append(
                {
                    "rps_id": original_row["RPS_id"],
                    "coordinates": (float(original_row.geometry.x), float(original_row.geometry.y)),
                    "poi_counts_1km": counts_1km,
                    "poi_names_1km": names_1km,
                    "poi_counts_sensitive_0.1km": sensitive_counts,
                    "poi_names_sensitive_0.1km": sensitive_names_dict,
                }
            )

        return results

    def integrate_with_intermediate(
        self, intermediate: Dict[str, Dict[str, object]]
    ) -> Dict[str, Dict[str, object]]:
        """Merge buffer statistics into the perception-module intermediate dict."""
        enrichment = self.build_intermediate_records()
        for record in enrichment:
            rps_id = str(record["rps_id"])
            if rps_id not in intermediate:
                intermediate[rps_id] = {}
            intermediate[rps_id]["poi_counts_1km"] = record["poi_counts_1km"]
            intermediate[rps_id]["poi_names_1km"] = record["poi_names_1km"]
            intermediate[rps_id]["poi_counts_sensitive_0.1km"] = record[
                "poi_counts_sensitive_0.1km"
            ]
            intermediate[rps_id]["poi_names_sensitive_0.1km"] = record[
                "poi_names_sensitive_0.1km"
            ]
            intermediate[rps_id]["coordinates"] = record["coordinates"]
        return intermediate


@register_tool
class POIStatisticsAnalysisTool(BaseTool):
    """Registered tool: layered POI buffer counts per parking spot."""

    name = "poi_statistics_analysis"
    category = ToolCategory.MEMORY_CONSTRUCTION
    description = (
        "Layered buffer POI statistics for RPS shapefile and POI directory; "
        "returns intermediate record list"
    )

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        parking_path: str,
        poi_dir: str,
        **kwargs: Any,
    ) -> List[Dict[str, object]]:
        tool = POIStatisticsTool(Path(parking_path), Path(poi_dir))
        recs = tool.build_intermediate_records()
        if ctx is not None:
            ctx.set("poi_statistics_record_count", len(recs))
        return recs
