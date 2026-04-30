# -*- coding: utf-8 -*-
"""POI and grid power point loader for review metrics."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from configs.data_config import POI_FILENAME_MAPPING

logger = logging.getLogger(__name__)


def _category_from_csv_stem(stem: str) -> Optional[str]:
    """
    Map POI data to categories.
    """
    stem_lower = stem.lower()
    for key in sorted(POI_FILENAME_MAPPING.keys(), key=len, reverse=True):
        if stem_lower == key or stem_lower.endswith("_" + key):
            return POI_FILENAME_MAPPING[key]
    return None


class ReviewDataLoader:
    def __init__(self, poi_dir: Optional[Path] = None):
        self.poi_dir = Path(poi_dir) if poi_dir else Path("data/POI")

    def load_poi_data(self) -> gpd.GeoDataFrame:
        logger.info("Loading POI from %s", self.poi_dir)
        records: List[pd.DataFrame] = []
        for csv_path in sorted(self.poi_dir.glob("*.csv")):
            category_label = _category_from_csv_stem(csv_path.stem)
            if not category_label:
                continue
            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
            piece = pd.DataFrame(
                {
                    "name": df["name"].astype(str),
                    "category": category_label,
                    "geometry": [
                        Point(lon, lat) for lon, lat in zip(df["lng"], df["lat"])
                    ],
                }
            )
            records.append(piece)
        if not records:
            raise ValueError(f"No valid POI CSV under {self.poi_dir}")
        poi_df = pd.concat(records, ignore_index=True)
        poi_gdf = gpd.GeoDataFrame(poi_df, geometry="geometry", crs="EPSG:4326")
        logger.info("Loaded %s POIs", len(poi_gdf))
        return poi_gdf

    def load_grid_points(self, poi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Returns the rows with the category label "grid" for use in NLE metric calculation.
        """
        if poi_gdf is None or len(poi_gdf) == 0 or "category" not in poi_gdf.columns:
            logger.warning("Grid subset: empty POI or missing category column")
            return gpd.GeoDataFrame()
        mask = poi_gdf["category"] == "grid"
        if not bool(mask.any()):
            logger.warning("No grid_power rows (category=grid) in POI under %s", self.poi_dir)
            return gpd.GeoDataFrame()
        grid = gpd.GeoDataFrame(
            poi_gdf.loc[mask],
            geometry="geometry",
            crs=poi_gdf.crs,
        )
        logger.debug("Grid subset: %s points from merged POI", len(grid))
        return grid
