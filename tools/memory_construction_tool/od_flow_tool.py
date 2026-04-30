# -*- coding: utf-8 -*-
"""
Commuting OD flow calculation
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import geopandas as gpd
import pandas as pd

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.registry import register_tool


def load_od_flow_data(regions_path: Path, od_flows_path: Path) -> pd.DataFrame:
    grids_gdf = gpd.read_file(regions_path)
    grids_gdf = grids_gdf.to_crs(epsg=4326)
    od_gdf = gpd.read_file(od_flows_path)
    od_df = pd.DataFrame(od_gdf)

    outflow = od_df.groupby("origin_loc")["flow"].sum().reset_index()
    outflow.columns = ["locations", "outflow"]

    inflow = od_df.groupby("dest_loc")["flow"].sum().reset_index()
    inflow.columns = ["locations", "inflow"]

    flow_summary = pd.merge(grids_gdf[["locations", "geometry"]], inflow, on="locations", how="left")
    flow_summary = pd.merge(flow_summary, outflow, on="locations", how="left")

    flow_summary["inflow"] = flow_summary["inflow"].fillna(0)
    flow_summary["outflow"] = flow_summary["outflow"].fillna(0)

    return flow_summary


def load_od_flow_data_from_gdfs(grids_gdf: gpd.GeoDataFrame, od_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    grids_gdf = grids_gdf.to_crs(epsg=4326)
    od_df = pd.DataFrame(od_gdf)

    outflow = od_df.groupby("origin_loc")["flow"].sum().reset_index()
    outflow.columns = ["locations", "outflow"]

    inflow = od_df.groupby("dest_loc")["flow"].sum().reset_index()
    inflow.columns = ["locations", "inflow"]

    flow_summary = pd.merge(grids_gdf[["locations", "geometry"]], inflow, on="locations", how="left")
    flow_summary = pd.merge(flow_summary, outflow, on="locations", how="left")

    flow_summary["inflow"] = flow_summary["inflow"].fillna(0)
    flow_summary["outflow"] = flow_summary["outflow"].fillna(0)

    return flow_summary


@register_tool
class ODFlowSummaryTool(BaseTool):
    name = "od_flow_summary"
    category = ToolCategory.MEMORY_CONSTRUCTION

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        regions_path: str,
        od_flows_path: str,
        **kwargs: Any,
    ) -> List[dict]:
        df = load_od_flow_data(Path(regions_path), Path(od_flows_path))
        tab = df.drop(columns=["geometry"], errors="ignore")
        recs = tab.to_dict(orient="records")
        if ctx is not None:
            ctx.set("od_flow_summary_rows", len(recs))
        return recs
