# -*- coding: utf-8 -*-
"""
Import the raw data directory of the study area into SpatiaLite.
Before using, put the data under your raw-data root (default: RCPPAGENT_RAW_DATA_BASE or {workspace}/data/China_jiangsu_nanjing_gulou).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any, Literal, Optional, cast

import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from configs.data_config import POI_FILENAME_MAPPING

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.data_flow_tool.spatialite_area_metadata import upsert_area_metadata
from tools.rcpp_paths import default_district_import_base, default_spatialite_db_path
from tools.registry import register_tool

_GEO_IO_ENGINE = "pyogrio"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_bsv_image_rows(image_dir: Path) -> list[tuple[str, str]]:
    """
    Scan BSV image directory; return (image_name, image_path) with POSIX paths.
    """
    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Street view image directory not found: {image_dir}")

    rows: list[tuple[str, str]] = []
    for p in sorted(image_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        name = p.name
        path_str = str(p).replace("\\", "/")
        rows.append((name, path_str))

    if not rows:
        raise ValueError(f"No image files found in directory: {image_dir}")
    return rows


def populate_bsv_image_table(db_path: Path, rows: list[tuple[str, str]]) -> int:
    """Replace bsv_image with a plain attribute table (image_name, image_path)."""
    if not rows:
        raise ValueError("bsv_image: no rows to insert")

    db_path = db_path.resolve()
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute('DROP TABLE IF EXISTS "bsv_image"')
        cur.execute('CREATE TABLE "bsv_image" (image_name TEXT, image_path TEXT)')
        cur.executemany(
            'INSERT INTO "bsv_image" (image_name, image_path) VALUES (?, ?)',
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def _promote_to_multi(geom: BaseGeometry | None) -> BaseGeometry | None:
    """Match ogr2ogr -nlt PROMOTE_TO_MULTI for polygon / line / point."""
    if geom is None or geom.is_empty:
        return geom
    t = geom.geom_type
    if t == "Polygon":
        return MultiPolygon([cast(Polygon, geom)])
    if t == "LineString":
        return MultiLineString([cast(LineString, geom)])
    if t == "Point":
        return MultiPoint([cast(Point, geom)])
    return geom


def _concat_shapefiles(paths: list[Path]) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for p in paths:
        g = gpd.read_file(p, engine=_GEO_IO_ENGINE)
        if g.crs is None:
            g = g.set_crs("EPSG:4326")
        g = g.to_crs("EPSG:4326")
        g = g.copy()
        g["geometry"] = g.geometry.map(_promote_to_multi)
        frames.append(g)
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    return merged


def _write_spatialite_layer(
    gdf: gpd.GeoDataFrame,
    db_path: Path,
    layer: str,
    *,
    mode: Literal["w", "a"],
) -> None:
    print(
        f"+ geopandas to_file SQLite (SpatiaLite, {_GEO_IO_ENGINE}) "
        f"layer={layer!r} mode={mode} -> {db_path}"
    )
    # pyogrio：追加图层稳定；Fiona 对 SpatiaLite mode='a' 易出现 NULL pointer。
    # 勿传 crs=：GeoPandas 对 pyogrio 传 crs 会报错，CRS 以 gdf.crs 为准。
    gdf.to_file(
        db_path,
        driver="SQLite",
        layer=layer,
        mode=mode,
        spatialite=True,
        encoding="utf-8",
        engine=_GEO_IO_ENGINE,
    )


def _list_roads_shps(base: Path, roads_shp: str | Path | None) -> list[Path]:
    if roads_shp is not None:
        p = Path(roads_shp)
        p = p if p.is_absolute() else (base / p)
        return [p]
    return sorted((base / "roads").rglob("*.shp"))


def merge_poi_csvs(poi_dir: Path, out_csv: Path) -> int:
    poi_dir = poi_dir.resolve()
    if not poi_dir.is_dir():
        raise FileNotFoundError(f"POI directory not found: {poi_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(poi_dir.rglob("*.csv")):
        stem = csv_path.stem
        key_part = stem.rsplit("_", 1)[-1].lower() if "_" in stem else stem.lower()
        category = POI_FILENAME_MAPPING.get(key_part, key_part)
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        if "lng" not in df.columns or "lat" not in df.columns:
            raise ValueError(f"POI CSV missing lng/lat: {csv_path}")
        if "name" not in df.columns:
            df["name"] = ""
        df = df.assign(category=category)
        frames.append(cast(pd.DataFrame, df[["name", "category", "lng", "lat"]]))

    if not frames:
        raise ValueError(f"No POI CSV found under {poi_dir}")

    merged = pd.concat(frames, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    return len(merged)


def import_all(
    base: Path,
    db_path: Path,
    overwrite: bool,
    *,
    area: str,
    rps_subdir: str = "RPS",
    poi_subdir: str = "POI",
    od_subdir: str = "OD",
    bsv_subdir: str = "BSV",
    roads_shp: str | Path | None = None,
) -> None:
    base = base.resolve()
    db_path = db_path.resolve()
    rps_dir = base / rps_subdir
    poi_dir = base / poi_subdir
    od_dir = base / od_subdir
    bsv_dir = base / bsv_subdir
    road_shps = _list_roads_shps(base, roads_shp)

    temp_dir = base / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    poi_merged = temp_dir / "poi_merged.csv"

    if db_path.exists() and not overwrite:
        raise FileExistsError(f"Database already exists: {db_path}, enable overwrite to rebuild")

    rps_shapes = sorted(rps_dir.rglob("*.shp"))
    if not rps_shapes:
        raise FileNotFoundError(f"No RPS shp found under {rps_dir}")

    region_shps = sorted(od_dir.rglob("regions.shp"))
    flow_shps = sorted(od_dir.rglob("od_flows.shp"))
    if not region_shps:
        raise FileNotFoundError(f"No regions.shp under {od_dir}")
    if not flow_shps:
        raise FileNotFoundError(f"No od_flows.shp under {od_dir}")

    if not road_shps:
        raise FileNotFoundError(f"No *.shp under {base / 'roads'}")
    for p in road_shps:
        if not p.exists():
            raise FileNotFoundError(p)

    merge_poi_csvs(poi_dir, poi_merged)

    bsv_rows = collect_bsv_image_rows(bsv_dir)
    print(f"Collected {len(bsv_rows)} BSV image path(s) from {bsv_dir}")

    if overwrite and db_path.exists():
        db_path.unlink()

    rps_gdf = _concat_shapefiles(rps_shapes)
    _write_spatialite_layer(rps_gdf, db_path, "rps", mode="w")

    od_region_gdf = _concat_shapefiles(region_shps)
    _write_spatialite_layer(od_region_gdf, db_path, "od_region", mode="a")

    od_flow_gdf = _concat_shapefiles(flow_shps)
    _write_spatialite_layer(od_flow_gdf, db_path, "od_flow", mode="a")

    roads_gdf = _concat_shapefiles(road_shps)
    _write_spatialite_layer(roads_gdf, db_path, "roads", mode="a")

    poi_df = pd.read_csv(poi_merged, encoding="utf-8")
    poi_gdf = gpd.GeoDataFrame(
        poi_df,
        geometry=gpd.points_from_xy(poi_df["lng"], poi_df["lat"]),
        crs="EPSG:4326",
    )
    poi_gdf["geometry"] = poi_gdf.geometry.map(_promote_to_multi)
    _write_spatialite_layer(poi_gdf, db_path, "poi", mode="a")

    n_bsv = populate_bsv_image_table(db_path, bsv_rows)
    print(f"Inserted {n_bsv} row(s) into bsv_image")

    upsert_area_metadata(
        db_path,
        area,
        str(rps_dir.resolve()),
        district_data_root=str(base.resolve()),
    )
    print(f"Completed: {db_path} (area metadata written: {area!r})")


@register_tool
class ImportSpatialiteTool(BaseTool):
    name = "import_spatialite"
    category = ToolCategory.DATA_FLOW

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        area: str,
        base: Optional[Path] = None,
        db: Optional[Path] = None,
        overwrite: bool = True,
        rps_subdir: str = "RPS",
        poi_subdir: str = "POI",
        od_subdir: str = "OD",
        bsv_subdir: str = "BSV",
        roads_shp: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        b = base if base is not None else default_district_import_base()
        d = db if db is not None else default_spatialite_db_path()
        import_all(
            b,
            d,
            overwrite,
            area=area,
            rps_subdir=rps_subdir,
            poi_subdir=poi_subdir,
            od_subdir=od_subdir,
            bsv_subdir=bsv_subdir,
            roads_shp=roads_shp,
        )
        if ctx is not None:
            ctx.set("db_path", str(d.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import SpatiaLite")
    parser.add_argument(
        "--base",
        type=Path,
        default= None,
        help=f"Raw data root directory, default {default_district_import_base()} (please replace with your local district data path)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"SpatiaLite path, default {default_spatialite_db_path()}",
    )
    parser.add_argument("--rps_subdir", type=str, default="RPS")
    parser.add_argument("--poi_subdir", type=str, default="POI")
    parser.add_argument("--od_subdir", type=str, default="OD")
    parser.add_argument("--bsv_subdir", type=str, default="BSV")
    parser.add_argument("--roads_shp", type=str, default=None)
    parser.add_argument("--area", type=str, default='gulou')
    args = parser.parse_args()
    base = args.base if args.base is not None else default_district_import_base()
    db = args.db if args.db is not None else default_spatialite_db_path()
    import_all(
        base,
        db,
        True,
        area=args.area,
        rps_subdir=args.rps_subdir,
        poi_subdir=args.poi_subdir,
        od_subdir=args.od_subdir,
        bsv_subdir=args.bsv_subdir,
        roads_shp=args.roads_shp,
    )


if __name__ == "__main__":
    main()
