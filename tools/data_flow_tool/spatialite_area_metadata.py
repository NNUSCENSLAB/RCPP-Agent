# -*- coding: utf-8 -*-
"""
SpatiaLite area metadata: during runtime, merge RPS shp directory on disk; other vectors are already in the database, read through layers.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

METADATA_TABLE = "rcpp_agent_area_metadata"

DDL = f"""
CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
  area TEXT NOT NULL PRIMARY KEY,
  rps_shp_path TEXT NOT NULL,
  district_data_root TEXT
);
"""


def ensure_area_metadata_table(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    cur = conn.execute(f"PRAGMA table_info({METADATA_TABLE})")
    col_names = [row[1] for row in cur.fetchall()]
    if col_names and "district_data_root" not in col_names:
        conn.execute(f"ALTER TABLE {METADATA_TABLE} ADD COLUMN district_data_root TEXT")
    conn.commit()


def upsert_area_metadata(
    db_path: Path,
    area: str,
    rps_shp_path: str,
    district_data_root: Optional[str] = None,
) -> None:
    """
    Insert or update area metadata.
    """
    p = Path(db_path).resolve()
    conn = sqlite3.connect(str(p))
    try:
        ensure_area_metadata_table(conn)
        conn.execute(
            f"INSERT OR REPLACE INTO {METADATA_TABLE} (area, rps_shp_path, district_data_root) VALUES (?, ?, ?)",
            (area, rps_shp_path, district_data_root),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_area_metadata(db_path: Path, area: str) -> Optional[Dict[str, Any]]:
    """
    Fetch area metadata.
    """
    p = Path(db_path).resolve()
    if not p.exists():
        return None
    conn = sqlite3.connect(str(p))
    try:
        cur = conn.execute(
            f"SELECT rps_shp_path, district_data_root FROM {METADATA_TABLE} WHERE area = ?",
            (area,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "rps_shp_path": str(row[0]),
            "district_data_root": str(row[1]) if row[1] else None,
        }
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def fetch_rps_shp_path(db_path: Path, area: str) -> Optional[str]:
    meta = fetch_area_metadata(db_path, area)
    return meta["rps_shp_path"] if meta else None
