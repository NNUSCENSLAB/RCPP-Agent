# -*- coding: utf-8 -*-
"""
Verify that the required tables exist in the SpatiaLite database and have more than 0 rows.
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Optional

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.rcpp_paths import default_spatialite_db_path
from tools.registry import register_tool

EXPECTED_TABLES = ("rps", "bsv_image", "poi", "od_region", "od_flow", "roads")


def verify(db_path: Path, use_ogrinfo: bool) -> None:
    db_path = db_path.resolve()
    if not db_path.exists():
        raise FileNotFoundError(str(db_path))

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    for t in EXPECTED_TABLES:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?",
            (t,),
        )
        if not cur.fetchone():
            raise RuntimeError(f"Missing table or view: {t}")
       
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        n = cur.fetchone()[0]
        if n == 0:
            raise RuntimeError(f"{t} has no rows")
        print(f"{t}: {n} rows")
    conn.close()

    if use_ogrinfo:
        r = subprocess.run(
            ["ogrinfo", "-so", str(db_path)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ogrinfo failed: {r.stderr}")
        print(r.stdout)


@register_tool
class VerifySpatialiteTool(BaseTool):
    name = "verify_spatialite"
    category = ToolCategory.DATA_FLOW

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        db_path: Optional[Path] = None,
        use_ogrinfo: bool = False,
        **kwargs: Any,
    ) -> None:
        db = db_path if db_path is not None else default_spatialite_db_path()
        verify(Path(db), use_ogrinfo)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db",
        type=Path,
        default=default_spatialite_db_path(),
    )
    p.add_argument("--ogrinfo", action="store_true")
    args = p.parse_args()
    verify(args.db, args.ogrinfo)


if __name__ == "__main__":
    main()
