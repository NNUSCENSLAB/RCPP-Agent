"""Persistent fake domain repository for failure-injection tests; no network/GPU."""
import json
from contextlib import closing
import os
from pathlib import Path
import sqlite3

from rcpp_core.graph_memory_repository import GraphMemoryRepository


def scene(**overrides):
    value = {"has_existing_RCP": False, "is_functional_zone": False,
             "functional_zone_type": "none", "has_ground_obstacle": False,
             "ground_obstacle_types": "none", "ground_obstacle_count": 0,
             "visual_distractors_noted": ["none"], "clearance_visual_assessment": True,
             "scene_reasoning": "Photo shows a clear lateral strip", "confidence_score": 5}
    return {**value, **overrides}


def semantic(rid):
    return {"rps_id": rid, "functional_zone_type": "mixed", "commuting_flow": "low",
            "sensitive_constraints": "none", "grid_accessibility": "good"}


class FakeRepository(GraphMemoryRepository):
    def __init__(self, path: Path, *, crash=False):
        self.path = path
        self.crash = crash
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS memories(kind TEXT,rid TEXT,payload TEXT,PRIMARY KEY(kind,rid))")
            conn.execute("CREATE TABLE IF NOT EXISTS calls(kind TEXT,rid TEXT,n INTEGER,PRIMARY KEY(kind,rid))")
            conn.execute("CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY,payload TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS queries(kind TEXT,ids TEXT)")

    def list_rps(self, area):
        return [{"rps_id": rid, "longitude": 1, "latitude": 2,
                 "source_images": [f"{rid}.jpg"], "street_name": "test road"} for rid in ("A", "B", "C")]

    def _current(self, kind, ids):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("INSERT INTO queries VALUES(?,?)", (kind, json.dumps(ids)))
            rows = conn.execute("SELECT rid,payload FROM memories WHERE kind=?", (kind,)).fetchall()
        return {rid: json.loads(payload) for rid, payload in rows if ids is None or rid in ids}

    def get_current_scene(self, area, rps_ids=None):
        return self._current("scene", rps_ids)

    def get_current_semantic(self, area, rps_ids=None):
        return self._current("semantic", rps_ids)

    def _append(self, kind, rid, memory, metadata):
        if self.crash and kind == "scene" and rid == "B":
            os._exit(73)  # result receipt is committed, Neo4j-equivalent write is not
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO memories VALUES(?,?,?)",
                         (kind, rid, json.dumps({**memory, "_meta": {"status": "active", **metadata}})))
        return metadata.get("memory_id", "fake")

    def append_scene_memory(self, area, rps_id, memory, **metadata):
        return self._append("scene", rps_id, memory, metadata)

    def append_semantic_memory(self, area, rps_id, memory, **metadata):
        return self._append("semantic", rps_id, memory, metadata)

    def append_decisions(self, run_id, decisions):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            for row in decisions:
                conn.execute("INSERT OR REPLACE INTO decisions VALUES(?,?)",
                             (row.get("decision_id", row["rps_id"] + row["outcome"]), json.dumps(row)))

    def worker(self, kind):
        def call(area, ids, context):
            with closing(sqlite3.connect(self.path)) as conn, conn:
                for rid in ids:
                    conn.execute("""INSERT INTO calls VALUES(?,?,1) ON CONFLICT(kind,rid)
                        DO UPDATE SET n=n+1""", (kind, rid))
            return {rid: scene() if kind == "scene" else semantic(rid) for rid in ids}
        return call
