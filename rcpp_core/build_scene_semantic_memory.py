# -*- coding: utf-8 -*-
"""
Build and persist global scene-semantic memory (Neo4j), keyed by study area.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from configs.data_config import (
    ENV_NEO4J_DATABASE,
    ENV_NEO4J_PASSWORD,
    ENV_NEO4J_URI,
    ENV_NEO4J_USER,
)

logger = logging.getLogger(__name__)

_DRIVER: Any = None


def _require_neo4j_uri() -> str:
    uri = os.environ.get(ENV_NEO4J_URI, "").strip()
    if not uri:
        raise ValueError(
            "NEO4J_URI must be set in the environment for global scene_semantic_memory in Neo4j."
        )
    return uri


def _get_driver() -> Any:
    """
    Initializes and returns the Neo4j database driver.
    """
    global _DRIVER
    if _DRIVER is None:
        from neo4j import GraphDatabase

        uri = _require_neo4j_uri()
        user = os.environ.get(ENV_NEO4J_USER, "neo4j").strip() or "neo4j"
        password = os.environ.get(ENV_NEO4J_PASSWORD, "")
        _DRIVER = GraphDatabase.driver(uri, auth=(user, password))
    return _DRIVER


def _session_kwargs() -> Dict[str, Any]:
    """
    Constructs a dictionary with Neo4j session parameters, primarily specifying which database to use.
    """
    db = os.environ.get(ENV_NEO4J_DATABASE, "").strip()
    if db:
        return {"database": db}
    return {}


def _deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges two dictionaries; new values from the update dictionary will override items
    with the same name in the base, and sub-dictionaries are merged recursively.
    """
    result = dict(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def merge_scene_semantic_memory(prior: Dict[str, Any], fresh: Dict[str, Any]) -> Dict[str, Any]:
    """
    Incremental merge: same RPS id merges nested dicts deeply; new RPS ids are added.
    """
    out = dict(prior)
    for rps_id, inc in fresh.items():
        if rps_id in out and isinstance(out[rps_id], dict) and isinstance(inc, dict):
            out[rps_id] = _deep_merge_dict(out[rps_id], inc)
        else:
            out[rps_id] = inc
    return out


def resolve_scene_semantic_memory_json_path(
    raw: Union[str, Path],
    workspace_dir: Optional[str] = None,
) -> Path:
    """
    Resolve a path from global_instruction; relative paths are under workspace_dir or cwd.
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    base = Path(workspace_dir).expanduser().resolve() if workspace_dir else Path.cwd()
    return (base / p).resolve()


def load_scene_semantic_memory_json_file(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a scene_semantic_memory JSON object.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"scene_semantic_memory JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        out: Dict[str, Any] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("scene_semantic_memory JSON list entries must be objects")
            rid = item.get("RPS_id")
            if rid is None or str(rid).strip() == "":
                raise ValueError("scene_semantic_memory JSON list entry missing RPS_id")
            out[str(rid)] = item
        return out
    raise ValueError("scene_semantic_memory JSON root must be an object or array")


def list_all_global_scene_memory_area_slugs() -> List[str]:
    """
    Return sorted area_slug values that have a GlobalSceneSemanticMemory node in Neo4j.
    """
    _require_neo4j_uri()
    driver = _get_driver()
    with driver.session(**_session_kwargs()) as session:
        recs = session.run(
            """
            MATCH (n:GlobalSceneSemanticMemory)
            RETURN n.area_slug AS area
            ORDER BY area
            """
        )
        out: List[str] = []
        for r in recs:
            a = r.get("area")
            if a:
                out.append(str(a))
    return out


def load_global_scene_semantic_memory(area_slug: str) -> Dict[str, Any]:
    """
    Load the global scene-semantic memory for a given area from Neo4j and return as a dict.
    """
    _require_neo4j_uri()
    slug = str(area_slug).strip().lower()
    if not slug:
        return {}
    driver = _get_driver()
    with driver.session(**_session_kwargs()) as session:
        rec = session.run(
            """
            MATCH (n:GlobalSceneSemanticMemory {area_slug: $area})
            RETURN n.payload_json AS payload
            LIMIT 1
            """,
            area=slug,
        ).single()
    if not rec or not rec.get("payload"):
        return {}
    raw = rec["payload"]
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in GlobalSceneSemanticMemory for area=%s", slug)
            return {}
    return {}


def save_global_scene_semantic_memory(area_slug: str, memory: Dict[str, Any]) -> None:
    """
    Save provided scene-semantic memory as JSON for a given area in Neo4j.
    """
    _require_neo4j_uri()
    slug = str(area_slug).strip().lower()
    if not slug:
        raise ValueError("area_slug is required for save_global_scene_semantic_memory.")
    payload = json.dumps(memory, ensure_ascii=False, default=str)
    driver = _get_driver()
    with driver.session(**_session_kwargs()) as session:
        session.run(
            """
            MERGE (n:GlobalSceneSemanticMemory {area_slug: $area})
            SET n.payload_json = $payload, n.updated_at = datetime()
            """,
            area=slug,
            payload=payload,
        )
        _save_versioned_memory_nodes(session, slug, memory)
    logger.info("Neo4j global scene_semantic_memory saved for area=%s (%s RPS keys).", slug, len(memory))


def _version_row(
    area_slug: str,
    rps_id: str,
    kind: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    meta = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}
    version = str(meta.get("adapter_version") or "legacy-v1")
    return {
        "area": area_slug,
        "rps_id": rps_id,
        "version": version,
        "payload": json.dumps(payload, ensure_ascii=False, default=str),
        "base_model": str(meta.get("base_model") or "unknown"),
        "schema_version": str(meta.get("schema_version") or "1.0"),
        "generated_at": str(meta.get("generated_at") or ""),
        "quality_score": float(meta.get("quality_score") or 0.0),
        "evidence_source": str(meta.get("evidence_source") or "legacy"),
        "status": str(meta.get("status") or "active"),
        "expires_at": str(meta.get("expires_at") or ""),
    }


def _save_versioned_memory_nodes(session: Any, area_slug: str, memory: Dict[str, Any]) -> None:
    """Persist independently versioned SceneMemory and SemanticMemory nodes.

    The legacy aggregate node remains the BaseStore compatibility surface.  The
    normalized nodes make model versions queryable and independently rollable.
    """
    scene_rows: List[Dict[str, Any]] = []
    semantic_rows: List[Dict[str, Any]] = []
    rps_rows: List[Dict[str, Any]] = []
    for key, record in memory.items():
        if not isinstance(record, dict):
            continue
        rps_id = str(record.get("RPS_id") or key)
        coords = record.get("coordinates") or []
        rps_rows.append(
            {
                "area": area_slug,
                "rps_id": rps_id,
                "longitude": coords[0] if len(coords) > 0 else None,
                "latitude": coords[1] if len(coords) > 1 else None,
                "street_name": str(record.get("street_name") or ""),
            }
        )
        node = record.get("memory_node") or {}
        scene = node.get("scene_memory")
        semantic = node.get("semantic_memory")
        if isinstance(scene, dict):
            scene_rows.append(_version_row(area_slug, rps_id, "scene", scene))
        if isinstance(semantic, dict):
            semantic_rows.append(_version_row(area_slug, rps_id, "semantic", semantic))

    if rps_rows:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (r:RPS {area_slug: row.area, rps_id: row.rps_id})
            SET r.longitude = row.longitude,
                r.latitude = row.latitude,
                r.street_name = row.street_name,
                r.updated_at = datetime()
            """,
            rows=rps_rows,
        )
    for label, relation, rows in (
        ("SceneMemory", "HAS_SCENE_MEMORY", scene_rows),
        ("SemanticMemory", "HAS_SEMANTIC_MEMORY", semantic_rows),
    ):
        if not rows:
            continue
        # Labels and relationship names are fixed constants, never user input.
        session.run(
            f"""
            UNWIND $rows AS row
            MATCH (r:RPS {{area_slug: row.area, rps_id: row.rps_id}})
            OPTIONAL MATCH (r)-[:{relation}]->(old:{label})
            WHERE old.is_current = true
            SET old.is_current = false, old.status = 'superseded'
            WITH r, row, collect(old) AS previous
            MERGE (m:{label} {{area_slug: row.area, rps_id: row.rps_id, version: row.version}})
            SET m.payload_json = row.payload,
                m.base_model = row.base_model,
                m.schema_version = row.schema_version,
                m.generated_at = row.generated_at,
                m.quality_score = row.quality_score,
                m.evidence_source = row.evidence_source,
                m.status = row.status,
                m.expires_at = row.expires_at,
                m.is_current = true,
                m.updated_at = datetime()
            MERGE (r)-[:{relation}]->(m)
            WITH m, [old IN previous WHERE old IS NOT NULL AND old <> m] AS replaced
            FOREACH (old IN replaced | MERGE (m)-[:SUPERSEDES]->(old))
            """,
            rows=rows,
        )
