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
    logger.info("Neo4j global scene_semantic_memory saved for area=%s (%s RPS keys).", slug, len(memory))
