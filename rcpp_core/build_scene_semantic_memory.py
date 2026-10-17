# -*- coding: utf-8 -*-
"""Neo4j connection and legacy JSON parsing helpers.

Durable memory operations belong to :mod:`rcpp_core.graph_memory_repository`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

from configs.data_config import ENV_NEO4J_DATABASE, ENV_NEO4J_PASSWORD, ENV_NEO4J_URI, ENV_NEO4J_USER

_DRIVER: Any = None


def _require_neo4j_uri() -> str:
    uri = os.environ.get(ENV_NEO4J_URI, "").strip()
    if not uri:
        raise ValueError("NEO4J_URI must be set for Scene–Semantic graph memory")
    return uri


def _get_driver() -> Any:
    global _DRIVER
    if _DRIVER is None:
        from neo4j import GraphDatabase
        user = os.environ.get(ENV_NEO4J_USER, "neo4j").strip() or "neo4j"
        password = os.environ.get(ENV_NEO4J_PASSWORD, "")
        _DRIVER = GraphDatabase.driver(_require_neo4j_uri(), auth=(user, password))
    return _DRIVER


def _session_kwargs() -> Dict[str, Any]:
    database = os.environ.get(ENV_NEO4J_DATABASE, "").strip()
    return {"database": database} if database else {}


def load_scene_semantic_memory_json_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Parse a legacy export for validation or explicit migration only."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Scene/Semantic JSON not found: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        output: Dict[str, Any] = {}
        for item in data:
            if not isinstance(item, dict) or not str(item.get("RPS_id") or "").strip():
                raise ValueError("Legacy JSON array entries must contain RPS_id")
            output[str(item["RPS_id"])] = item
        return output
    raise ValueError("Legacy Scene/Semantic JSON root must be an object or array")
