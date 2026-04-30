"""
Environment Perception Agent: extracts environmental and semantic features around each RPS
and builds merged scene-semantic memory persisted through the long-term store.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from langgraph.store.base import BaseStore

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import (
    PERCEPTION_SCENE_ADAPTER_PATH,
    PERCEPTION_SCENE_BASE_MODEL,
    PERCEPTION_SEMANTIC_ADAPTER_PATH,
    PERCEPTION_SEMANTIC_BASE_MODEL,
)
from configs.data_config import (
    DEFAULT_WORKSPACE,
    path_default_scene_semantic_memory_json,
    path_output_environment_perception_agent,
)
from rcpp_core.build_scene_semantic_memory import (
    load_scene_semantic_memory_json_file,
    merge_scene_semantic_memory,
    resolve_scene_semantic_memory_json_path,
)
from rcpp_core.neo4j_scene_semantic_store import (
    SCENE_SEMANTIC_KEY_GLOBAL,
    area_slug_from_spatialite_context,
    scene_semantic_namespace,
)
from tools.memory_construction_tool.perception_pipeline import PerceptionPipeline

logger = logging.getLogger(__name__)


def _find_scene_semantic_memory_json(data_root: Path) -> Optional[Path]:
    """
    First ``*.json`` directly under *data_root* whose filename contains scene, semantic, and memory
    (case-insensitive; no fixed naming pattern).
    """
    if not data_root.is_dir():
        return None

    def matches(p: Path) -> bool:
        if not p.is_file() or p.suffix.lower() != ".json":
            return False
        n = p.name.lower()
        return "scene" in n and "semantic" in n and "memory" in n

    found = sorted(p for p in data_root.iterdir() if matches(p))
    return found[0] if found else None


class EnvironmentPerceptionAgent:
    def __init__(
        self,
        workspace_dir: str = DEFAULT_WORKSPACE,
        scene_base_model: str = PERCEPTION_SCENE_BASE_MODEL,
        scene_adapter_path: str = PERCEPTION_SCENE_ADAPTER_PATH,
        semantic_base_model: str = PERCEPTION_SEMANTIC_BASE_MODEL,
        semantic_adapter_path: str = PERCEPTION_SEMANTIC_ADAPTER_PATH,
    ):
        self._pipeline = PerceptionPipeline(
            workspace_dir=workspace_dir,
            scene_base_model=scene_base_model,
            scene_adapter_path=scene_adapter_path,
            semantic_base_model=semantic_base_model,
            semantic_adapter_path=semantic_adapter_path,
        )

    def process_from_spatialite(self, spatialite_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the Environment Perception Agent pipeline: 
        Input multi-source spatial data -> construct scene memory -> construct semantic memory ->
        merge scene and semantic memory -> output scene-semantic memory.
        """
        print("Environment Perception Agent running...", flush=True)
        db_path = spatialite_context.get("db_path")
        area = spatialite_context.get("area")
        logger.info(
            "Environment Perception Agent running. db_path=%s area=%s scenario=%s time_horizon=%s",
            db_path,
            area,
            spatialite_context.get("scenario"),
            spatialite_context.get("time_horizon"),
        )

        if not db_path or not os.path.exists(db_path):
            raise FileNotFoundError(f"SpatiaLite database file not found: {db_path}")

        result = self._pipeline.run(spatialite_context)
        n = len(result) if isinstance(result, dict) else 0
        print(
            f"[Environment Perception] Pipeline finished. Loaded {n} scene-semantic memory.",
            flush=True,
        )
        return result

    def perceive(
        self,
        spatialite_context: Dict[str, Any],
        global_instruction: Dict[str, Any],
        *,
        store: BaseStore,
    ) -> Dict[str, Any]:
        """
        Graph node entry: load scene-semantic memory.
        """
        if not spatialite_context:
            raise ValueError("spatialite_context is required for Environment Perception.")

        area_slug = area_slug_from_spatialite_context(spatialite_context)
        ns = scene_semantic_namespace(area_slug)
        gi = global_instruction or {}
        raw_json = gi.get("scene_semantic_memory_json_path") or (
            spatialite_context or {}
        ).get("scene_semantic_memory_json_path")
        workspace_for_resolve = gi.get("workspace_dir") or spatialite_context.get("workspace_dir")

        prior_item = store.get(ns, SCENE_SEMANTIC_KEY_GLOBAL)
        prior: Dict[str, Any] = (
            dict(prior_item.value)
            if prior_item is not None and isinstance(prior_item.value, dict)
            else {}
        )
        if prior:
            logger.info(
                "Loaded global scene-semantic memory from store for area=%s (%s RPS keys).",
                area_slug,
                len(prior),
            )

        json_path: Optional[Path] = None
        if raw_json:
            json_path = resolve_scene_semantic_memory_json_path(
                str(raw_json),
                workspace_for_resolve,
            )
        else:
            adr = spatialite_context.get("area_data_root") or spatialite_context.get(
                "district_data_root"
            )
            if adr:
                json_path = _find_scene_semantic_memory_json(Path(str(adr)))
            if json_path is None:
                out_default = path_default_scene_semantic_memory_json(
                    area_slug, workspace_for_resolve
                )
                if out_default.is_file():
                    json_path = out_default

        if json_path is not None:
            logger.info(
                "Using scene_semantic_memory JSON: %s",
                json_path,
            )
            fresh = load_scene_semantic_memory_json_file(json_path)
        else:
            fresh = self.process_from_spatialite(spatialite_context)

        scene_semantic_memory = merge_scene_semantic_memory(prior, fresh)
        store.put(ns, SCENE_SEMANTIC_KEY_GLOBAL, dict(scene_semantic_memory))
        logger.info(
            "Saved global scene_semantic_memory for area=%s (%s RPS keys).",
            area_slug,
            len(scene_semantic_memory),
        )

        ep_dir = path_output_environment_perception_agent()
        ep_dir.mkdir(parents=True, exist_ok=True)
        ep_json = ep_dir / f"scene_semantic_memory_{area_slug}.json"
        try:
            with ep_json.open("w", encoding="utf-8") as f:
                json.dump(dict(scene_semantic_memory), f, ensure_ascii=False, indent=2)
            logger.info("Wrote scene_semantic_memory JSON to %s", ep_json)
        except OSError as exc:
            logger.warning("Could not write scene_semantic_memory JSON (%s): %s", ep_json, exc)

        return {"workflow_status": "perception_completed"}
