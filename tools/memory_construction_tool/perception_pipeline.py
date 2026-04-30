# -*- coding: utf-8 -*-
"""
Environment Perception Agent pipeline
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from tools.data_flow_tool.spatialite_area_metadata import fetch_rps_shp_path
from tools.memory_construction_tool.memory_inference_tool import MemoryInferenceTool
from tools.memory_construction_tool.merge_memory_tool import run_merge
from tools.memory_construction_tool.scene_inference_input_tool import (
    generate_scene_inference_from_spatialite,
)
from tools.memory_construction_tool.semantic_inference_input_tool import (
    generate_semantic_inference_from_spatialite,
)
from tools.base import BaseTool, ToolCategory, ToolContext
from tools.rcpp_paths import perception_artifact_dir
from tools.registry import register_tool

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """Pipeline orchestrated by the Environment Perception Agent; intermediate file paths are determined by workspace + area + scenario + time_horizon."""

    def __init__(
        self,
        workspace_dir: str,
        scene_base_model: Optional[str] = None,
        scene_adapter_path: Optional[str] = None,
        semantic_base_model: Optional[str] = None,
        semantic_adapter_path: Optional[str] = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.scene_base_model = scene_base_model
        self.scene_adapter_path = scene_adapter_path
        self.semantic_base_model = semantic_base_model
        self.semantic_adapter_path = semantic_adapter_path
        self._memory_infer = MemoryInferenceTool()

    def run(self, spatialite_context: Dict[str, Any]) -> Dict[str, Any]:
        db_path = spatialite_context.get("db_path")
        area = spatialite_context.get("area")
        if not area:
            raise ValueError(
                "spatialite_context missing area: should be parsed from user instruction by Task Orchestration (e.g. Gulou -> gulou)"
            )
        scenario = spatialite_context.get("scenario", "balance_oriented")
        time_horizon = spatialite_context.get("time_horizon")
        tables = spatialite_context.get("tables") or {}

        if not db_path:
            raise ValueError("spatialite_context is missing db_path")

        if spatialite_context.get("workspace_dir"):
            workspace = Path(str(spatialite_context["workspace_dir"])).resolve()
        else:
            workspace = self.workspace_dir.resolve()

        artifact_dir = perception_artifact_dir(
            workspace,
            area=area,
            scenario=scenario,
            time_horizon=time_horizon,
        )
        scene_inference_input_path = artifact_dir / "scene_inference_input.json"
        scene_inference_output_path = artifact_dir / "scene_inference_output.json"
        semantic_inference_input_path = artifact_dir / "semantic_inference_input.json"
        semantic_inference_output_path = artifact_dir / "semantic_inference_output.json"
        merged_memory_output_path = artifact_dir / "scene_semantic_memory.json"
        mismatch_report_path = artifact_dir / "mismatch_report.txt"

        db = Path(db_path)
        explicit_shp = spatialite_context.get("rps_shp_path")
        shp_path = explicit_shp or fetch_rps_shp_path(db, str(area))
        if not shp_path:
            raise ValueError(
                f"RPS shp path not found for area {area!r}: provide rps_shp_path in spatialite_context, or use import_spatialite with consistent area to write to rcp_agent_area_metadata table"
            )

        logger.info("Toolchain: generate scene memory input -> %s", scene_inference_input_path)
        print(
            "[Perception] Step 1/5: Building scene memory input from SpatiaLite...",
            flush=True,
        )
        generate_scene_inference_from_spatialite(
            Path(db_path),
            scene_inference_input_path,
            str(area),
            tables,
        )
        print(
            f"[Perception] Step 1/5: Scene memoryinput ready -> {scene_inference_input_path}",
            flush=True,
        )

        logger.info("Toolchain: scene model inference (Qwen2.5-VL-7B + LoRA)")
        print(
            "[Perception] Step 2/5: Running fine-tuned scene model inference (this may take a while)...",
            flush=True,
        )
        self._memory_infer.run(
            kind="scene",
            input_file=str(scene_inference_input_path),
            output_file=str(scene_inference_output_path),
            scene_base_model=self.scene_base_model,
            scene_adapter_path=self.scene_adapter_path,
        )
        print(
            f"[Perception] Step 2/5: Scene inference finished -> {scene_inference_output_path}",
            flush=True,
        )

        logger.info("Toolchain: generate semantic memory input")
        print(
            "[Perception] Step 3/5: Building semantic memory input from SpatiaLite...",
            flush=True,
        )
        generate_semantic_inference_from_spatialite(
            Path(db_path),
            semantic_inference_input_path,
            str(area),
            tables,
        )
        print(
            f"[Perception] Step 3/5: Semantic memoryinput ready -> {semantic_inference_input_path}",
            flush=True,
        )

        logger.info("Toolchain: semantic model inference (Qwen2.5-7B + LoRA)")
        print(
            "[Perception] Step 4/5: Running fine-tuned semantic model inference (this may take a while)...",
            flush=True,
        )
        self._memory_infer.run(
            kind="semantic",
            input_file=str(semantic_inference_input_path),
            output_file=str(semantic_inference_output_path),
            semantic_base_model=self.semantic_base_model,
            semantic_adapter_path=self.semantic_adapter_path,
        )
        print(
            f"[Perception] Step 4/5: Semantic inference finished -> {semantic_inference_output_path}",
            flush=True,
        )

        logger.info("Toolchain: merge scene-semantic memory (rps_shp_path=%s)", shp_path)
        print("[Perception] Step 5/5: Merging scene and semantic memory...", flush=True)
        run_merge(
            str(scene_inference_output_path),
            str(semantic_inference_output_path),
            shp_path,
            str(merged_memory_output_path),
            mismatch_report_path=str(mismatch_report_path),
            area_slug=str(area),
        )
        print(
            f"[Perception] Step 5/5: Merge complete -> {merged_memory_output_path}",
            flush=True,
        )

        with open(merged_memory_output_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return {item["RPS_id"]: item for item in raw}
        if isinstance(raw, dict):
            return raw
        raise ValueError("Merged scene-semantic memory format is neither list nor dict")


@register_tool
class PerceptionPipelineTool(BaseTool):
    name = "perception_pipeline"
    category = ToolCategory.MEMORY_CONSTRUCTION

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        workspace_dir: str,
        spatialite_context: Dict[str, Any],
        scene_base_model: Optional[str] = None,
        scene_adapter_path: Optional[str] = None,
        semantic_base_model: Optional[str] = None,
        semantic_adapter_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pipe = PerceptionPipeline(
            workspace_dir,
            scene_base_model=scene_base_model,
            scene_adapter_path=scene_adapter_path,
            semantic_base_model=semantic_base_model,
            semantic_adapter_path=semantic_adapter_path,
        )
        out = pipe.run(spatialite_context)
        if ctx is not None:
            ctx.set("perception_memory_count", len(out) if isinstance(out, dict) else 0)
        return out
