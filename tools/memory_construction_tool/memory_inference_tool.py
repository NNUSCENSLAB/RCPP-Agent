# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from rcpp_core import (
    run_scene_inference_file,
    run_semantic_inference_file,
    PERCEPTION_SCENE_BASE_MODEL,
    PERCEPTION_SCENE_ADAPTER_PATH,
    PERCEPTION_SEMANTIC_BASE_MODEL,
    PERCEPTION_SEMANTIC_ADAPTER_PATH,
)
from tools.base import BaseTool, ToolCategory, ToolContext
from tools.registry import register_tool

MemoryInferenceKind = Literal["scene", "semantic"]


@register_tool
class MemoryInferenceTool(BaseTool):
    name = "memory_inference"
    category = ToolCategory.MEMORY_CONSTRUCTION

    def __init__(self) -> None:
        self._scene_cache: Dict[Tuple[str, str], Tuple[Any, Any, Any]] = {}
        self._semantic_cache: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        kind: MemoryInferenceKind,
        input_file: str,
        output_file: str,
        scene_base_model: str = PERCEPTION_SCENE_BASE_MODEL,
        scene_adapter_path: str = PERCEPTION_SCENE_ADAPTER_PATH,
        semantic_base_model: str = PERCEPTION_SEMANTIC_BASE_MODEL,
        semantic_adapter_path: str = PERCEPTION_SEMANTIC_ADAPTER_PATH,
        max_new_tokens: int = 256,
        max_length: int = 1024,
        **kwargs: Any,
    ) -> None:
        if kind == "scene":
            run_scene_inference_file(
                input_file,
                output_file,
                scene_base_model,
                scene_adapter_path,
                self._scene_cache,
            )
        elif kind == "semantic":
            run_semantic_inference_file(
                input_file,
                output_file,
                semantic_base_model,
                semantic_adapter_path,
                self._semantic_cache,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )
        else:
            raise ValueError(f"unknown kind: {kind}")
        if ctx is not None:
            ctx.set(f"{kind}_inference_output", str(Path(output_file).resolve()))
