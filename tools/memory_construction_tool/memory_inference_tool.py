# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import logging
import os
from typing import Any, Dict, Literal, Optional, Tuple

from rcpp_core import run_scene_inference_file, run_semantic_inference_file
from rcpp_core.model_registry import get_deployment, shadow_deployment
from tools.base import BaseTool, ToolCategory, ToolContext
from tools.registry import register_tool

MemoryInferenceKind = Literal["scene", "semantic"]
logger = logging.getLogger(__name__)


def _shadow_enabled() -> bool:
    return os.getenv("RCPP_SHADOW_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


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
        model_key: Optional[str] = None,
        scene_base_model: Optional[str] = None,
        scene_adapter_path: Optional[str] = None,
        semantic_base_model: Optional[str] = None,
        semantic_adapter_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if kind == "scene":
            deployment = get_deployment(
                "scene",
                model_key,
                base_model=scene_base_model,
                adapter_path=scene_adapter_path,
            )
            run_scene_inference_file(
                input_file,
                output_file,
                deployment.base_model,
                deployment.adapter_path,
                self._scene_cache,
                deployment=deployment,
            )
        elif kind == "semantic":
            deployment = get_deployment(
                "semantic",
                model_key,
                base_model=semantic_base_model,
                adapter_path=semantic_adapter_path,
            )
            run_semantic_inference_file(
                input_file,
                output_file,
                deployment.base_model,
                deployment.adapter_path,
                self._semantic_cache,
                max_new_tokens=max_new_tokens or deployment.max_new_tokens,
                max_length=max_length or deployment.max_length,
                deployment=deployment,
            )
        else:
            raise ValueError(f"unknown kind: {kind}")

        if _shadow_enabled():
            candidate = shadow_deployment(kind)
            if candidate is None or candidate.availability != "ready":
                logger.info("Shadow %s inference skipped: candidate model is disabled.", kind)
            else:
                shadow_output = str(Path(output_file).with_suffix(".shadow.json"))
                try:
                    if kind == "scene":
                        run_scene_inference_file(
                            input_file,
                            shadow_output,
                            candidate.base_model,
                            candidate.adapter_path,
                            self._scene_cache,
                            deployment=candidate,
                        )
                    else:
                        run_semantic_inference_file(
                            input_file,
                            shadow_output,
                            candidate.base_model,
                            candidate.adapter_path,
                            self._semantic_cache,
                            max_new_tokens=candidate.max_new_tokens,
                            max_length=candidate.max_length,
                            deployment=candidate,
                        )
                    logger.info("Shadow %s output written to %s; it is not merged.", kind, shadow_output)
                except Exception as exc:
                    logger.warning("Shadow %s inference failed without affecting production: %s", kind, exc)
        if ctx is not None:
            ctx.set(f"{kind}_inference_output", str(Path(output_file).resolve()))
