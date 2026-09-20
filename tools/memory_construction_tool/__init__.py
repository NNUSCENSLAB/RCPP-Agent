# -*- coding: utf-8 -*-
"""Memory construction tools with dependency-safe lazy exports."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "MemoryInferenceTool": "memory_inference_tool",
    "MergeSceneSemanticMemoryTool": "merge_memory_tool",
    "run_merge": "merge_memory_tool",
    "ODFlowSummaryTool": "od_flow_tool",
    "load_od_flow_data": "od_flow_tool",
    "load_od_flow_data_from_gdfs": "od_flow_tool",
    "POIStatisticsAnalysisTool": "poi_statistics_tool",
    "POIStatisticsTool": "poi_statistics_tool",
    "SceneInferenceInputTool": "scene_inference_input_tool",
    "generate_inference_input_json": "scene_inference_input_tool",
    "generate_scene_inference_from_spatialite": "scene_inference_input_tool",
    "SemanticInferenceInputTool": "semantic_inference_input_tool",
    "generate_semantic_inference_from_spatialite": "semantic_inference_input_tool",
    "process_parking_spots": "semantic_inference_input_tool",
    "PerceptionPipeline": "perception_pipeline",
    "PerceptionPipelineTool": "perception_pipeline",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
