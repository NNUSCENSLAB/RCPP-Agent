# -*- coding: utf-8 -*-
"""
Global instruction and output paths derived from spatialite_context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from configs.data_config import (
    DEFAULT_WORKSPACE,
    REL_OUTPUT_TMP_MEMORY_INFERENCE_INPUTS,
    default_district_import_base,
    default_poi_dir,
    default_spatialite_db_path,
    workspace_path,
)


def default_workspace_path() -> Path:
    return workspace_path()


def normalize_area_slug(area: str) -> str:
    return str(area).strip().lower()


def normalize_scenario_slug(scenario: str) -> str:
    return str(scenario).strip().lower().replace(" ", "_").replace("-", "_")


def time_horizon_segment(time_horizon: Any) -> str:
    if time_horizon is None:
        return "default_period"
    if isinstance(time_horizon, (list, tuple)) and len(time_horizon) >= 2:
        return f"{time_horizon[0]}_{time_horizon[1]}"
    if isinstance(time_horizon, (list, tuple)) and len(time_horizon) == 1:
        return str(time_horizon[0])
    return str(time_horizon)


def perception_artifact_dir(
    workspace_dir: Union[str, Path, None] = None,
    *,
    area: str,
    scenario: str,
    time_horizon: Any = None,
) -> Path:
    """
    Intermediate directory for a single Environment Perception Agent run (scene/semantic input JSON, etc.):
    {workspace}/output/tmp/memory_inference_inputs/{area_slug}/{scenario_slug}/{t0_t1}/
    """
    root = Path(workspace_dir or DEFAULT_WORKSPACE).resolve()
    a = normalize_area_slug(area)
    s = normalize_scenario_slug(scenario)
    seg = time_horizon_segment(time_horizon)
    out = root / REL_OUTPUT_TMP_MEMORY_INFERENCE_INPUTS / a / s / seg
    out.mkdir(parents=True, exist_ok=True)
    return out
