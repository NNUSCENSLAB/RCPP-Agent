# -*- coding: utf-8 -*-
"""
This module contains all data configuration settings for the project.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Union

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


# Environment variable names
ENV_RCPPAGENT_WORKSPACE = "RCPPAGENT_WORKSPACE"
ENV_RCPPAGENT_POI_DIR = "RCPPAGENT_POI_DIR"
ENV_RCPPAGENT_RAW_DATA_BASE = "RCPPAGENT_RAW_DATA_BASE"
ENV_RCPPAGENT_EXISTING_RCP_PATH = "RCPPAGENT_EXISTING_RCP_PATH"

# Neo4j global scene-semantic memory
ENV_NEO4J_URI = "NEO4J_URI"
ENV_NEO4J_USER = "NEO4J_USER"
ENV_NEO4J_PASSWORD = "NEO4J_PASSWORD"
ENV_NEO4J_DATABASE = "NEO4J_DATABASE"

# Default workspace when RCPPAGENT_WORKSPACE is not set
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE_FALLBACK = str(_DEFAULT_PROJECT_ROOT)

DEFAULT_WORKSPACE = os.environ.get(ENV_RCPPAGENT_WORKSPACE, DEFAULT_WORKSPACE_FALLBACK)
 
RCPP_AGENT_MAX_WORKFLOW_ITERATIONS = 3
REVIEW_SCORE_THRESHOLD_DEFAULT = 0.6

# Path segments relative to workspace root
REL_SPATIALITE_DB = Path("data") / "rcpp_spatial.sqlite"
REL_CHECKPOINT_DB = Path("data") / "langgraph_checkpoints.sqlite"
REL_POI_SUBDIR = Path("data") / "POI"
REL_DEFAULT_DISTRICT_DATA = Path("data") / "China_jiangsu_nanjing_gulou"
REL_EXISTING_RCP_SHP = Path("data") / "gulou_existing_rcp" / "gulou_existing_rcp.shp"

REL_OUTPUT_TMP_MEMORY_INFERENCE_INPUTS = Path("output") / "tmp" / "memory_inference_inputs"

# Default SpatiaLite logical table names
DEFAULT_SPATIALITE_TABLES: Dict[str, str] = {
    "rps": "rps",
    "bsv_image": "bsv_image",
    "poi": "poi",
    "od_region": "od_region",
    "od_flow": "od_flow",
    "roads": "roads",
}

# SpatiaLite import and POI load mapping for review metrics.
POI_FILENAME_MAPPING: Dict[str, str] = {
    "residential": "residential",
    "commercial": "commercial",
    "medical": "medical",
    "education": "education",
    "public": "public_admin",
    "transportation": "transportation",
    "grid_power": "grid",
}

# Review Agent configuration
REVIEW_CONFIG: Dict[str, Any] = {
    "service_radius_r": 1000.0,
    "E_charge": 48.0,
    "n_per_rps": 1,
    "mu_charges_per_day": 20,
    "gas_price": 6.8,
    "electricity_price": 1.2,
    "icev_consumption": 0.07,
    "ev_efficiency": 0.15,
    "P_rate": 62.5,
    "V_nom": 10.0,
    "rho": 0.2,
    "baseline_ratio": 0.1,
    "temporal_periods": {
        "morning_peak": {"start": "08:30", "end": "10:00", "duration_hours": 1.5, "factor": 1.2},
        "evening_peak": {"start": "17:30", "end": "19:30", "duration_hours": 2.0, "factor": 1.2},
        "other": {"duration_hours": 20.5, "factor": 0.6},
    },
}


def workspace_path() -> Path:
    return Path(os.environ.get(ENV_RCPPAGENT_WORKSPACE, DEFAULT_WORKSPACE_FALLBACK)).resolve()


def default_spatialite_db_path() -> Path:
    return workspace_path() / REL_SPATIALITE_DB


def default_checkpoint_db_path() -> Path:
    return workspace_path() / REL_CHECKPOINT_DB


def default_poi_dir(workspace_dir: Union[str, Path, None] = None) -> Path:
    override = os.environ.get(ENV_RCPPAGENT_POI_DIR)
    if override:
        return Path(override).expanduser().resolve()
    root = Path(workspace_dir or os.environ.get(ENV_RCPPAGENT_WORKSPACE, DEFAULT_WORKSPACE_FALLBACK)).resolve()
    return root / REL_POI_SUBDIR


def default_district_import_base() -> Path:
    override = os.environ.get(ENV_RCPPAGENT_RAW_DATA_BASE)
    if override:
        return Path(override).expanduser().resolve()
    return workspace_path() / REL_DEFAULT_DISTRICT_DATA


def normalize_area_slug(area: str) -> str:
    return str(area).strip().lower()


def default_existing_rcp_shp_path() -> str:
    env = os.environ.get(ENV_RCPPAGENT_EXISTING_RCP_PATH)
    if env:
        return str(Path(env).expanduser().resolve())
    return str(workspace_path() / REL_EXISTING_RCP_SHP)


def path_output_environment_perception_agent() -> Path:
    return workspace_path() / "output" / "environment_perception_agent_output"


def path_output_suitability_assessment_agent() -> Path:
    return workspace_path() / "output" / "suitability_assessment_agent_output"


def path_output_multi_scenario_evaluation_agent() -> Path:
    return workspace_path() / "output" / "multi_scenario_evaluation_agent_output"


def path_output_phased_decision_making_agent() -> Path:
    return workspace_path() / "output" / "phased_decision_making_agent_output"


def path_output_review_agent() -> Path:
    return workspace_path() / "output" / "review_agent_output"


def path_default_scene_semantic_memory_json(
    area: str,
    workspace_dir: Union[str, Path, None] = None,
) -> Path:
    """
    Default pipeline output JSON.
    """
    a = normalize_area_slug(area)
    root = (
        Path(str(workspace_dir)).expanduser().resolve()
        if workspace_dir
        else workspace_path()
    )
    return root / "output" / "environment_perception_agent_output" / f"scene_semantic_memory_{a}.json"


def path_default_suitable_rps_shp(area: str) -> Path:
    a = normalize_area_slug(area)
    return path_output_suitability_assessment_agent() / f"suitable_rps_{a}.shp"


def build_review_config() -> Dict[str, Any]:
    cfg = dict(REVIEW_CONFIG)
    cfg["existing_rps_path"] = default_existing_rcp_shp_path()
    return cfg


REVIEW_AGENT_CONFIG: Dict[str, Any] = build_review_config()
