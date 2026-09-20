# -*- coding: utf-8 -*-
"""
This module contains all configuration settings for the project.
"""
from __future__ import annotations
import os
from enum import Enum
from typing import Dict, Optional
from configs.data_config import DEFAULT_WORKSPACE


PROJECT_NAME = "RCPP-Agent"
VERSION = "1.1.0"

ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"

ENV_INSTRUCTION_PARSE_MODEL = "INSTRUCTION_PARSE_MODEL"
DEFAULT_INSTRUCTION_PARSE_MODEL = "deepseek-chat"
DEFAULT_INSTRUCTION_PARSE_TEMPERATURE = 0.0


def get_provider_api_key(provider: str) -> Optional[str]:
    """Return credentials for exactly one provider; never cross-wire keys."""
    normalized = provider.strip().lower()
    if normalized == "deepseek":
        return os.getenv(ENV_DEEPSEEK_API_KEY) or None
    if normalized == "openai":
        return os.getenv(ENV_OPENAI_API_KEY) or None
    env_name = f"{normalized.upper()}_API_KEY"
    return os.getenv(env_name) or None


def provider_for_model(model_name: str) -> str:
    configured = os.getenv("RCPP_GENERAL_PROVIDER", "").strip().lower()
    if configured:
        return configured
    return "deepseek" if "deepseek" in model_name.lower() else "openai"


def get_openai_api_key() -> Optional[str]:
    """Compatibility helper for genuine OpenAI calls only."""
    return get_provider_api_key("openai")


# Backwards-compatible defaults. Runtime selection now lives in
# ``rcpp_core.model_registry`` and can be switched with
# RCPP_SCENE_MODEL_KEY / RCPP_SEMANTIC_MODEL_KEY after shadow evaluation.
PERCEPTION_SCENE_BASE_MODEL = f"{DEFAULT_WORKSPACE}/Qwen2.5-VL/Qwen2.5-VL-7B-Instruct"
PERCEPTION_SCENE_ADAPTER_PATH = f"{DEFAULT_WORKSPACE}/Qwen2.5-VL/qwen-vl-finetune/RPS_SCENE_MEMORY/output/Qwen2.5-VL-7B/checkpoints_20251220_062503"
PERCEPTION_SEMANTIC_BASE_MODEL = f"{DEFAULT_WORKSPACE}/Qwen2.5-VL/Qwen2.5-7B-Instruct"
PERCEPTION_SEMANTIC_ADAPTER_PATH = f"{DEFAULT_WORKSPACE}/Qwen2.5-VL/qwen-vl-finetune/RPS_SEMANTIC_MEMORY/output/Qwen2.5-VL-7B/checkpoints_20251223_072424"


# Suitability Evaluation Agent: rule categories
RULE_CATEGORIES: Dict[str, str] = {
    "physical_space": "Physical Space",
    "surrounding_environment": "Surrounding Environment",
    "grid": "Power Grid",
    "traffic": "Traffic",
    "regulation": "Regulation",
}

# Multi-scenario Evaluation Agent: MCDA criterion dimensions
MCDA_DIMENSIONS: Dict[str, str] = {
    "technical": "Technical",
    "economic": "Economic",
    "social": "Social",
    "traffic": "Traffic",
    "policy": "Policy",
}

ENV_LLM_AHP_MODEL = "LLM_AHP_MODEL"
DEFAULT_LLM_AHP_MODEL = "deepseek-chat"
DEFAULT_LLM_AHP_TEMPERATURE = 0.1
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com"

LLM_AHP_MAX_RETRIES = 3

MULTI_SCENARIO = ("efficiency_oriented", "equity_oriented", "balance_oriented")

SCENARIO_RPS_PLANNING_RATIO: Dict[str, float] = {
    "efficiency_oriented": 0.6,  # Efficiency-oriented: 60% of candidates (stronger emphasis on best sites).
    "equity_oriented": 1.0,  # Equity-oriented: 100% of candidates (coverage-first).
    "balance_oriented": 0.8,  # Balanced: 80% of candidates (compromise).
}

# Phased Decision-Making Agent: planning phases
_PLANNING_PHASE_INITIATION = "Planning Initiation Phase"
_PLANNING_PHASE_SCALE_UP = "Scale-up Expansion Phase"
_PLANNING_PHASE_REFINEMENT = "Refinement and Improvement Phase"


class PlanningPhase(Enum):
    INITIAL_EXPLORATION = _PLANNING_PHASE_INITIATION
    RAPID_DEVELOPMENT = _PLANNING_PHASE_SCALE_UP
    MATURITY_OPTIMIZATION = _PLANNING_PHASE_REFINEMENT

PLANNING_PHASE: Dict[str, str] = {
    _PLANNING_PHASE_INITIATION: f"{_PLANNING_PHASE_INITIATION} (January 2025-December 2026)",
    _PLANNING_PHASE_SCALE_UP: f"{_PLANNING_PHASE_SCALE_UP} (January 2027-December 2028)",
    _PLANNING_PHASE_REFINEMENT: f"{_PLANNING_PHASE_REFINEMENT} (January 2029-December 2030)",
}


def get_instruction_parse_model() -> str:
    return os.getenv(ENV_INSTRUCTION_PARSE_MODEL, DEFAULT_INSTRUCTION_PARSE_MODEL)


def get_llm_ahp_model() -> str:
    return os.getenv(ENV_LLM_AHP_MODEL, DEFAULT_LLM_AHP_MODEL)
