# -*- coding: utf-8 -*-
"""Central registry and routing policy for memory extraction models."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Dict, Literal, Optional

from configs.data_config import DEFAULT_WORKSPACE

MemoryKind = Literal["scene", "semantic"]


@dataclass(frozen=True)
class ModelDeployment:
    key: str
    kind: MemoryKind
    base_model: str
    adapter_path: str = ""
    family: str = "auto"
    stage: Literal["production", "shadow", "baseline"] = "baseline"
    max_length: int = 2048
    max_new_tokens: int = 512
    evidence_source: str = "model_inference"

    @property
    def adapter_version(self) -> str:
        if not self.adapter_path:
            return "base"
        return os.path.basename(self.adapter_path.rstrip("/\\")) or "adapter"

    @property
    def availability(self) -> str:
        """Report local readiness without loading model weights."""
        base_is_remote_id = "/" in self.base_model and not Path(self.base_model).is_absolute()
        if not (Path(self.base_model).exists() or base_is_remote_id):
            return "disabled"
        if self.adapter_path and not Path(self.adapter_path).exists():
            return "disabled"
        return "ready"


_OLD_ROOT = f"{DEFAULT_WORKSPACE}/Qwen2.5-VL"
_NEW_ROOT = os.getenv("RCPP_QWEN3_ROOT", f"{DEFAULT_WORKSPACE}/models")

MODEL_REGISTRY: Dict[str, ModelDeployment] = {
    "scene_qwen25_current": ModelDeployment(
        key="scene_qwen25_current",
        kind="scene",
        base_model=f"{_OLD_ROOT}/Qwen2.5-VL-7B-Instruct",
        adapter_path=(
            f"{_OLD_ROOT}/qwen-vl-finetune/RPS_SCENE_MEMORY/output/Qwen2.5-VL-7B/"
            "checkpoints_20251220_062503"
        ),
        family="qwen2_5_vl",
        stage="production",
        max_length=4096,
    ),
    "scene_qwen3_vl_8b_qlora": ModelDeployment(
        key="scene_qwen3_vl_8b_qlora",
        kind="scene",
        base_model=os.getenv("RCPP_SCENE_QWEN3_BASE", "Qwen/Qwen3-VL-8B-Instruct"),
        adapter_path=os.getenv(
            "RCPP_SCENE_QWEN3_ADAPTER", f"{_NEW_ROOT}/adapters/scene_qwen3_vl_8b_qlora"
        ),
        family="qwen3_vl",
        stage="shadow",
        max_length=4096,
    ),
    "semantic_qwen25_current": ModelDeployment(
        key="semantic_qwen25_current",
        kind="semantic",
        base_model=f"{_OLD_ROOT}/Qwen2.5-7B-Instruct",
        adapter_path=(
            f"{_OLD_ROOT}/qwen-vl-finetune/RPS_SEMANTIC_MEMORY/output/Qwen2.5-VL-7B/"
            "checkpoints_20251223_072424"
        ),
        family="qwen2_5",
        stage="production",
        max_length=2048,
        max_new_tokens=512,
    ),
    "semantic_qwen3_8b_qlora": ModelDeployment(
        key="semantic_qwen3_8b_qlora",
        kind="semantic",
        base_model=os.getenv("RCPP_SEMANTIC_QWEN3_BASE", "Qwen/Qwen3-8B"),
        adapter_path=os.getenv(
            "RCPP_SEMANTIC_QWEN3_ADAPTER", f"{_NEW_ROOT}/adapters/semantic_qwen3_8b_qlora"
        ),
        family="qwen3",
        stage="shadow",
        max_length=2048,
        max_new_tokens=512,
    ),
}

DEFAULT_MODEL_KEYS: Dict[MemoryKind, str] = {
    "scene": "scene_qwen25_current",
    "semantic": "semantic_qwen25_current",
}
MODEL_KEY_ENV: Dict[MemoryKind, str] = {
    "scene": "RCPP_SCENE_MODEL_KEY",
    "semantic": "RCPP_SEMANTIC_MODEL_KEY",
}

# Non-extraction roles are kept visible in the same pool without pretending a
# text-only API can inspect scene images.
AUXILIARY_MODEL_POOL = {
    "general_reasoning": {
        "provider": os.getenv("RCPP_GENERAL_PROVIDER", "deepseek"),
        "model": os.getenv("RCPP_GENERAL_MODEL", "deepseek-chat"),
        "roles": ("instruction_parse", "ahp_generation", "semantic_explanation_repair"),
        "required": False,
    },
    "deterministic_rules": {
        "provider": "local",
        "model": "rcpp_core.semantic_rules",
        "roles": ("semantic_thresholds", "schema_validation", "constraint_validation"),
        "required": True,
    },
}


def get_deployment(
    kind: MemoryKind,
    key: Optional[str] = None,
    *,
    base_model: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> ModelDeployment:
    selected = key or os.getenv(MODEL_KEY_ENV[kind], DEFAULT_MODEL_KEYS[kind])
    if selected not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model deployment {selected!r}")
    deployment = MODEL_REGISTRY[selected]
    if deployment.kind != kind:
        raise ValueError(f"Model {selected!r} is for {deployment.kind}, not {kind}")
    if base_model is not None or adapter_path is not None:
        deployment = replace(
            deployment,
            base_model=base_model or deployment.base_model,
            adapter_path=adapter_path if adapter_path is not None else deployment.adapter_path,
        )
    return deployment


def shadow_deployment(kind: MemoryKind) -> Optional[ModelDeployment]:
    env_name = f"RCPP_{kind.upper()}_SHADOW_MODEL_KEY"
    explicit = os.getenv(env_name)
    if explicit:
        return get_deployment(kind, explicit)
    for deployment in MODEL_REGISTRY.values():
        if deployment.kind == kind and deployment.stage == "shadow":
            return deployment
    return None
