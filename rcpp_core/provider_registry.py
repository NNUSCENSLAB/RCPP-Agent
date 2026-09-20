"""Provider-neutral registry for optional general-purpose language models."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

from configs.config import get_provider_api_key


@dataclass(frozen=True)
class ProviderDeployment:
    key: str
    provider: str
    model: str
    roles: Tuple[str, ...]
    required: bool = False

    @property
    def ready(self) -> bool:
        return bool(get_provider_api_key(self.provider))


def general_reasoning_deployment() -> ProviderDeployment:
    provider = os.getenv("RCPP_GENERAL_PROVIDER", "deepseek").strip().lower()
    default_model = "deepseek-chat" if provider == "deepseek" else "gpt-4.1-mini"
    return ProviderDeployment(
        key="general_reasoning",
        provider=provider,
        model=os.getenv("RCPP_GENERAL_MODEL", default_model),
        roles=("instruction_parse", "ahp_generation", "explanation_repair"),
        required=False,
    )


def auxiliary_registry() -> Dict[str, ProviderDeployment | Dict[str, object]]:
    return {
        "general_reasoning": general_reasoning_deployment(),
        "deterministic_rules": {
            "provider": "local",
            "model": "rcpp_core.semantic_rules",
            "roles": ("fact_compute", "schema_validate", "consistency_check"),
            "required": True,
            "ready": True,
        },
        "cached_ahp": {
            "provider": "local",
            "model": "versioned-static-policy",
            "roles": ("ahp_generation",),
            "required": True,
            "ready": True,
        },
    }
