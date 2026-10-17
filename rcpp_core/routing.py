"""Memory-aware routing with measured quality, cost and latency constraints."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from rcpp_core.model_registry import MODEL_REGISTRY, ModelDeployment, get_deployment, shadow_deployment

MemoryState = Literal["miss", "fresh", "partial", "stale", "conflicted"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class RoutingContext:
    run_id: str
    task_type: Literal["scene", "semantic"]
    memory_state: MemoryState = "miss"
    memory_quality: float = 0.0
    request_risk: RiskLevel = "medium"
    latency_budget_ms: Optional[float] = None
    cost_budget: Optional[float] = None  # USD per complete extraction, including normal retries
    shadow_enabled: bool = False

    def __post_init__(self):
        for name in ("latency_budget_ms", "cost_budget"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ModelProfile:
    """Measured per-extraction statistics; never inferred from a model name."""
    p95_latency_ms: float
    mean_cost_usd: float
    quality: float
    sample_count: int
    measured_at: str
    approved: bool = False
    base_model: str = ""
    adapter_version: str = ""

    def __post_init__(self):
        for name in ("p95_latency_ms", "mean_cost_usd", "quality"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid profile {name}")
        if self.quality > 1 or type(self.sample_count) is not int or self.sample_count < 1:
            raise ValueError("Profile requires quality in [0,1] and positive sample_count")
        if type(self.approved) is not bool:
            raise ValueError("approved must be a boolean")
        measured = datetime.fromisoformat(self.measured_at.replace("Z", "+00:00"))
        if measured.tzinfo is None or measured > datetime.now(timezone.utc):
            raise ValueError("measured_at must be a past timezone-aware timestamp")


def load_profiles(path: str | None = None) -> dict[str, ModelProfile]:
    source = path or os.getenv("RCPP_ROUTING_PROFILES")
    if not source:
        return {}
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    return {key: ModelProfile(**value) for key, value in data.items()}


@dataclass
class RouteDecision:
    action: Literal["reuse_memory", "run_model", "incremental_inference", "human_review", "rule_fallback"]
    production_model: Optional[str]
    shadow_model: Optional[str]
    reasons: list[str] = field(default_factory=list)
    safeguards: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deployment_ready(deployment: Optional[ModelDeployment]) -> bool:
    return deployment is not None and deployment.availability == "ready"


class MemoryAwareRouter:
    def __init__(self, profiles: Mapping[str, ModelProfile] | None = None,
                 registry: Mapping[str, ModelDeployment] | None = None):
        self.profiles = dict(profiles) if profiles is not None else load_profiles()
        self.registry = registry if registry is not None else MODEL_REGISTRY

    def decide(self, context: RoutingContext) -> RouteDecision:
        production = get_deployment(context.task_type)
        candidate = shadow_deployment(context.task_type) if context.shadow_enabled else None
        bounded = context.latency_budget_ms is not None or context.cost_budget is not None
        shadow_key = candidate.key if not bounded and _deployment_ready(candidate) else None
        safeguards = ["json_schema", "required_fields", "cross_field_consistency"]
        if context.task_type == "scene":
            safeguards.append("no_text_only_fallback")
            if context.request_risk == "high":
                safeguards.append("human_review_on_second_failure")
        else:
            safeguards.extend(["numeric_copy_check", "deterministic_rule_fallback"])
        if context.memory_state == "fresh" and context.memory_quality >= 0.9:
            return RouteDecision("reuse_memory", None, None,
                                 ["active memory is fresh and quality_score >= 0.9"], safeguards)
        reasons = [{
            "partial": "memory is incomplete; refresh the affected memory layer",
            "conflicted": "memory versions conflict; refresh and preserve evidence",
            "stale": "memory is expired; refresh from current evidence",
        }.get(context.memory_state, "no reusable memory was found")]
        if context.memory_state == "conflicted":
            safeguards.append("conflict_preservation")
        action = "incremental_inference" if context.memory_state == "partial" else "run_model"
        if not bounded and not self.profiles:
            return RouteDecision(action, production.key, shadow_key, reasons, safeguards)

        minimum_quality = 0.95 if context.request_risk == "high" else 0.9
        evaluated = []
        eligible = []
        for key, deployment in sorted(self.registry.items()):
            if deployment.kind != context.task_type:
                continue
            profile = self.profiles.get(key)
            rejected = []
            if deployment.stage != "production":
                rejected.append("not a production-approved deployment")
            if not _deployment_ready(deployment):
                rejected.append("model or adapter is unavailable")
            if profile is None:
                rejected.append("measured profile is missing")
            else:
                if not profile.approved:
                    rejected.append("profile is not approved")
                if profile.base_model != deployment.base_model or profile.adapter_version != deployment.adapter_version:
                    rejected.append("profile does not match the deployed model and adapter")
                if profile.quality < minimum_quality:
                    rejected.append("quality below risk threshold")
                if context.cost_budget is not None and profile.mean_cost_usd > context.cost_budget:
                    rejected.append("mean cost exceeds budget")
                if context.latency_budget_ms is not None and profile.p95_latency_ms > context.latency_budget_ms:
                    rejected.append("P95 latency exceeds budget")
            evaluated.append({"model": key, "eligible": not rejected, "reasons": rejected,
                              "profile": asdict(profile) if profile else None})
            if not rejected:
                eligible.append((key, profile))
        if not eligible:
            reasons.append("no approved model satisfies measured quality, cost and latency constraints")
            fallback = "human_review" if context.task_type == "scene" else "rule_fallback"
            return RouteDecision(fallback, None, None, reasons, safeguards, evaluated)
        selected, profile = min(eligible, key=lambda x: (x[1].mean_cost_usd, x[1].p95_latency_ms, x[0]))
        reasons.append(f"selected {selected}: mean USD {profile.mean_cost_usd}, P95 {profile.p95_latency_ms} ms")
        return RouteDecision(action, selected, shadow_key, reasons, safeguards, evaluated)


def write_route_trace(path: Path, context: RoutingContext, decision: RouteDecision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"context": asdict(context), "decision": decision.to_dict()},
                               ensure_ascii=False, indent=2), encoding="utf-8")
