"""Memory-aware, capability-constrained routing policy with explainable traces."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import json

from rcpp_core.model_registry import ModelDeployment, get_deployment, shadow_deployment

MemoryState = Literal["miss", "fresh", "partial", "stale", "conflicted"]
RiskLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class RoutingContext:
    run_id: str
    task_type: Literal["scene", "semantic"]
    memory_state: MemoryState = "miss"
    memory_quality: float = 0.0
    request_risk: RiskLevel = "medium"
    latency_budget_ms: Optional[int] = None
    cost_budget: Optional[float] = None
    shadow_enabled: bool = False


@dataclass
class RouteDecision:
    action: Literal["reuse_memory", "run_model", "incremental_inference", "human_review"]
    production_model: Optional[str]
    shadow_model: Optional[str]
    reasons: list[str] = field(default_factory=list)
    safeguards: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deployment_ready(deployment: Optional[ModelDeployment]) -> bool:
    if deployment is None:
        return False
    base = deployment.base_model
    adapter = deployment.adapter_path
    base_ready = Path(base).exists() or "/" in base  # remote model identifiers are resolvable at runtime
    adapter_ready = not adapter or Path(adapter).exists()
    return base_ready and adapter_ready


class MemoryAwareRouter:
    """Pure policy object; model execution stays in the inference layer."""

    def decide(self, context: RoutingContext) -> RouteDecision:
        production = get_deployment(context.task_type)
        candidate = shadow_deployment(context.task_type) if context.shadow_enabled else None
        shadow_key = candidate.key if _deployment_ready(candidate) else None
        safeguards = ["json_schema", "required_fields", "cross_field_consistency"]
        reasons: list[str] = []

        if context.memory_state == "fresh" and context.memory_quality >= 0.9:
            reasons.append("active memory is fresh and quality_score >= 0.9")
            return RouteDecision("reuse_memory", None, shadow_key, reasons, safeguards)
        if context.memory_state == "partial":
            reasons.append("memory is usable but incomplete; infer only missing fields")
            return RouteDecision(
                "incremental_inference", production.key, shadow_key, reasons, safeguards
            )
        if context.memory_state == "conflicted":
            reasons.append("memory versions conflict; force new evidence and retain both versions")
            safeguards.append("conflict_preservation")
        elif context.memory_state == "stale":
            reasons.append("memory is expired; refresh from current evidence")
        else:
            reasons.append("no reusable memory was found")

        if context.task_type == "scene":
            safeguards.append("no_text_only_fallback")
            if context.request_risk == "high":
                safeguards.append("human_review_on_second_failure")
        else:
            safeguards.extend(["numeric_copy_check", "deterministic_rule_fallback"])

        return RouteDecision("run_model", production.key, shadow_key, reasons, safeguards)


def write_route_trace(path: Path, context: RoutingContext, decision: RouteDecision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"context": asdict(context), "decision": decision.to_dict()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
