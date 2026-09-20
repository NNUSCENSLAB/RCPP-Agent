"""Lifecycle classification used before long-term memory reuse."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Literal

MemoryLifecycleState = Literal["active", "superseded", "conflicted", "expired"]


def classify_memory(
    memory: Mapping[str, Any], *, now: datetime | None = None
) -> MemoryLifecycleState:
    meta = memory.get("_meta") if isinstance(memory.get("_meta"), Mapping) else {}
    status = str(meta.get("status") or "active").lower()
    if status in {"superseded", "conflicted", "expired"}:
        return status  # type: ignore[return-value]
    expires_at = str(meta.get("expires_at") or "").strip()
    if expires_at:
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            current = now or datetime.now(timezone.utc)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if current >= expiry:
                return "expired"
        except ValueError:
            return "conflicted"
    return "active"


def reusable(memory: Mapping[str, Any], minimum_quality: float = 0.9) -> bool:
    meta = memory.get("_meta") if isinstance(memory.get("_meta"), Mapping) else {}
    try:
        quality = float(meta.get("quality_score", 0.0))
    except (TypeError, ValueError):
        return False
    return classify_memory(memory) == "active" and quality >= minimum_quality
