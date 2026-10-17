# -*- coding: utf-8 -*-
"""Live Semantic-call ablation for eager and Scene→Semantic retrieval.

Scene outcomes come from the locked memory dataset because the original image
files are not part of this repository. Each selected Semantic workload is sent
to the provider once; both policies are aggregated from the same call records
so the comparison does not double API cost.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from configs.config import get_provider_api_key
from evaluation.retrieval_ablation import JsonMemorySource, RpsCase, cases_from_plan
from rcpp_core.memory_migration import prepare_migration


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-8B"


@dataclass(frozen=True)
class ApiCall:
    rps_id: str
    scene_passed: bool
    success: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    trace_id: str
    response: str
    error: str = ""


Transport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    tuple[dict[str, Any], Mapping[str, str]],
]


def _stdlib_transport(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> tuple[dict[str, Any], Mapping[str, str]]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured API endpoint
            body = json.loads(response.read().decode("utf-8"))
            return body, dict(response.headers.items())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Provider connection failed: {exc.reason}") from exc


class SiliconFlowClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 90.0,
        max_retries: int = 2,
        transport: Transport = _stdlib_transport,
    ) -> None:
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY is not configured")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport

    def complete(
        self, *, rps_id: str, semantic: Mapping[str, Any], scene_passed: bool
    ) -> ApiCall:
        messages = [
            {
                "role": "system",
                "content": (
                    "You normalize roadside-charging Semantic Memory. Return one JSON object only. "
                    "Preserve every supplied numeric fact exactly and do not add evidence. Include "
                    "rps_id, functional_zone, commuting_flow, sensitive_constraints, "
                    "grid_accessibility, and evidence_summary."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"rps_id": rps_id, "semantic_memory": dict(semantic)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RCPP-Agent/1.1 live-retrieval-ablation",
        }
        started = time.perf_counter()
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                body, response_headers = self.transport(
                    f"{self.base_url}/chat/completions", headers, payload, self.timeout
                )
                elapsed = (time.perf_counter() - started) * 1000
                usage = body.get("usage") or {}
                content = str(
                    (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                )
                json.loads(content)
                return ApiCall(
                    rps_id=rps_id,
                    scene_passed=scene_passed,
                    success=True,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                    latency_ms=elapsed,
                    trace_id=str(response_headers.get("x-siliconcloud-trace-id") or ""),
                    response=content,
                )
            except Exception as exc:  # retry transport, provider, and invalid JSON failures
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        return ApiCall(
            rps_id=rps_id,
            scene_passed=scene_passed,
            success=False,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            trace_id="",
            response="",
            error=last_error,
        )


def select_cases(cases: Sequence[RpsCase], max_calls: int, seed: int) -> list[RpsCase]:
    ordered = sorted(cases, key=lambda item: item.rps_id)
    if max_calls == 0 or max_calls >= len(ordered):
        return ordered
    if max_calls < 1:
        raise ValueError("--max-calls must be 0 (all) or a positive integer")
    passed = [item for item in ordered if item.reference_needs_semantic]
    stopped = [item for item in ordered if not item.reference_needs_semantic]
    rng = random.Random(seed)
    pass_count = round(max_calls * len(passed) / len(ordered)) if ordered else 0
    if max_calls >= 2 and passed and stopped:
        pass_count = min(max(pass_count, 1), max_calls - 1)
    selected = rng.sample(passed, min(pass_count, len(passed)))
    selected += rng.sample(stopped, min(max_calls - len(selected), len(stopped)))
    if len(selected) < max_calls:
        remaining = [item for item in ordered if item not in selected]
        selected += rng.sample(remaining, max_calls - len(selected))
    return sorted(selected, key=lambda item: item.rps_id)


def _cache_key(model: str, rps_id: str, semantic: Mapping[str, Any]) -> str:
    raw = json.dumps([model, rps_id, semantic], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> dict[str, ApiCall]:
    cache: dict[str, ApiCall] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.pop("cache_key"))
        call = ApiCall(**row)
        # Transient provider failures must be retryable on the next run.
        if call.success:
            cache[key] = call
    return cache


def _aggregate(calls: Sequence[ApiCall], progressive: bool) -> dict[str, Any]:
    selected = [call for call in calls if call.scene_passed or not progressive]
    successful = [call for call in selected if call.success]
    latencies = sorted(call.latency_ms for call in successful)

    def percentile(q: float) -> float:
        if not latencies:
            return 0.0
        return latencies[min(round((len(latencies) - 1) * q), len(latencies) - 1)]

    return {
        "semantic_request_count": len(selected),
        "successful_request_count": len(successful),
        "failed_request_count": len(selected) - len(successful),
        "success_rate": len(successful) / len(selected) if selected else 1.0,
        "prompt_tokens": sum(call.prompt_tokens for call in successful),
        "completion_tokens": sum(call.completion_tokens for call in successful),
        "total_tokens": sum(call.total_tokens for call in successful),
        "cumulative_api_latency_ms": sum(call.latency_ms for call in successful),
        "request_latency_ms_p50": percentile(0.5),
        "request_latency_ms_p95": percentile(0.95),
    }


def run_live_experiment(
    *,
    scene_file: str,
    combined_file: str,
    output_dir: Path,
    area: str = "gulou",
    cohort: str = "detailed-active",
    max_calls: int = 20,
    seed: int = 42,
    execute: bool = False,
    model: str = DEFAULT_MODEL,
    client: SiliconFlowClient | None = None,
) -> dict[str, Any]:
    plan = prepare_migration(area, scene_file, combined_file)
    source = JsonMemorySource(plan)
    cases = select_cases(cases_from_plan(plan, cohort), max_calls, seed)
    semantics = source.get_semantic(area, [case.rps_id for case in cases])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "api_calls.jsonl"
    metadata = {
        "provider": "siliconflow",
        "model": model,
        "area": area,
        "cohort": cohort,
        "seed": seed,
        "selected_rps_count": len(cases),
        "scene_pass_count": sum(case.reference_needs_semantic for case in cases),
        "scene_stop_count": sum(not case.reference_needs_semantic for case in cases),
        "mode": "live" if execute else "dry-run",
        "scene_source": "locked previously inferred Scene Memory; source images unavailable locally",
        "call_reuse": "one provider call per selected RPS, policy totals aggregated from identical calls",
    }
    if not execute:
        result = {"metadata": metadata, "planned_semantic_calls": len(cases)}
        (output_dir / "plan.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    live_client = client or SiliconFlowClient(
        api_key=get_provider_api_key("siliconflow") or "",
        model=model,
        base_url=os.getenv("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL),
    )
    cache = _read_cache(cache_path)
    calls: list[ApiCall] = []
    with cache_path.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            semantic = semantics.get(case.rps_id)
            if not semantic:
                calls.append(
                    ApiCall(
                        case.rps_id, case.reference_needs_semantic, False,
                        0, 0, 0, 0, "", "", "semantic memory missing",
                    )
                )
                continue
            key = _cache_key(model, case.rps_id, semantic)
            call = cache.get(key)
            if call is None:
                call = live_client.complete(
                    rps_id=case.rps_id,
                    semantic={k: v for k, v in semantic.items() if k != "_meta"},
                    scene_passed=case.reference_needs_semantic,
                )
                handle.write(
                    json.dumps({"cache_key": key, **asdict(call)}, ensure_ascii=False) + "\n"
                )
                handle.flush()
            calls.append(call)
            print(f"[{index}/{len(cases)}] {case.rps_id}: {'ok' if call.success else 'failed'}")

    eager = _aggregate(calls, progressive=False)
    progressive = _aggregate(calls, progressive=True)

    def reduction(key: str) -> float:
        denominator = float(eager[key])
        return 1.0 - float(progressive[key]) / denominator if denominator else 0.0

    result = {
        "metadata": metadata,
        "policies": {"eager": eager, "progressive": progressive},
        "comparison": {
            "semantic_request_reduction": reduction("semantic_request_count"),
            "prompt_token_reduction": reduction("prompt_tokens"),
            "completion_token_reduction": reduction("completion_tokens"),
            "total_token_reduction": reduction("total_tokens"),
            "cumulative_api_latency_reduction": reduction("cumulative_api_latency_ms"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pct = lambda value: f"{100 * value:.2f}%"  # noqa: E731
    report = f"""# Live Scene→Semantic retrieval ablation

## Configuration

- Provider: `siliconflow`
- Model: `{model}`
- Cohort: `{cohort}` ({len(cases)} sampled RPS)
- Scene pass / stop: {metadata['scene_pass_count']} / {metadata['scene_stop_count']}
- Scene source: locked previously inferred memory; source images unavailable locally
- Billing control: one cached provider call per selected RPS

## Results

| Metric | Eager | Progressive | Reduction |
|---|---:|---:|---:|
| Semantic requests | {eager['semantic_request_count']} | {progressive['semantic_request_count']} | {pct(result['comparison']['semantic_request_reduction'])} |
| Prompt tokens | {eager['prompt_tokens']} | {progressive['prompt_tokens']} | {pct(result['comparison']['prompt_token_reduction'])} |
| Completion tokens | {eager['completion_tokens']} | {progressive['completion_tokens']} | {pct(result['comparison']['completion_token_reduction'])} |
| Total tokens | {eager['total_tokens']} | {progressive['total_tokens']} | {pct(result['comparison']['total_token_reduction'])} |
| Cumulative API latency | {eager['cumulative_api_latency_ms']:.1f} ms | {progressive['cumulative_api_latency_ms']:.1f} ms | {pct(result['comparison']['cumulative_api_latency_reduction'])} |
| Success rate | {pct(eager['success_rate'])} | {pct(progressive['success_rate'])} | — |

## Interpretation boundary

Token counts come from the provider's `usage` object. Cumulative latency is the
sum of observed request durations and represents sequential Semantic workload;
it is not wall-clock latency under a particular concurrency or rate-limit policy.
Both policies reuse the same successful call records to avoid duplicate billing.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return result


def run_from_args(args: Any) -> dict[str, Any]:
    model = args.model or os.getenv("RCPP_SILICONFLOW_SEMANTIC_MODEL", DEFAULT_MODEL)
    return run_live_experiment(
        scene_file=args.scene_file,
        combined_file=args.combined_file,
        output_dir=Path(args.output),
        area=str(args.area).lower(),
        cohort=args.cohort,
        max_calls=args.max_calls,
        seed=args.seed,
        execute=args.execute,
        model=model,
    )
