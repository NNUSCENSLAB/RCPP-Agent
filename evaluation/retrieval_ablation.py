# -*- coding: utf-8 -*-
"""Eager vs. Scene→Semantic progressive retrieval ablation.

The eager policy is evaluation-only. Production continues to use progressive
retrieval through ``GraphMemoryRepository``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Protocol, Sequence

from rcpp_core.graph_memory_repository import GraphMemoryRepository, assess_scene
from rcpp_core.memory_migration import MigrationPlan, prepare_migration


POLICIES = ("eager", "progressive")


def _clean(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (payload or {}).items() if key != "_meta"}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class TokenCounter:
    """Exact Hugging Face token counting with a clearly labelled fallback."""

    def __init__(self, tokenizer: str | None = None):
        self.name = tokenizer or "utf8-bytes/4-estimate"
        self.mode = "estimated"
        self._tokenizer: Any = None
        if tokenizer:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("--tokenizer requires transformers") from exc
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
            self.mode = "exact"

    def count(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        # Useful for local smoke tests only. Reports retain token_count_mode=estimated.
        return math.ceil(len(text.encode("utf-8")) / 4)


class MemorySource(Protocol):
    def get_scene(self, area: str, rps_ids: Sequence[str]) -> dict[str, dict[str, Any]]: ...
    def get_semantic(self, area: str, rps_ids: Sequence[str]) -> dict[str, dict[str, Any]]: ...


class JsonMemorySource:
    def __init__(self, plan: MigrationPlan):
        self.scene = {row["rps_id"]: {**row["memory"], "_meta": {"status": row["status"]}}
                      for row in plan.scene}
        self.semantic = {row["rps_id"]: {**row["memory"], "_meta": {"status": row["status"]}}
                         for row in plan.semantic}

    def get_scene(self, area: str, rps_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        return {rps_id: self.scene[rps_id] for rps_id in rps_ids if rps_id in self.scene}

    def get_semantic(self, area: str, rps_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        return {rps_id: self.semantic[rps_id] for rps_id in rps_ids if rps_id in self.semantic}


class Neo4jMemorySource:
    def __init__(self, repository: GraphMemoryRepository | None = None):
        self.repository = repository or GraphMemoryRepository()

    def get_scene(self, area: str, rps_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        return self.repository.get_current_scene(area, rps_ids)

    def get_semantic(self, area: str, rps_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        return self.repository.get_current_semantic(area, rps_ids)


@dataclass(frozen=True)
class RpsCase:
    rps_id: str
    street_name: str
    reference_needs_semantic: bool


@dataclass
class RunMetrics:
    policy: str
    repeat: int
    rps_count: int
    scene_pass_count: int
    semantic_request_count: int
    unnecessary_semantic_count: int
    semantic_route_precision: float
    semantic_route_recall: float
    retrieved_context_bytes: int
    retrieved_context_tokens: int
    planning_context_bytes: int
    planning_context_tokens: int
    scene_query_ms: float
    route_latency_ms: float
    semantic_query_ms: float
    context_build_ms: float
    total_latency_ms: float


def cases_from_plan(plan: MigrationPlan, cohort: str) -> list[RpsCase]:
    rps_meta = {row["rps_id"]: row for row in plan.rps}
    cases: list[RpsCase] = []
    for row in plan.scene:
        if cohort == "detailed-active" and row["status"] != "active":
            continue
        if cohort == "detailed-all" and row["status"] == "partial":
            continue
        assessment = assess_scene(row["memory"], declared_status=row["status"])
        cases.append(RpsCase(
            rps_id=row["rps_id"],
            street_name=str(rps_meta.get(row["rps_id"], {}).get("street_name") or "unknown"),
            reference_needs_semantic=assessment.allows_semantic,
        ))
    return cases


def cases_from_graph(repository: GraphMemoryRepository, area: str, cohort: str) -> list[RpsCase]:
    rps_rows = repository.list_rps(area)
    ids = [str(row["rps_id"]) for row in rps_rows]
    scene = repository.get_current_scene(area, ids)
    cases: list[RpsCase] = []
    for row in rps_rows:
        rps_id = str(row["rps_id"])
        payload = scene.get(rps_id)
        status = str((payload or {}).get("_meta", {}).get("status") or "miss")
        if cohort == "detailed-active" and status != "active":
            continue
        if cohort == "detailed-all" and status == "partial":
            continue
        assessment = assess_scene(payload or {}, declared_status=status)
        cases.append(RpsCase(rps_id, str(row.get("street_name") or "unknown"),
                             assessment.allows_semantic))
    return cases


def run_policy(
    *, policy: str, repeat: int, source: MemorySource, area: str,
    cases: Sequence[RpsCase], counter: TokenCounter,
) -> tuple[RunMetrics, list[dict[str, Any]]]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown retrieval policy: {policy}")
    ids = [case.rps_id for case in cases]
    reference_ids = {case.rps_id for case in cases if case.reference_needs_semantic}
    street_by_id = {case.rps_id: case.street_name for case in cases}
    total_start = time.perf_counter_ns()

    start = time.perf_counter_ns()
    scenes = source.get_scene(area, ids)
    scene_query_ms = (time.perf_counter_ns() - start) / 1_000_000

    start = time.perf_counter_ns()
    passed_ids = []
    states: dict[str, str] = {}
    for rps_id in ids:
        payload = scenes.get(rps_id)
        status = str((payload or {}).get("_meta", {}).get("status") or "miss")
        assessment = assess_scene(payload or {}, declared_status=status)
        states[rps_id] = assessment.state
        if assessment.allows_semantic:
            passed_ids.append(rps_id)
    requested_ids = ids if policy == "eager" else passed_ids
    route_latency_ms = (time.perf_counter_ns() - start) / 1_000_000

    start = time.perf_counter_ns()
    semantics = source.get_semantic(area, requested_ids) if requested_ids else {}
    semantic_query_ms = (time.perf_counter_ns() - start) / 1_000_000

    start = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    retrieved_bytes = retrieved_tokens = planning_bytes = planning_tokens = 0
    requested = set(requested_ids)
    passed = set(passed_ids)
    for rps_id in ids:
        scene_clean = _clean(scenes.get(rps_id))
        semantic_clean = _clean(semantics.get(rps_id)) if rps_id in requested else {}
        retrieved_payload = {
            "RPS_id": rps_id,
            "scene_memory": scene_clean,
            **({"semantic_memory": semantic_clean} if semantic_clean else {}),
        }
        retrieved_text = _json_text(retrieved_payload)
        row_retrieved_bytes = len(retrieved_text.encode("utf-8"))
        row_retrieved_tokens = counter.count(retrieved_text)
        retrieved_bytes += row_retrieved_bytes
        retrieved_tokens += row_retrieved_tokens

        enters_planning = policy == "eager" or rps_id in passed
        if enters_planning:
            planning_bytes += row_retrieved_bytes
            planning_tokens += row_retrieved_tokens
        rows.append({
            "policy": policy, "repeat": repeat, "rps_id": rps_id,
            "street_name": street_by_id[rps_id], "scene_state": states[rps_id],
            "scene_passed": rps_id in passed,
            "reference_needs_semantic": rps_id in reference_ids,
            "semantic_requested": rps_id in requested,
            "semantic_returned": rps_id in semantics,
            "retrieved_context_bytes": row_retrieved_bytes,
            "retrieved_context_tokens": row_retrieved_tokens,
            "planning_context_bytes": row_retrieved_bytes if enters_planning else 0,
            "planning_context_tokens": row_retrieved_tokens if enters_planning else 0,
        })
    context_build_ms = (time.perf_counter_ns() - start) / 1_000_000
    total_latency_ms = (time.perf_counter_ns() - total_start) / 1_000_000

    true_positive = len(requested & reference_ids)
    precision = true_positive / len(requested) if requested else 1.0
    recall = true_positive / len(reference_ids) if reference_ids else 1.0
    metrics = RunMetrics(
        policy=policy, repeat=repeat, rps_count=len(ids), scene_pass_count=len(passed),
        semantic_request_count=len(requested),
        unnecessary_semantic_count=len(requested - reference_ids),
        semantic_route_precision=precision, semantic_route_recall=recall,
        retrieved_context_bytes=retrieved_bytes, retrieved_context_tokens=retrieved_tokens,
        planning_context_bytes=planning_bytes, planning_context_tokens=planning_tokens,
        scene_query_ms=scene_query_ms, route_latency_ms=route_latency_ms,
        semantic_query_ms=semantic_query_ms, context_build_ms=context_build_ms,
        total_latency_ms=total_latency_ms,
    )
    return metrics, rows


def _bootstrap_token_reduction(
    eager_rows: Sequence[Mapping[str, Any]], progressive_rows: Sequence[Mapping[str, Any]],
    *, samples: int, seed: int = 42,
) -> tuple[float, float]:
    eager = {str(row["rps_id"]): row for row in eager_rows}
    progressive = {str(row["rps_id"]): row for row in progressive_rows}
    clusters: dict[str, list[str]] = {}
    for rps_id, row in eager.items():
        clusters.setdefault(str(row["street_name"]), []).append(rps_id)
    names = sorted(clusters)
    if not names or samples <= 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    reductions = []
    for _ in range(samples):
        selected = rng.choices(names, k=len(names))
        eager_total = progressive_total = 0
        for name in selected:
            for rps_id in clusters[name]:
                eager_total += int(eager[rps_id]["retrieved_context_tokens"])
                progressive_total += int(progressive[rps_id]["retrieved_context_tokens"])
        reductions.append(1.0 - progressive_total / eager_total if eager_total else 0.0)
    return _percentile(reductions, 0.025), _percentile(reductions, 0.975)


def summarize(
    run_metrics: Sequence[RunMetrics], per_rps_rows: Sequence[Mapping[str, Any]],
    *, bootstrap_samples: int,
) -> dict[str, Any]:
    by_policy = {policy: [m for m in run_metrics if m.policy == policy] for policy in POLICIES}
    summary: dict[str, Any] = {"policies": {}}
    for policy, metrics in by_policy.items():
        latency = [m.total_latency_ms for m in metrics]
        first = metrics[0]
        summary["policies"][policy] = {
            "rps_count": first.rps_count,
            "scene_pass_count": first.scene_pass_count,
            "semantic_request_count": first.semantic_request_count,
            "unnecessary_semantic_count": first.unnecessary_semantic_count,
            "semantic_route_precision": first.semantic_route_precision,
            "semantic_route_recall": first.semantic_route_recall,
            "retrieved_context_bytes": first.retrieved_context_bytes,
            "retrieved_context_tokens": first.retrieved_context_tokens,
            "planning_context_bytes": first.planning_context_bytes,
            "planning_context_tokens": first.planning_context_tokens,
            "total_latency_ms_p50": _percentile(latency, 0.5),
            "total_latency_ms_p95": _percentile(latency, 0.95),
            "route_latency_ms_p50": _percentile([m.route_latency_ms for m in metrics], 0.5),
            "semantic_query_ms_p50": _percentile([m.semantic_query_ms for m in metrics], 0.5),
        }
    eager = summary["policies"]["eager"]
    progressive = summary["policies"]["progressive"]
    first_rows = {
        policy: [row for row in per_rps_rows if row["policy"] == policy and row["repeat"] == 1]
        for policy in POLICIES
    }
    low, high = _bootstrap_token_reduction(
        first_rows["eager"], first_rows["progressive"], samples=bootstrap_samples
    )
    summary["comparison"] = {
        "semantic_request_reduction": 1.0 - progressive["semantic_request_count"] / eager["semantic_request_count"],
        "retrieved_context_token_reduction": 1.0 - progressive["retrieved_context_tokens"] / eager["retrieved_context_tokens"],
        "retrieved_context_token_reduction_ci95": [low, high],
        "planning_context_token_reduction": 1.0 - progressive["planning_context_tokens"] / eager["planning_context_tokens"],
        "p50_latency_reduction": 1.0 - progressive["total_latency_ms_p50"] / eager["total_latency_ms_p50"],
        "p95_latency_reduction": 1.0 - progressive["total_latency_ms_p95"] / eager["total_latency_ms_p95"],
    }
    return summary


def _report(summary: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    eager = summary["policies"]["eager"]
    progressive = summary["policies"]["progressive"]
    comparison = summary["comparison"]

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    return f"""# Scene→Semantic retrieval ablation

## Configuration

- Source: `{metadata['source']}`
- Area: `{metadata['area']}`
- Cohort: `{metadata['cohort']}`
- Repeats: {metadata['repeats']} (+ {metadata['warmup']} warm-up)
- Token counter: `{metadata['tokenizer']}` ({metadata['token_count_mode']})

## Results

| Metric | Eager | Progressive | Change |
|---|---:|---:|---:|
| RPS | {eager['rps_count']} | {progressive['rps_count']} | — |
| Semantic requests | {eager['semantic_request_count']} | {progressive['semantic_request_count']} | {pct(comparison['semantic_request_reduction'])} reduction |
| Unnecessary Semantic requests | {eager['unnecessary_semantic_count']} | {progressive['unnecessary_semantic_count']} | — |
| Semantic route precision | {pct(eager['semantic_route_precision'])} | {pct(progressive['semantic_route_precision'])} | — |
| Semantic route recall | {pct(eager['semantic_route_recall'])} | {pct(progressive['semantic_route_recall'])} | — |
| Retrieved context tokens | {eager['retrieved_context_tokens']} | {progressive['retrieved_context_tokens']} | {pct(comparison['retrieved_context_token_reduction'])} reduction |
| Planning context tokens | {eager['planning_context_tokens']} | {progressive['planning_context_tokens']} | {pct(comparison['planning_context_token_reduction'])} reduction |
| End-to-end latency P50 | {eager['total_latency_ms_p50']:.3f} ms | {progressive['total_latency_ms_p50']:.3f} ms | {pct(comparison['p50_latency_reduction'])} |
| End-to-end latency P95 | {eager['total_latency_ms_p95']:.3f} ms | {progressive['total_latency_ms_p95']:.3f} ms | {pct(comparison['p95_latency_reduction'])} |

Retrieved-context token reduction 95% street-cluster bootstrap CI:
`[{pct(comparison['retrieved_context_token_reduction_ci95'][0])}, {pct(comparison['retrieved_context_token_reduction_ci95'][1])}]`.

## Interpretation boundary

`json` source measures in-process replay and context assembly. It does not establish Neo4j or
model-serving latency. Use `--source neo4j` for graph-query latency and a live model experiment
for actual prompt tokens, GPU time, and generation latency. Estimated token counts must not be
reported as exact model usage.
"""


def run_experiment(
    *, source: MemorySource, cases: Sequence[RpsCase], area: str, output_dir: Path,
    repeats: int = 10, warmup: int = 2, tokenizer: str | None = None,
    bootstrap_samples: int = 1000, source_name: str = "json", cohort: str = "detailed-active",
) -> dict[str, Any]:
    if not cases:
        raise ValueError("The selected cohort contains no RPS records")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup must be >= 0")
    counter = TokenCounter(tokenizer)
    for _ in range(warmup):
        for policy in POLICIES:
            run_policy(policy=policy, repeat=0, source=source, area=area,
                       cases=cases, counter=counter)

    run_metrics: list[RunMetrics] = []
    rows: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        order = POLICIES if repeat % 2 else tuple(reversed(POLICIES))
        for policy in order:
            metrics, detail = run_policy(policy=policy, repeat=repeat, source=source,
                                         area=area, cases=cases, counter=counter)
            run_metrics.append(metrics)
            rows.extend(detail)
    summary = summarize(run_metrics, rows, bootstrap_samples=bootstrap_samples)
    metadata = {
        "source": source_name, "area": area, "cohort": cohort,
        "repeats": repeats, "warmup": warmup,
        "tokenizer": counter.name, "token_count_mode": counter.mode,
        "bootstrap_samples": bootstrap_samples,
    }
    result = {"metadata": metadata, **summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rps_metrics.jsonl").write_text(
        "".join(_json_text(row) + "\n" for row in rows), encoding="utf-8"
    )
    (output_dir / "run_metrics.jsonl").write_text(
        "".join(_json_text(asdict(row)) + "\n" for row in run_metrics), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_report(result, metadata), encoding="utf-8")
    return result


def run_from_args(args: Any) -> dict[str, Any]:
    area = str(args.area).lower()
    if args.source == "json":
        if not args.scene_file or not args.combined_file:
            raise ValueError("--scene-file and --combined-file are required for --source json")
        plan = prepare_migration(area, args.scene_file, args.combined_file)
        source: MemorySource = JsonMemorySource(plan)
        cases = cases_from_plan(plan, args.cohort)
    else:
        repository = GraphMemoryRepository()
        source = Neo4jMemorySource(repository)
        cases = cases_from_graph(repository, area, args.cohort)
    return run_experiment(
        source=source, cases=cases, area=area, output_dir=Path(args.output),
        repeats=args.repeats, warmup=args.warmup, tokenizer=args.tokenizer,
        bootstrap_samples=args.bootstrap_samples, source_name=args.source,
        cohort=args.cohort,
    )
