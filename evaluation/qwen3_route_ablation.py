# -*- coding: utf-8 -*-
"""Evaluate eager retrieval against Qwen3-driven Scene→Semantic retrieval."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from evaluation.retrieval_ablation import TokenCounter, _json_text, _percentile


def _load_latest(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[str(row["rps_id"])] = row
    return latest


def _load_combined(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("combined file must contain a JSON list")
    return {str(row["RPS_id"]): row for row in payload}


def _allows_semantic(scene: Mapping[str, Any]) -> bool:
    return (
        scene.get("clearance_visual_assessment") is True
        and scene.get("has_existing_RCP") is False
        and scene.get("is_functional_zone") is False
    )


def _reference_allows_semantic(row: Mapping[str, Any]) -> bool | None:
    value = row.get("memory_node", {}).get("scene_memory", {}).get(
        "clearance_visual_assessment"
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _bootstrap_reduction(
    rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int = 42
) -> list[float]:
    clusters: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        clusters.setdefault(str(row["street_name"]), []).append(row)
    names = sorted(clusters)
    if not names or samples <= 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        eager = progressive = 0
        for name in rng.choices(names, k=len(names)):
            for row in clusters[name]:
                eager += int(row["eager_retrieved_tokens"])
                progressive += int(row["progressive_retrieved_tokens"])
        values.append(1.0 - progressive / eager if eager else 0.0)
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _render_report(result: Mapping[str, Any]) -> str:
    scene = result["scene_api"]
    route = result["route_ablation"]
    ref = route["historical_final_clearance_reference"]
    ci = route["retrieved_context_token_reduction_ci95"]
    historical = result.get("historical_lora_route_ablation")
    historical_section = ""
    if historical:
        eager = historical["policies"]["eager"]
        progressive = historical["policies"]["progressive"]
        comparison = historical["comparison"]
        stopped = eager["rps_count"] - progressive["semantic_request_count"]
        historical_section = f"""
## Upstream-model sensitivity check

The same retrieval policy was replayed on the historical LoRA Scene output:

| Metric | Eager retrieval | Progressive retrieval | Reduction |
|---|---:|---:|---:|
| Evaluated detailed-active RPS | {eager['rps_count']:,} | {progressive['rps_count']:,} | — |
| Semantic retrievals | {eager['semantic_request_count']:,} | {progressive['semantic_request_count']:,} | {_pct(comparison['semantic_request_reduction'])} |
| Retrieved context tokens | {eager['retrieved_context_tokens']:,} | {progressive['retrieved_context_tokens']:,} | {_pct(comparison['retrieved_context_token_reduction'])} |
| Planning context tokens | {eager['planning_context_tokens']:,} | {progressive['planning_context_tokens']:,} | {_pct(comparison['planning_context_token_reduction'])} |

This sensitivity check separates the retrieval mechanism from its upstream Scene extractor.
With the historical LoRA output, the policy stops {stopped:,} RPS. With the unfine-tuned
Qwen3-VL output, it stops only {route['scene_stop_count']:,}. The current base-model experiment
does not support a claim that the deployed Qwen3 configuration materially reduces inference.
It shows that the benefit depends on a selective and validated Scene layer.
"""
    return f"""# Qwen3 Scene extraction and Scene→Semantic retrieval experiment

## Dataset and execution

- Local street-view images: {scene['selected_case_count']:,}
- Successful Qwen3-VL responses: {scene['success_count']:,}
- Provider failures: {scene['failure_count']:,}
- JSON Schema success rate: {_pct(scene['json_schema_success_rate'])}
- Successful provider responses recorded: {scene['request_count']:,}
- Actual API usage: {scene['total_tokens']:,} tokens
- Latency: P50 {scene['latency_ms_p50'] / 1000:.2f}s, P95 {scene['latency_ms_p95'] / 1000:.2f}s

One image repeatedly returned provider HTTP 500. It is retained as a failure rather than
replaced with a fabricated result. The combined file references another 164 images that are
not present in the local dataset, so they were not sent to the vision model.

## Retrieval ablation

| Metric | Eager retrieval | Scene→Semantic progressive retrieval | Reduction |
|---|---:|---:|---:|
| Evaluated RPS | {route['evaluated_rps_count']:,} | {route['evaluated_rps_count']:,} | — |
| Semantic retrievals | {route['eager_semantic_request_count']:,} | {route['progressive_semantic_request_count']:,} | {_pct(route['semantic_request_reduction'])} |
| Retrieved context tokens | {route['eager_retrieved_context_tokens']:,} | {route['progressive_retrieved_context_tokens']:,} | {_pct(route['retrieved_context_token_reduction'])} |
| Planning context tokens | {route['eager_planning_context_tokens']:,} | {route['progressive_planning_context_tokens']:,} | {_pct(route['planning_context_token_reduction'])} |

Street-cluster bootstrap 95% CI for retrieved-context token reduction:
`[{_pct(ci[0])}, {_pct(ci[1])}]`.

The progressive policy stopped {route['scene_stop_count']:,} RPS after Scene evaluation and
therefore avoided the same number of Semantic retrievals. Token counts in this section use
the explicitly labelled `{result['metadata']['token_counter']}` estimator; the Scene API token
and latency figures above are actual provider usage.

## Reference boundary

The historical final clearance result agrees with the Qwen3-based route on
{ref['agreement_count']:,}/{ref['sample_count']:,} cases ({_pct(ref['agreement_rate'])}).
This measures agreement with an earlier model-generated artefact. It is not accuracy against
human annotations. The old LoRA output and final combined memory must therefore be described
as pseudo-references in a report or interview.

{historical_section}

The experiment establishes the number of Semantic calls and serialized context avoided by
the retrieval policy. It does not yet establish end-to-end Semantic-model latency savings,
because skipped Semantic generations were intentionally not issued and billed.
"""


def run_experiment(
    *, calls_file: Path, combined_file: Path, output_dir: Path,
    tokenizer: str | None = None, bootstrap_samples: int = 1000,
    historical_route_summary: Path | None = None,
) -> dict[str, Any]:
    calls = _load_latest(calls_file)
    combined = _load_combined(combined_file)
    counter = TokenCounter(tokenizer)
    rows: list[dict[str, Any]] = []
    pass_count = stop_count = 0
    eager_retrieved = progressive_retrieved = 0
    eager_planning = progressive_planning = 0
    reference_samples = reference_agreements = 0

    for rps_id, call in sorted(calls.items()):
        if not call.get("success") or not isinstance(call.get("prediction"), dict):
            continue
        source = combined.get(rps_id)
        if source is None:
            continue
        scene = call["prediction"]
        semantic = source.get("memory_node", {}).get("semantic_memory", {})
        allows_semantic = _allows_semantic(scene)
        pass_count += int(allows_semantic)
        stop_count += int(not allows_semantic)

        scene_payload = {"RPS_id": rps_id, "scene_memory": scene}
        full_payload = {**scene_payload, "semantic_memory": semantic}
        scene_tokens = counter.count(_json_text(scene_payload))
        full_tokens = counter.count(_json_text(full_payload))
        eager_retrieved += full_tokens
        progressive_tokens = full_tokens if allows_semantic else scene_tokens
        progressive_retrieved += progressive_tokens
        eager_planning += full_tokens
        progressive_planning += full_tokens if allows_semantic else 0

        reference = _reference_allows_semantic(source)
        if reference is not None:
            reference_samples += 1
            reference_agreements += int(reference == allows_semantic)
        rows.append({
            "rps_id": rps_id,
            "street_name": str(call.get("street_name") or source.get("street_name") or "unknown"),
            "scene_allows_semantic": allows_semantic,
            "historical_final_clearance": reference,
            "eager_retrieved_tokens": full_tokens,
            "progressive_retrieved_tokens": progressive_tokens,
            "eager_planning_tokens": full_tokens,
            "progressive_planning_tokens": full_tokens if allows_semantic else 0,
        })

    if not rows:
        raise ValueError("No successful Scene API results can be paired with the combined file")
    scene_summary_path = calls_file.parent / "summary.json"
    scene_summary = json.loads(scene_summary_path.read_text(encoding="utf-8"))["metrics"]
    result = {
        "metadata": {
            "calls_file": str(calls_file.resolve()),
            "combined_file": str(combined_file.resolve()),
            "token_counter": counter.name,
            "token_count_mode": counter.mode,
            "bootstrap_samples": bootstrap_samples,
        },
        "scene_api": scene_summary,
        "route_ablation": {
            "evaluated_rps_count": len(rows),
            "scene_pass_count": pass_count,
            "scene_stop_count": stop_count,
            "eager_semantic_request_count": len(rows),
            "progressive_semantic_request_count": pass_count,
            "semantic_request_reduction": stop_count / len(rows),
            "eager_retrieved_context_tokens": eager_retrieved,
            "progressive_retrieved_context_tokens": progressive_retrieved,
            "retrieved_context_token_reduction": 1.0 - progressive_retrieved / eager_retrieved,
            "retrieved_context_token_reduction_ci95": _bootstrap_reduction(
                rows, samples=bootstrap_samples
            ),
            "eager_planning_context_tokens": eager_planning,
            "progressive_planning_context_tokens": progressive_planning,
            "planning_context_token_reduction": 1.0 - progressive_planning / eager_planning,
            "historical_final_clearance_reference": {
                "sample_count": reference_samples,
                "agreement_count": reference_agreements,
                "agreement_rate": reference_agreements / reference_samples if reference_samples else 0.0,
                "label_type": "model-generated pseudo-reference",
            },
        },
    }
    if historical_route_summary:
        result["historical_lora_route_ablation"] = json.loads(
            historical_route_summary.read_text(encoding="utf-8")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rps_metrics.jsonl").write_text(
        "".join(_json_text(row) + "\n" for row in rows), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_report(result), encoding="utf-8")
    return result


def run_from_args(args: Any) -> dict[str, Any]:
    return run_experiment(
        calls_file=Path(args.calls_file),
        combined_file=Path(args.combined_file),
        output_dir=Path(args.output),
        tokenizer=args.tokenizer,
        bootstrap_samples=args.bootstrap_samples,
        historical_route_summary=(
            Path(args.historical_route_summary) if args.historical_route_summary else None
        ),
    )
