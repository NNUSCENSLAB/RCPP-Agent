# -*- coding: utf-8 -*-
"""Run base VLM Scene-Memory extraction through an OpenAI-compatible API."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import random
import statistics
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from configs.config import get_provider_api_key
from evaluation.evaluate_memory import f1_binary, macro_f1
from rcpp_core.memory_schema import validate_scene_memory
from rcpp_core.prompts import SCENE_SYSTEM_PROMPT


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
USER_PROMPT = (
    "Assess the physical environment of the RED-BOXED roadside parking stall "
    "and its lateral zone according to the system prompt. Return JSON only."
)


@dataclass(frozen=True)
class SceneCase:
    rps_id: str
    street_name: str
    image_path: str
    image_name: str
    final_scene_reference: dict[str, Any]
    semantic_memory: dict[str, Any]
    old_lora_reference: dict[str, Any] | None


@dataclass
class SceneCallResult:
    cache_key: str
    rps_id: str
    street_name: str
    image: str
    model: str
    success: bool
    prediction: dict[str, Any] | None
    validation_errors: list[str]
    raw_response: str
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    trace_ids: list[str]
    error: str = ""


def _extract_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string and char == "{":
            depth += 1
        elif not in_string and char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:index + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _assistant_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    for message in reversed(record.get("conversations") or record.get("messages") or []):
        if (message.get("from") or message.get("role")) not in {"gpt", "assistant"}:
            continue
        raw = message.get("value", message.get("content"))
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
            return value if isinstance(value, dict) and "error" not in value else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def load_cases(
    *, image_dir: str | Path, combined_file: str | Path,
    old_lora_file: str | Path | None = None,
) -> tuple[list[SceneCase], dict[str, Any]]:
    image_root = Path(image_dir)
    local_images = {path.name: path for path in image_root.glob("*.jpg")}
    combined = json.loads(Path(combined_file).read_text(encoding="utf-8"))
    old_by_name: dict[str, dict[str, Any]] = {}
    if old_lora_file:
        for record in json.loads(Path(old_lora_file).read_text(encoding="utf-8")):
            name = Path(str(record.get("image") or "")).name
            payload = _assistant_payload(record)
            if name and payload:
                old_by_name[name] = payload

    cases: list[SceneCase] = []
    missing: list[str] = []
    for row in combined:
        refs = row.get("street_view_images") or []
        if not refs:
            continue
        name = Path(str(refs[0])).name
        image = local_images.get(name)
        if image is None:
            missing.append(name)
            continue
        node = row.get("memory_node") or {}
        cases.append(SceneCase(
            rps_id=str(row.get("RPS_id") or row.get("rps_id") or name),
            street_name=str(row.get("street_name") or "unknown"),
            image_path=str(image.resolve()),
            image_name=name,
            final_scene_reference=dict(node.get("scene_memory") or {}),
            semantic_memory=dict(node.get("semantic_memory") or {}),
            old_lora_reference=old_by_name.get(name),
        ))
    audit = {
        "combined_rps_count": len(combined),
        "local_image_count": len(local_images),
        "runnable_case_count": len(cases),
        "missing_image_count": len(missing),
        "old_lora_paired_count": sum(case.old_lora_reference is not None for case in cases),
        "missing_image_examples": missing[:20],
    }
    return cases, audit


def select_cases(cases: Sequence[SceneCase], max_samples: int, seed: int) -> list[SceneCase]:
    ordered = sorted(cases, key=lambda case: case.rps_id)
    if max_samples == 0 or max_samples >= len(ordered):
        return ordered
    if max_samples < 1:
        raise ValueError("--max-samples must be 0 (all) or a positive integer")
    return sorted(random.Random(seed).sample(ordered, max_samples), key=lambda case: case.rps_id)


class SiliconFlowVisionClient:
    def __init__(
        self, *, api_key: str, model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL, detail: str = "high",
        timeout: float = 120.0, transport_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("SILICONFLOW_API_KEY is not configured")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.detail = detail
        self.timeout = timeout
        self.transport_retries = transport_retries

    @staticmethod
    def _trace_id(headers: Mapping[str, Any]) -> str:
        for key, value in headers.items():
            if key.lower() == "x-siliconcloud-trace-id":
                return str(value)
        return ""

    def _post(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        trace_id = uuid.uuid4().hex
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "RCPP-Agent/1.1 scene-api-experiment",
                "X-Trace-Id": trace_id,
            },
            method="POST",
        )
        last_error = ""
        for attempt in range(self.transport_retries + 1):
            retry_delay = min(30, 2 ** attempt + random.random())
            try:
                with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    body = json.loads(response.read().decode("utf-8"))
                    return body, self._trace_id(dict(response.headers.items())) or trace_id
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in {429, 500, 502, 503, 504}:
                    break
                if exc.code == 429:
                    retry_delay = min(60, 15 * (attempt + 1) + random.random())
            except (URLError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.transport_retries:
                time.sleep(retry_delay)
        raise RuntimeError(last_error or "provider request failed")

    def infer(self, case: SceneCase) -> SceneCallResult:
        image_bytes = Path(case.image_path).read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        cache_key = hashlib.sha256(
            f"{self.model}|{self.detail}|{SCENE_SYSTEM_PROMPT}|{image_hash}".encode("utf-8")
        ).hexdigest()
        mime = mimetypes.guess_type(case.image_path)[0] or "image/jpeg"
        data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SCENE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": self.detail}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        trace_ids: list[str] = []
        request_count = 0
        started = time.perf_counter()
        raw = ""
        errors: list[str] = []
        try:
            for schema_attempt in range(2):
                body, trace_id = self._post({
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 512,
                })
                request_count += 1
                trace_ids.append(trace_id)
                usage = body.get("usage") or {}
                for key in totals:
                    totals[key] += int(usage.get(key) or 0)
                raw = str((((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""))
                prediction = _extract_json(raw)
                if prediction is not None:
                    count = prediction.get("ground_obstacle_count")
                    if (
                        isinstance(count, int)
                        and not isinstance(count, bool)
                        and isinstance(prediction.get("has_existing_RCP"), bool)
                        and isinstance(prediction.get("is_functional_zone"), bool)
                    ):
                        derived_clearance = (
                            not prediction["has_existing_RCP"]
                            and not prediction["is_functional_zone"]
                            and count < 3
                        )
                        model_clearance = prediction.get("clearance_visual_assessment")
                        if model_clearance != derived_clearance:
                            prediction["_model_clearance_visual_assessment"] = model_clearance
                            prediction["clearance_visual_assessment"] = derived_clearance
                            prediction["_derived_fields"] = ["clearance_visual_assessment"]
                errors = validate_scene_memory(prediction) if prediction is not None else ["invalid JSON"]
                if not errors:
                    return SceneCallResult(
                        cache_key, case.rps_id, case.street_name, case.image_path,
                        self.model, True, prediction, [], raw, request_count,
                        totals["prompt_tokens"], totals["completion_tokens"], totals["total_tokens"],
                        (time.perf_counter() - started) * 1000, trace_ids,
                    )
                if schema_attempt == 0:
                    messages.extend([
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": "Return corrected JSON only. Errors: " + "; ".join(errors)},
                    ])
            return SceneCallResult(
                cache_key, case.rps_id, case.street_name, case.image_path,
                self.model, False, prediction, errors, raw, request_count,
                totals["prompt_tokens"], totals["completion_tokens"], totals["total_tokens"],
                (time.perf_counter() - started) * 1000, trace_ids,
                "output failed schema validation",
            )
        except Exception as exc:
            return SceneCallResult(
                cache_key, case.rps_id, case.street_name, case.image_path,
                self.model, False, None, errors or ["provider failure"], raw, request_count,
                totals["prompt_tokens"], totals["completion_tokens"], totals["total_tokens"],
                (time.perf_counter() - started) * 1000, trace_ids,
                f"{type(exc).__name__}: {exc}",
            )


def _read_results(path: Path) -> dict[str, SceneCallResult]:
    results: dict[str, SceneCallResult] = {}
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            results[str(row["rps_id"])] = SceneCallResult(**row)
    return results


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def summarize(cases: Sequence[SceneCase], results: Mapping[str, SceneCallResult]) -> dict[str, Any]:
    available = [results[case.rps_id] for case in cases if case.rps_id in results]
    successes = [row for row in available if row.success and row.prediction is not None]
    paired = [case for case in cases if case.old_lora_reference and case.rps_id in results]
    paired_success = [case for case in paired if results[case.rps_id].success]
    final_success = [case for case in cases if case.rps_id in results and results[case.rps_id].success]

    def percentile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(round((len(ordered) - 1) * q), len(ordered) - 1)]

    full_agreement: dict[str, Any] = {"sample_count": len(paired_success)}
    if paired_success:
        gold = [case.old_lora_reference or {} for case in paired_success]
        pred = [results[case.rps_id].prediction or {} for case in paired_success]
        bool_fields = (
            "has_existing_RCP", "is_functional_zone", "has_ground_obstacle",
            "clearance_visual_assessment",
        )
        full_agreement["boolean_field_accuracy"] = {
            field: statistics.mean(_bool(a.get(field)) == _bool(b.get(field)) for a, b in zip(gold, pred))
            for field in bool_fields
        }
        full_agreement["deployment_f1"] = f1_binary(
            [_bool(x.get("clearance_visual_assessment")) for x in gold],
            [_bool(x.get("clearance_visual_assessment")) for x in pred],
        )
        full_agreement["raw_model_deployment_f1"] = f1_binary(
            [_bool(x.get("clearance_visual_assessment")) for x in gold],
            [
                _bool(x.get("_model_clearance_visual_assessment", x.get("clearance_visual_assessment")))
                for x in pred
            ],
        )
        full_agreement["obstacle_type_macro_f1"] = macro_f1(
            [str(x.get("ground_obstacle_types", "invalid")) for x in gold],
            [str(x.get("ground_obstacle_types", "invalid")) for x in pred],
        )
        full_agreement["obstacle_count_mae"] = statistics.mean(
            abs(int(a.get("ground_obstacle_count", 0)) - int(b.get("ground_obstacle_count", 0)))
            for a, b in zip(gold, pred)
        )

    final_clearance = {}
    if final_success:
        final_clearance = {
            "sample_count": len(final_success),
            "accuracy": statistics.mean(
                _bool(case.final_scene_reference.get("clearance_visual_assessment"))
                == _bool((results[case.rps_id].prediction or {}).get("clearance_visual_assessment"))
                for case in final_success
            ),
            "f1": f1_binary(
                [_bool(case.final_scene_reference.get("clearance_visual_assessment")) for case in final_success],
                [_bool((results[case.rps_id].prediction or {}).get("clearance_visual_assessment")) for case in final_success],
            ),
        }

    latencies = [row.latency_ms for row in successes]
    return {
        "selected_case_count": len(cases),
        "completed_case_count": len(available),
        "success_count": len(successes),
        "failure_count": len(available) - len(successes),
        "json_schema_success_rate": len(successes) / len(available) if available else 0.0,
        "request_count": sum(row.request_count for row in available),
        "prompt_tokens": sum(row.prompt_tokens for row in available),
        "completion_tokens": sum(row.completion_tokens for row in available),
        "total_tokens": sum(row.total_tokens for row in available),
        "derived_clearance_count": sum(
            "clearance_visual_assessment" in ((row.prediction or {}).get("_derived_fields") or [])
            for row in successes
        ),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "cumulative_latency_ms": sum(latencies),
        "historical_lora_agreement": full_agreement,
        "final_clearance_agreement": final_clearance,
    }


def run_experiment(
    *, image_dir: str, combined_file: str, old_lora_file: str | None,
    output_dir: Path, model: str = DEFAULT_MODEL, detail: str = "high",
    max_samples: int = 20, seed: int = 42, workers: int = 2,
    execute: bool = False, retry_failures: bool = False, progress_every: int = 10,
) -> dict[str, Any]:
    cases_all, audit = load_cases(
        image_dir=image_dir, combined_file=combined_file, old_lora_file=old_lora_file
    )
    cases = select_cases(cases_all, max_samples, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "scene_calls.jsonl"
    existing = _read_results(result_path)
    pending = [
        case for case in cases
        if case.rps_id not in existing or (retry_failures and not existing[case.rps_id].success)
    ]
    plan = {
        "provider": "siliconflow", "model": model, "detail": detail,
        "seed": seed, "workers": workers, "execute": execute,
        "audit": audit, "selected_case_count": len(cases),
        "cached_case_count": len(cases) - len(pending), "pending_case_count": len(pending),
        "reference_notice": "Old LoRA and final Scene outputs are historical pseudo-references, not human gold.",
    }
    (output_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    if not execute:
        return {"plan": plan}

    client = SiliconFlowVisionClient(
        api_key=get_provider_api_key("siliconflow") or "", model=model,
        base_url=os.getenv("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL), detail=detail,
    )
    completed = 0
    lock = threading.Lock()
    with result_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(client.infer, case): case for case in pending}
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    handle.flush()
                existing[result.rps_id] = result
                completed += 1
                if (
                    completed == 1 or completed == len(pending)
                    or completed % max(1, progress_every) == 0 or not result.success
                ):
                    print(
                        f"[{completed}/{len(pending)}] {result.rps_id}: "
                        f"{'ok' if result.success else 'failed'}; {result.total_tokens} tokens",
                        flush=True,
                    )
    summary = {"plan": plan, "metrics": summarize(cases, existing)}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_from_args(args: Any) -> dict[str, Any]:
    return run_experiment(
        image_dir=args.image_dir,
        combined_file=args.combined_file,
        old_lora_file=args.old_lora_file,
        output_dir=Path(args.output),
        model=args.model or DEFAULT_MODEL,
        detail=args.image_detail,
        max_samples=args.max_samples,
        seed=args.seed,
        workers=args.workers,
        execute=args.execute,
        retry_failures=args.retry_failures,
        progress_every=args.progress_every,
    )
