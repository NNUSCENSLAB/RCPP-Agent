#!/usr/bin/env python3
"""Evaluate scene/semantic memory predictions and decompose experiment gains."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcpp_core.memory_schema import validate_scene_memory, validate_semantic_memory


def verify_test_lock(path: Path) -> None:
    lock_path = path.resolve().parent / "test_lock.json"
    if not lock_path.exists():
        raise ValueError(f"Locked-test manifest is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    if value != lock.get("sha256"):
        raise ValueError("Test dataset hash differs from test_lock.json")


def read_records(path: Path) -> list[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else list(value.values())


def payload(record: Mapping[str, Any], kind: str) -> Optional[Dict[str, Any]]:
    messages = record.get("messages") or record.get("conversations") or []
    for message in reversed(messages):
        role = message.get("role") or message.get("from")
        if role in {"assistant", "gpt"}:
            raw = message.get("content", message.get("value"))
            try:
                value = json.loads(raw) if isinstance(raw, str) else raw
                return value if isinstance(value, dict) and "error" not in value else None
            except json.JSONDecodeError:
                return None
    memory_node = record.get("memory_node") or {}
    value = memory_node.get(f"{kind}_memory")
    return value if isinstance(value, dict) else None


def record_id(record: Mapping[str, Any], index: int, kind: str) -> str:
    metadata = record.get("metadata") or {}
    if metadata.get("source_id"):
        return str(metadata["source_id"])
    value = payload(record, kind) or {}
    if value.get("rps_id"):
        return str(value["rps_id"])
    if record.get("image"):
        return Path(str(record["image"])).name
    images = record.get("images") or []
    return Path(str(images[0])).name if images else str(index)


def f1_binary(gold: list[bool], pred: list[bool]) -> float:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    if not labels:
        return 1.0
    scores = []
    for label in labels:
        scores.append(f1_binary([x == label for x in gold], [x == label for x in pred]))
    return statistics.mean(scores)


def canonical_semantic(memory: Mapping[str, Any], field: str) -> str:
    facts = memory.get("rule_facts") or {}
    labels = facts.get("labels") or {}
    label_key = {
        "functional_zone_type": "functional_zone",
        "commuting_flow": "commuting_flow",
        "sensitive_constraints": "sensitive_risk",
        "grid_accessibility": "grid_accessibility",
    }[field]
    if labels.get(label_key):
        return str(labels[label_key])
    normalized = str(memory.get(field, "")).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        ("sensitive_constraints", "no_risk"): "none",
        ("sensitive_constraints", "risk_warning"): "high",
        ("grid_accessibility", "general"): "moderate",
    }
    for (alias_field, token), canonical in aliases.items():
        if field == alias_field and token in normalized:
            return canonical
    candidates = {
        "functional_zone_type": ("residential_oriented", "commercial_oriented", "mixed_functional", "low_density"),
        "commuting_flow": ("high", "medium", "low"),
        "sensitive_constraints": ("high", "low", "none"),
        "grid_accessibility": ("excellent", "good", "moderate", "poor", "very_poor", "unknown"),
    }[field]
    return next((name for name in candidates if name in normalized), "invalid")


def per_sample(kind: str, gold: Mapping[str, Any], pred: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    if pred is None:
        return {"primary": 0.0, "json_valid": 0.0, "conflict_free": 0.0}
    runtime = ((pred.get("_meta") or {}).get("runtime") or {})
    runtime_metrics = {}
    if runtime.get("latency_ms") is not None:
        runtime_metrics["latency_ms"] = float(runtime["latency_ms"])
    if runtime.get("peak_gpu_memory_mb") is not None:
        runtime_metrics["peak_gpu_memory_mb"] = float(runtime["peak_gpu_memory_mb"])
    if kind == "scene":
        valid = not validate_scene_memory(pred)
        obstacle_ok = str(gold.get("ground_obstacle_types")) == str(pred.get("ground_obstacle_types"))
        deploy_ok = bool(gold.get("clearance_visual_assessment")) == bool(pred.get("clearance_visual_assessment"))
        count_error = abs(float(gold.get("ground_obstacle_count", 0)) - float(pred.get("ground_obstacle_count", 0)))
        hallucination = float(not gold.get("has_ground_obstacle") and bool(pred.get("has_ground_obstacle")))
        return {
            "primary": statistics.mean((float(obstacle_ok), float(deploy_ok))),
            "obstacle_correct": float(obstacle_ok),
            "deploy_correct": float(deploy_ok),
            "obstacle_count_abs_error": count_error,
            "json_valid": float(valid),
            "conflict_free": float(valid),
            "visual_hallucination": hallucination,
            **runtime_metrics,
        }
    fields = ("functional_zone_type", "commuting_flow", "sensitive_constraints", "grid_accessibility")
    matches = [canonical_semantic(gold, field) == canonical_semantic(pred, field) for field in fields]
    valid = not validate_semantic_memory(pred)
    gold_facts = gold.get("rule_facts") or {}
    pred_facts = pred.get("rule_facts") or {}
    facts_match = bool(gold_facts) and pred_facts == gold_facts
    gold_evidence = (gold_facts.get("evidence") or {}) if isinstance(gold_facts, dict) else {}
    pred_evidence = (pred_facts.get("evidence") or {}) if isinstance(pred_facts, dict) else {}
    numeric_keys = [key for key, value in gold_evidence.items() if isinstance(value, (int, float))]
    numeric_copy = all(pred_evidence.get(key) == gold_evidence[key] for key in numeric_keys)
    cited = set(pred.get("evidence_fields") or [])
    coverage = len(cited & set(gold_evidence)) / len(gold_evidence) if gold_evidence else 1.0
    return {
        "primary": statistics.mean(map(float, matches)),
        "field_accuracy": statistics.mean(map(float, matches)),
        "fact_consistency": float(facts_match),
        "numeric_copy_accuracy": float(numeric_copy),
        "evidence_coverage": coverage,
        "json_valid": float(valid),
        "conflict_free": float(valid and facts_match),
        **runtime_metrics,
    }


def bootstrap_delta(a: list[float], b: list[float], seed: int = 42, rounds: int = 2000) -> Dict[str, float]:
    if len(a) != len(b) or not a:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    deltas = []
    for _ in range(rounds):
        indexes = [rng.randrange(len(a)) for _ in a]
        deltas.append(statistics.mean(b[i] - a[i] for i in indexes))
    deltas.sort()
    return {
        "delta": statistics.mean(b) - statistics.mean(a),
        "ci_low": deltas[int(rounds * 0.025)],
        "ci_high": deltas[min(rounds - 1, int(rounds * 0.975))],
    }


def dataset_primary(
    kind: str,
    gold: list[Mapping[str, Any]],
    pred: list[Optional[Mapping[str, Any]]],
) -> float:
    if kind == "scene":
        obstacle = macro_f1(
            [str(value.get("ground_obstacle_types", "invalid")) for value in gold],
            [str((value or {}).get("ground_obstacle_types", "invalid")) for value in pred],
        )
        deploy = f1_binary(
            [bool(value.get("clearance_visual_assessment")) for value in gold],
            [bool((value or {}).get("clearance_visual_assessment")) for value in pred],
        )
        return statistics.mean((obstacle, deploy))
    fields = ("functional_zone_type", "commuting_flow", "sensitive_constraints", "grid_accessibility")
    return statistics.mean(
        macro_f1(
            [canonical_semantic(value, field) for value in gold],
            [canonical_semantic(value or {}, field) for value in pred],
        )
        for field in fields
    )


def bootstrap_primary_delta(
    kind: str,
    gold: list[Mapping[str, Any]],
    left: list[Optional[Mapping[str, Any]]],
    right: list[Optional[Mapping[str, Any]]],
    seed: int = 42,
    rounds: int = 2000,
) -> Dict[str, float]:
    if not gold:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    deltas = []
    for _ in range(rounds):
        indexes = [rng.randrange(len(gold)) for _ in gold]
        sampled_gold = [gold[i] for i in indexes]
        deltas.append(
            dataset_primary(kind, sampled_gold, [right[i] for i in indexes])
            - dataset_primary(kind, sampled_gold, [left[i] for i in indexes])
        )
    deltas.sort()
    return {
        "delta": dataset_primary(kind, gold, right) - dataset_primary(kind, gold, left),
        "ci_low": deltas[int(rounds * 0.025)],
        "ci_high": deltas[min(rounds - 1, int(rounds * 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("scene", "semantic"))
    parser.add_argument("gold", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("predictions", nargs="+", help="variant=path")
    parser.add_argument("--allow-unlocked", action="store_true")
    args = parser.parse_args()

    if not args.allow_unlocked:
        verify_test_lock(args.gold)
    gold_records = read_records(args.gold)
    gold = {
        record_id(record, index, args.kind): payload(record, args.kind)
        for index, record in enumerate(gold_records)
    }
    variants: Dict[str, Dict[str, Dict[str, float]]] = {}
    prediction_vectors: Dict[str, list[Optional[Dict[str, Any]]]] = {}
    gold_vector = [value or {} for value in gold.values()]
    for spec in args.predictions:
        name, raw_path = spec.split("=", 1)
        records = read_records(Path(raw_path))
        predictions = {
            record_id(record, index, args.kind): payload(record, args.kind)
            for index, record in enumerate(records)
        }
        samples = [per_sample(args.kind, gold_value or {}, predictions.get(key)) for key, gold_value in gold.items()]
        metric_names = sorted({metric for sample in samples for metric in sample})
        summary = {}
        for metric in metric_names:
            values = [sample[metric] for sample in samples if metric in sample]
            summary[metric] = {"mean": statistics.mean(values), "count": len(values)}
        summary["inference_failure_rate"] = {
            "mean": 1.0 - summary["json_valid"]["mean"],
            "count": len(samples),
        }
        if args.kind == "scene":
            gold_obstacles = [str((value or {}).get("ground_obstacle_types", "invalid")) for value in gold.values()]
            pred_obstacles = [
                str((predictions.get(key) or {}).get("ground_obstacle_types", "invalid")) for key in gold
            ]
            gold_deploy = [bool((value or {}).get("clearance_visual_assessment")) for value in gold.values()]
            pred_deploy = [
                bool((predictions.get(key) or {}).get("clearance_visual_assessment")) for key in gold
            ]
            summary["obstacle_macro_f1"] = {
                "mean": macro_f1(gold_obstacles, pred_obstacles),
                "count": len(samples),
            }
            summary["deployment_f1"] = {
                "mean": f1_binary(gold_deploy, pred_deploy),
                "count": len(samples),
            }
            summary["primary"]["mean"] = statistics.mean(
                (summary["obstacle_macro_f1"]["mean"], summary["deployment_f1"]["mean"])
            )
        else:
            field_scores = []
            for field in (
                "functional_zone_type",
                "commuting_flow",
                "sensitive_constraints",
                "grid_accessibility",
            ):
                gold_labels = [canonical_semantic(value or {}, field) for value in gold.values()]
                pred_labels = [canonical_semantic(predictions.get(key) or {}, field) for key in gold]
                score = macro_f1(gold_labels, pred_labels)
                summary[f"{field}_macro_f1"] = {"mean": score, "count": len(samples)}
                field_scores.append(score)
            summary["field_macro_f1"] = {"mean": statistics.mean(field_scores), "count": len(samples)}
            summary["primary"]["mean"] = summary["field_macro_f1"]["mean"]
        variants[name] = summary
        prediction_vectors[name] = [predictions.get(key) for key in gold]

    comparisons = {}
    pairs = {
        "model_generation_gain": ("old_base", "new_base"),
        "old_finetune_gain": ("old_base", "old_lora"),
        "new_qlora_gain": ("new_base", "new_qlora"),
        "final_system_gain": ("old_lora", "new_qlora"),
    }
    for label, (left, right) in pairs.items():
        if left in prediction_vectors and right in prediction_vectors:
            comparisons[label] = bootstrap_primary_delta(
                args.kind,
                gold_vector,
                prediction_vectors[left],
                prediction_vectors[right],
            )

    report = {"kind": args.kind, "sample_count": len(gold), "variants": variants, "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
