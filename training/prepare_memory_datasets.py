#!/usr/bin/env python3
"""Audit legacy memory datasets and convert them to ms-swift JSONL.

Examples:
  python training/prepare_memory_datasets.py scene old_scene.json artifacts/scene
  python training/prepare_memory_datasets.py semantic old_semantic.json artifacts/semantic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcpp_core.memory_schema import validate_scene_memory, validate_semantic_memory
from rcpp_core.semantic_rules import (
    derive_semantic_facts,
    extract_statistical_summary,
    validate_semantic_against_rules,
)


def read_records(path: Path) -> list[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    raise ValueError("Dataset root must be an array, object, or JSONL records")


def _assistant_payload(messages: list[Dict[str, Any]]) -> Dict[str, Any]:
    for message in reversed(messages):
        role = message.get("role") or message.get("from")
        if role in {"assistant", "gpt"}:
            raw = message.get("content", message.get("value", ""))
            value = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(value, dict):
                return value
    raise ValueError("assistant JSON label is missing")


def _group_id(item: Dict[str, Any], source_id: str, image: str = "") -> str:
    for key in ("group_id", "road_id", "street_name", "rps_id", "RPS_id"):
        if item.get(key) not in (None, ""):
            return f"{key}:{item[key]}"
    if image:
        parts = Path(image).stem.split("_")
        if len(parts) >= 2:
            try:
                return f"grid:{round(float(parts[0]), 3)}:{round(float(parts[1]), 3)}"
            except ValueError:
                pass
        return f"image-parent:{Path(image).parent.name}"
    return f"sample:{source_id}"


def _image_digest(path: Path) -> tuple[str, Optional[str]]:
    exact = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        from PIL import Image

        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((8, 8)).getdata())
        average = sum(pixels) / len(pixels)
        ahash = f"{sum((1 << i) for i, value in enumerate(pixels) if value >= average):016x}"
    except Exception:
        ahash = None
    return exact, ahash


def normalize_scene(
    item: Dict[str, Any], index: int, image_root: Optional[Path]
) -> tuple[Optional[Dict[str, Any]], list[str]]:
    errors: list[str] = []
    image = str(item.get("image") or item.get("image_path") or "")
    conversations = item.get("conversations") or item.get("messages") or []
    if not image:
        errors.append("image path is missing")
    try:
        label = _assistant_payload(conversations)
        errors.extend(validate_scene_memory(label))
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        label = {}
        errors.append(str(exc))
    source_id = str(item.get("source_id") or item.get("id") or f"scene-{index:06d}")
    resolved = Path(image)
    if image_root is not None and not resolved.is_absolute():
        resolved = image_root / resolved
    exact_hash = ahash = None
    if image and resolved.exists():
        exact_hash, ahash = _image_digest(resolved)
    elif image and not image.startswith(("http://", "https://")):
        errors.append(f"image does not exist: {resolved}")
    prompt = "Assess only the red-boxed roadside parking stall and return the required JSON."
    for message in conversations:
        if (message.get("role") or message.get("from")) in {"user", "human"}:
            prompt = str(message.get("content", message.get("value", prompt))).replace("<image>", "").strip()
            break
    record = {
        "messages": [
            {"role": "user", "content": f"<image>{prompt}"},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ],
        "images": [image],
        "metadata": {
            "source_id": source_id,
            "group_id": _group_id(item, source_id, image),
            "label_source": item.get("label_source", "legacy_human_verified"),
            "sha256": exact_hash,
            "perceptual_hash": ahash,
        },
    }
    return (record if not errors else None), errors


def normalize_semantic(item: Dict[str, Any], index: int) -> tuple[Optional[Dict[str, Any]], list[str]]:
    errors: list[str] = []
    messages = item.get("messages") or item.get("conversations") or []
    user = ""
    for message in messages:
        if (message.get("role") or message.get("from")) in {"user", "human"}:
            user = str(message.get("content", message.get("value", "")))
            break
    try:
        summary = extract_statistical_summary(user)
        facts = derive_semantic_facts(summary)
        label = _assistant_payload(messages)
        label["rule_facts"] = facts
        label.setdefault("rps_id", str(summary.get("rps_id") or ""))
        label.setdefault("bsv_image", summary.get("bsv_image"))
        errors.extend(validate_semantic_memory(label))
        errors.extend(validate_semantic_against_rules(label, summary))
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        summary, facts, label = {}, {}, {}
        errors.append(str(exc))
    source_id = str(item.get("source_id") or item.get("id") or f"semantic-{index:06d}")
    group_source = dict(item)
    group_source.setdefault("rps_id", summary.get("rps_id"))
    grounded_user = (
        user.split("【Rule Facts", 1)[0].rstrip()
        + "\n\n【Rule Facts - copy exactly】\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n\nDo not recalculate labels or numbers. /no_think"
    )
    record = {
        "messages": [
            {"role": "user", "content": grounded_user},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ],
        "metadata": {
            "source_id": source_id,
            "group_id": _group_id(group_source, source_id, str(summary.get("bsv_image") or "")),
            "label_source": item.get("label_source", "legacy_human_verified"),
        },
    }
    return (record if not errors else None), errors


def assign_splits(records: list[Dict[str, Any]], seed: int) -> Dict[str, list[Dict[str, Any]]]:
    by_group: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["metadata"]["group_id"]].append(record)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    total = len(records)
    targets = {"train": total * 0.70, "validation": total * 0.15}
    result = {"train": [], "validation": [], "test": []}
    for group in groups:
        if len(result["train"]) < targets["train"]:
            split = "train"
        elif len(result["validation"]) < targets["validation"]:
            split = "validation"
        else:
            split = "test"
        result[split].extend(by_group[group])
    return result


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("scene", "semantic"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = read_records(args.input)
    accepted: list[Dict[str, Any]] = []
    rejected: list[Dict[str, Any]] = []
    hashes: Dict[str, str] = {}
    duplicates = 0
    error_counts: Counter[str] = Counter()
    normalizer = normalize_scene if args.kind == "scene" else normalize_semantic
    for index, item in enumerate(raw):
        if args.kind == "scene":
            record, errors = normalizer(item, index, args.image_root)
        else:
            record, errors = normalizer(item, index)
        if record is not None and args.kind == "scene":
            digest = record["metadata"].get("sha256") or record["metadata"].get("perceptual_hash")
            if digest and digest in hashes:
                record["metadata"]["group_id"] = hashes[digest]
                duplicates += 1
            elif digest:
                hashes[digest] = record["metadata"]["group_id"]
        if errors:
            error_counts.update(errors)
            rejected.append({"index": index, "errors": errors, "record": item})
        elif record is not None:
            accepted.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(accepted, args.seed)
    for name, records in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", records)
    write_jsonl(args.output_dir / "rejected.jsonl", rejected)
    review_rng = random.Random(args.seed)
    review_sample = review_rng.sample(accepted, min(200, len(accepted))) if accepted else []
    write_jsonl(args.output_dir / "manual_review_sample.jsonl", review_sample)
    test_path = args.output_dir / "test.jsonl"
    test_lock = {
        "path": str(test_path.resolve()),
        "sha256": file_sha256(test_path),
        "record_count": len(splits["test"]),
        "seed": args.seed,
    }
    (args.output_dir / "test_lock.json").write_text(
        json.dumps(test_lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pass_rate = len(accepted) / len(raw) if raw else 0.0
    split_counts = {name: len(records) for name, records in splits.items()}
    report = {
        "kind": args.kind,
        "source": str(args.input.resolve()),
        "seed": args.seed,
        "total": len(raw),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "pass_rate": pass_rate,
        "gate_passed": pass_rate >= 0.95 and all(split_counts.values()),
        "duplicates_grouped": duplicates,
        "split_counts": split_counts,
        "top_errors": error_counts.most_common(20),
        "manual_review_sample_size": min(200, len(accepted)),
        "manual_review_approved": False,
        "test_lock": test_lock,
    }
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
