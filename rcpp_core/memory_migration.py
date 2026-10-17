# -*- coding: utf-8 -*-
"""Audited migration from legacy JSON exports into graph memory."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from rcpp_core.graph_memory_repository import GraphMemoryRepository, normalize_scene, stable_id


@dataclass
class MigrationPlan:
    area: str
    rps: list[dict[str, Any]]
    scene: list[dict[str, Any]]
    semantic: list[dict[str, Any]]
    audit: dict[str, Any]


def _load_list(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [{"RPS_id": key, **value} for key, value in data.items() if isinstance(value, dict)]
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError(f"Expected a JSON array/object of records: {path}")
    return data


def _scene_answer(record: Mapping[str, Any]) -> dict[str, Any]:
    for turn in reversed(record.get("conversations") or []):
        if isinstance(turn, Mapping) and str(turn.get("from", "")).lower() in {"gpt", "assistant"}:
            value = turn.get("value")
            if isinstance(value, str):
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
    raise ValueError("Scene record has no valid assistant JSON answer")


def _basename(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name


def _scene_conflicts(scene: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    count = scene.get("ground_obstacle_count")
    types = scene.get("ground_obstacle_types")
    type_count = 0 if types in (None, "none", ["none"], []) else len(types) if isinstance(types, list) else len([x for x in str(types).split(",") if x.strip()])
    if isinstance(count, int) and count != type_count:
        problems.append("obstacle_type_count_mismatch")
    if scene.get("has_ground_obstacle") is False and isinstance(count, int) and count != 0:
        problems.append("obstacle_false_with_nonzero_count")
    if scene.get("has_ground_obstacle") is True and count == 0:
        problems.append("obstacle_true_with_zero_count")
    return problems


def parse_semantic_facts(semantic: Mapping[str, Any], rps_id: str) -> dict[str, Any]:
    out = {"rps_id": rps_id, **dict(semantic)}
    raw = "\n".join(str(v) for v in semantic.values())
    traffic = re.search(r"Total:\s*([\d,.]+).*?Inflow:\s*([\d,.]+).*?Outflow:\s*([\d,.]+)", raw, re.I | re.S)
    if traffic:
        out.update(traffic_total=int(traffic.group(1).replace(",", "")),
                   traffic_inflow=int(traffic.group(2).replace(",", "")),
                   traffic_outflow=int(traffic.group(3).replace(",", "")))
    distance = re.search(r"Dist\s*=\s*([\d.]+)\s*m", raw, re.I)
    if distance:
        out["grid_distance_m"] = float(distance.group(1))
    poi = re.search(r"P_\{res\}\s*=\s*(\d+)\s*,\s*P_\{comm\}\s*=\s*(\d+)", raw)
    if poi:
        out["residential_poi_count"] = int(poi.group(1))
        out["commercial_poi_count"] = int(poi.group(2))
    risk = re.search(r"Risk Warning:\s*([^.]*)", raw, re.I)
    if risk:
        out["risk_level"] = risk.group(1).strip()
    out["raw_text"] = raw
    return out


def prepare_migration(area: str, scene_file: str | Path, combined_file: str | Path) -> MigrationPlan:
    raw_scene = _load_list(scene_file)
    combined = _load_list(combined_file)
    raw_by_image: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    invalid_scene = 0
    for record in raw_scene:
        try:
            raw_by_image[_basename(record.get("image"))] = (record, _scene_answer(record))
        except (ValueError, json.JSONDecodeError):
            invalid_scene += 1

    raw_answers = [answer for _, answer in raw_by_image.values()]
    raw_type_count_mismatches = sum("obstacle_type_count_mismatch" in _scene_conflicts(x) for x in raw_answers)
    raw_false_nonzero = sum("obstacle_false_with_nonzero_count" in _scene_conflicts(x) for x in raw_answers)
    raw_true_zero = sum("obstacle_true_with_zero_count" in _scene_conflicts(x) for x in raw_answers)
    raw_greenbelt_repairs = sum(
        isinstance(x.get("is_functional_zone"), str)
        and x.get("is_functional_zone", "").strip().lower() == "greenbelt"
        for x in raw_answers
    )
    raw_anomalies: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    for image_name, (_, answer) in raw_by_image.items():
        conflicts = _scene_conflicts(answer)
        if conflicts:
            raw_anomalies.append({"image": image_name, "status": "conflicted", "reasons": conflicts})
        if isinstance(answer.get("is_functional_zone"), str) and answer.get("is_functional_zone", "").strip().lower() == "greenbelt":
            repair_records.append({"image": image_name, "field": "is_functional_zone",
                                   "raw_value": answer.get("is_functional_zone"),
                                   "normalized_value": True})

    rps_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    matched_raw_images: set[str] = set()
    detailed = partial = conflicted = repaired = semantic_partial = 0

    for index, record in enumerate(combined, start=1):
        rps_id = str(record.get("RPS_id") or f"{area.upper()}-{index:04d}")
        images = [str(x) for x in (record.get("street_view_images") or [])]
        rps_rows.append({"rps_id": rps_id, "coordinates": record.get("coordinates") or [],
                         "street_name": record.get("street_name") or "", "source_images": images})
        match = next((raw_by_image[_basename(image)] for image in images if _basename(image) in raw_by_image), None)
        if match:
            raw_record, scene = match
            matched_raw_images.add(_basename(raw_record.get("image")))
            original_functional = scene.get("is_functional_zone")
            repair_notes: list[str] = []
            if isinstance(original_functional, str) and original_functional.strip().lower() == "greenbelt":
                scene = dict(scene)
                scene["is_functional_zone"] = True
                scene["functional_zone_type"] = scene.get("functional_zone_type") or "greenbelt"
                repair_notes.append("normalized is_functional_zone='greenbelt' to true")
                repaired += 1
            scene = normalize_scene(scene)
            conflicts = _scene_conflicts(scene)
            status = "conflicted" if conflicts else "active"
            conflicted += int(bool(conflicts))
            detailed += 1
            scene_rows.append({"rps_id": rps_id, "memory": scene, "status": status,
                               "source_images": images or [str(raw_record.get("image") or "")],
                               "raw_value": original_functional, "repair_notes": repair_notes,
                               "conflicts": conflicts})
        else:
            final_scene = dict((record.get("memory_node") or {}).get("scene_memory") or {})
            final_scene = normalize_scene(final_scene)
            scene_rows.append({"rps_id": rps_id, "memory": final_scene, "status": "partial",
                               "source_images": images, "repair_notes": [], "conflicts": []})
            partial += 1

        semantic_raw = dict((record.get("memory_node") or {}).get("semantic_memory") or {})
        parsed = parse_semantic_facts(semantic_raw, rps_id)
        required_parsed = ("traffic_total", "traffic_inflow", "traffic_outflow", "grid_distance_m")
        semantic_status = "active" if all(name in parsed for name in required_parsed) else "partial"
        semantic_partial += int(semantic_status == "partial")
        semantic_rows.append({"rps_id": rps_id, "memory": parsed, "status": semantic_status})

    unmatched = sorted(set(raw_by_image) - matched_raw_images)
    audit = {
        "area": area.lower(), "rps_count": len(rps_rows), "semantic_count": len(semantic_rows),
        "detailed_scene_count": detailed, "partial_scene_count": partial,
        "unmatched_raw_scene_count": len(unmatched), "invalid_raw_scene_count": invalid_scene,
        "conflicted_scene_count": conflicted, "repaired_scene_count": repaired,
        "raw_obstacle_type_count_mismatch_count": raw_type_count_mismatches,
        "raw_obstacle_false_nonzero_count": raw_false_nonzero,
        "raw_obstacle_true_zero_count": raw_true_zero,
        "raw_greenbelt_repair_count": raw_greenbelt_repairs,
        "raw_anomaly_records": raw_anomalies,
        "repair_records": repair_records,
        "partial_semantic_count": semantic_partial,
        "unmatched_raw_scene_images": unmatched,
    }
    return MigrationPlan(area.lower(), rps_rows, scene_rows, semantic_rows, audit)


def apply_migration(plan: MigrationPlan, repository: GraphMemoryRepository) -> dict[str, Any]:
    repository.ensure_schema()
    repository.upsert_rps(plan.area, plan.rps)
    for row in plan.scene:
        memory = row["memory"]
        repository.append_scene_memory(
            plan.area, row["rps_id"], memory,
            memory_id=stable_id("scene", plan.area, row["rps_id"], memory),
            status=row["status"], source_images=row["source_images"],
            evidence_source="legacy_scene_inference",
        )
    for row in plan.semantic:
        memory = row["memory"]
        repository.append_semantic_memory(
            plan.area, row["rps_id"], memory,
            memory_id=stable_id("semantic", plan.area, row["rps_id"], memory),
            status=row["status"], evidence_source="legacy_semantic_inference",
            raw_text=memory.get("raw_text", ""),
        )
    counts = repository.get_area_counts(plan.area)
    expected = len(plan.rps)
    if counts != {"rps": expected, "scene": expected, "semantic": expected}:
        raise RuntimeError(f"Migration verification failed: expected {expected} of each, got {counts}")
    deleted = repository.delete_legacy_aggregate(plan.area)
    return {**plan.audit, "applied": True, "graph_counts": counts,
            "legacy_aggregate_deleted": deleted}
