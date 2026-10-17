# -*- coding: utf-8 -*-
"""Targeted refresh workers used by Scene→Semantic progressive retrieval."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping, Sequence

from rcpp_core.semantic_rules import extract_statistical_summary
from tools.memory_construction_tool.merge_memory_tool import parse_scene_memory_from_conversations, parse_semantic_memory
from tools.rcpp_paths import perception_artifact_dir


class TargetedMemoryRefresher:
    """Run production extractors only for requested RPS records.

    Scene inference is completed and validated before the caller decides which IDs
    may enter ``refresh_semantic``. Shadow files produced by MemoryInferenceTool are
    never read by this class.
    """

    def __init__(self, *, scene_base_model: str | None = None,
                 scene_adapter_path: str | None = None,
                 semantic_base_model: str | None = None,
                 semantic_adapter_path: str | None = None):
        self._inference = None
        self._prepared: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.scene_base_model = scene_base_model
        self.scene_adapter_path = scene_adapter_path
        self.semantic_base_model = semantic_base_model
        self.semantic_adapter_path = semantic_adapter_path

    @property
    def inference(self):
        if self._inference is None:
            from tools.memory_construction_tool.memory_inference_tool import MemoryInferenceTool
            self._inference = MemoryInferenceTool()
        return self._inference

    @staticmethod
    def resolve_image_sources(context: Mapping[str, Any]) -> dict[str, str]:
        """Map historical image basenames to the local files actually used for inference."""
        db_path = Path(str(context.get("db_path") or ""))
        if not db_path.is_file():
            return {}
        table = str((context.get("tables") or {}).get("bsv_image", "bsv_image")).replace('"', '""')
        try:
            with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
                rows = conn.execute(f'SELECT image_path FROM "{table}"').fetchall()
        except sqlite3.Error:
            return {}  # The actual refresh records a schema/data failure for human review.
        return {Path(str(path).replace("\\", "/")).name: str(path) for (path,) in rows if path}

    def _samples(self, kind: str, area: str, context: Mapping[str, Any], db_path: Path,
                 input_path: Path) -> list[dict[str, Any]]:
        cache_key = (kind, str(context.get("input_revision") or ""))
        if cache_key not in self._prepared:
            if kind == "scene":
                from tools.memory_construction_tool.scene_inference_input_tool import generate_scene_inference_from_spatialite
                generate = generate_scene_inference_from_spatialite
            else:
                from tools.memory_construction_tool.semantic_inference_input_tool import generate_semantic_inference_from_spatialite
                generate = generate_semantic_inference_from_spatialite
            all_input = input_path.with_name(f"{kind}_refresh_all.json")
            generate(db_path, all_input, area, context.get("tables") or {})
            self._prepared[cache_key] = json.loads(all_input.read_text(encoding="utf-8"))
        return self._prepared[cache_key]

    @staticmethod
    def _paths(area: str, context: Mapping[str, Any], kind: str) -> tuple[Path, Path]:
        if context.get("run_id"):
            from rcpp_core.execution_journal import run_directory
            directory = run_directory(str(context["run_id"])) / "inputs" / area
            directory.mkdir(parents=True, exist_ok=True)
            return directory / f"{kind}_refresh_input.json", directory / f"{kind}_refresh_output.json"
        directory = perception_artifact_dir(
            context.get("workspace_dir"), area=area,
            scenario=str(context.get("scenario") or "balance_oriented"),
            time_horizon=context.get("time_horizon"),
        )
        return directory / f"{kind}_refresh_input.json", directory / f"{kind}_refresh_output.json"

    def refresh_scene(self, area: str, rps_ids: list[str], context: Mapping[str, Any],
                      *, rps_records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        db_path = Path(str(context.get("db_path") or ""))
        if not db_path.is_file():
            raise FileNotFoundError(f"SpatiaLite database file not found: {db_path}")
        source_to_rps = {
            Path(str(image).replace("\\", "/")).name: str(row["rps_id"])
            for row in rps_records if str(row.get("rps_id")) in rps_ids
            for image in (row.get("source_images") or [])
        }
        if not source_to_rps:
            return {}
        input_path, output_path = self._paths(area, context, "scene")
        samples = self._samples("scene", area, context, db_path, input_path)
        selected = [x for x in samples if Path(str(x.get("image", "")).replace("\\", "/")).name in source_to_rps]
        input_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        if not selected:
            return {}
        self.inference.run(kind="scene", input_file=str(input_path), output_file=str(output_path),
                           model_key=context.get("model_key"),
                           shadow_enabled=bool(context.get("shadow_enabled", False)),
                           scene_base_model=self.scene_base_model,
                           scene_adapter_path=self.scene_adapter_path)
        output = json.loads(output_path.read_text(encoding="utf-8"))
        refreshed: dict[str, dict[str, Any]] = {}
        for item in output:
            name = Path(str(item.get("image", "")).replace("\\", "/")).name
            rps_id = source_to_rps.get(name)
            if rps_id:
                refreshed[rps_id] = parse_scene_memory_from_conversations(item)
        return refreshed

    def refresh_semantic(self, area: str, rps_ids: list[str], context: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        db_path = Path(str(context.get("db_path") or ""))
        if not db_path.is_file():
            raise FileNotFoundError(f"SpatiaLite database file not found: {db_path}")
        target = set(rps_ids)
        input_path, output_path = self._paths(area, context, "semantic")
        samples = self._samples("semantic", area, context, db_path, input_path)

        def sample_id(item: Mapping[str, Any]) -> str:
            messages = item.get("messages") or []
            if not messages:
                return ""
            text = str(messages[0].get("content") or "")
            try:
                summary = extract_statistical_summary(text)
                if summary.get("rps_id"):
                    return str(summary["rps_id"])
            except ValueError:
                pass
            match = re.search(r"RPS ID:\s*([^\s]+)", text)
            return match.group(1) if match else ""

        selected = [item for item in samples if sample_id(item) in target]
        input_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        if not selected:
            return {}
        from rcpp_core.semantic_rules import deterministic_semantic_memory
        def rules_only():
            result = {}
            for item in selected:
                text = str(item["messages"][0]["content"])
                summary = extract_statistical_summary(text)
                result[sample_id(item)] = {**deterministic_semantic_memory(summary),
                                          "_meta": {"base_model": "deterministic_rules",
                                                    "adapter_version": "none",
                                                    "evidence_source": "deterministic_rules"}}
            return result
        if context.get("rules_only"):
            return rules_only()
        try:
            self.inference.run(kind="semantic", input_file=str(input_path), output_file=str(output_path),
                               model_key=context.get("model_key"),
                               shadow_enabled=bool(context.get("shadow_enabled", False)),
                               semantic_base_model=self.semantic_base_model,
                               semantic_adapter_path=self.semantic_adapter_path)
        except Exception:
            return rules_only()
        output = json.loads(output_path.read_text(encoding="utf-8"))
        refreshed: dict[str, dict[str, Any]] = {}
        for item in output:
            parsed = parse_semantic_memory(item.get("messages") or [])
            refreshed[str(parsed["rps_id"])] = {"rps_id": str(parsed["rps_id"]), **parsed["semantic_memory"]}
        return refreshed
