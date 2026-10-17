"""Scene→Semantic retrieval with durable per-RPS refresh receipts."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import os
from typing import Any, Callable, Dict, Mapping, Optional

from configs.data_config import DEFAULT_WORKSPACE, path_output_environment_perception_agent
from rcpp_core.execution_journal import ExecutionJournal, file_fingerprint, fingerprint, run_directory
from rcpp_core.graph_memory_repository import GraphMemoryRepository, assess_scene, assess_semantic
from rcpp_core.memory_refresh import TargetedMemoryRefresher
from rcpp_core.model_registry import get_deployment
from rcpp_core.routing import MemoryAwareRouter, RoutingContext


class EnvironmentPerceptionAgent:
    def __init__(self, workspace_dir: str = DEFAULT_WORKSPACE,
                 scene_base_model: Optional[str] = None, scene_adapter_path: Optional[str] = None,
                 semantic_base_model: Optional[str] = None, semantic_adapter_path: Optional[str] = None,
                 scene_refresher: Optional[Callable] = None, semantic_refresher: Optional[Callable] = None,
                 router: MemoryAwareRouter | None = None):
        self.workspace_dir = workspace_dir
        self.model_settings = dict(scene_base_model=scene_base_model, scene_adapter_path=scene_adapter_path,
                                   semantic_base_model=semantic_base_model, semantic_adapter_path=semantic_adapter_path)
        self.scene_refresher = scene_refresher
        self.semantic_refresher = semantic_refresher
        self.targeted_refresher = TargetedMemoryRefresher(**self.model_settings)
        if router is None and any(value is not None for value in self.model_settings.values()):
            # Profiles must match the weights actually requested by explicit overrides.
            deployments = {}
            for kind in ("scene", "semantic"):
                deployment = get_deployment(kind,
                    base_model=self.model_settings[f"{kind}_base_model"],
                    adapter_path=self.model_settings[f"{kind}_adapter_path"])
                deployments[deployment.key] = deployment
            self.router = MemoryAwareRouter(registry=deployments)
        else:
            self.router = router or MemoryAwareRouter()

    def perceive(self, spatialite_context: Dict[str, Any], global_instruction: Dict[str, Any],
                 *, repository: GraphMemoryRepository) -> Dict[str, Any]:
        area = str(spatialite_context.get("area") or "").strip().lower()
        if not area:
            raise ValueError("spatialite_context['area'] is required")
        if global_instruction.get("scene_semantic_memory_json_path"):
            raise ValueError("Runtime JSON memory input is disabled; use rcpp memory migrate")
        run_id = str(global_instruction.get("run_id") or "unknown")
        journal = ExecutionJournal(run_directory(run_id) / "execution.sqlite")
        excluded = set(global_instruction.get("excluded_rps") or [])
        rps_rows = [r for r in repository.list_rps(area) if r["rps_id"] not in excluded]
        if not rps_rows and not excluded:
            raise ValueError(f"No graph RPS is available for area={area}; run rcpp memory migrate")
        ids = [r["rps_id"] for r in rps_rows]
        records, events = repository.load_progressive_context(area, ids) if ids else ({}, [])
        # Fingerprints invalidate receipts after changing source data, model configuration or code.
        root = Path(__file__).resolve().parents[1]
        revision = {
            "database": file_fingerprint(str(spatialite_context.get("db_path") or "")),
            "database_wal": file_fingerprint(str(spatialite_context.get("db_path") or "") + "-wal"),
            "context": spatialite_context,
            "code": [file_fingerprint(root / p) for p in (
                "rcpp_core/prompts.py", "rcpp_core/memory_refresh.py", "rcpp_core/semantic_rules.py",
                "rcpp_core/scene_semantic_inference.py", "rcpp_core/memory_schema.py")],
        }
        input_revision = fingerprint(revision)
        context = {**spatialite_context, "run_id": run_id, "input_revision": input_revision}
        refresh_events = []
        journal_hits = 0
        row_by_id = {r["rps_id"]: r for r in rps_rows}
        local_images = self.targeted_refresher.resolve_image_sources(spatialite_context)
        targets_by_layer: dict[str, list[str]] = {}
        for kind in ("scene", "semantic"):
            targets = [e for e in events if e["layer"] == kind and e["action"] == f"refresh_{kind}"]
            targets_by_layer[kind] = [e["rps_id"] for e in targets]
            for event in targets:
                rid = event["rps_id"]
                route_context = RoutingContext(
                    run_id, kind, memory_state=event["state"],
                    request_risk=global_instruction.get("request_risk", "medium"),
                    latency_budget_ms=global_instruction.get("latency_budget_ms"),
                    cost_budget=global_instruction.get("cost_budget_usd"),
                    shadow_enabled=os.getenv("RCPP_SHADOW_ENABLED", "false").lower() in {"true", "1", "yes"},
                )
                route = self.router.decide(route_context)
                refresh_events.append({"rps_id": rid, "layer": kind, "action": "route",
                                       "context": asdict(route_context), "decision": route.to_dict()})
                if route.action == "human_review":
                    continue
                worker_context = {**context, "model_key": route.production_model,
                                  "rules_only": route.action == "rule_fallback",
                                  "shadow_enabled": bool(route.shadow_model)}
                deployment = (get_deployment(kind, route.production_model,
                    base_model=self.model_settings.get(f"{kind}_base_model"),
                    adapter_path=self.model_settings.get(f"{kind}_adapter_path"))
                    if route.production_model else None)
                source = row_by_id[rid]
                image_paths = [local_images.get(Path(str(p).replace("\\", "/")).name, str(p))
                               for p in source.get("source_images", [])] if kind == "scene" else []
                images = [file_fingerprint(p) for p in image_paths]

                def compute():
                    try:
                        callback = self.scene_refresher if kind == "scene" else self.semantic_refresher
                        if callback:
                            output = callback(area, [rid], worker_context)
                        elif kind == "scene":
                            output = self.targeted_refresher.refresh_scene(area, [rid], worker_context, rps_records=rps_rows)
                        else:
                            output = self.targeted_refresher.refresh_semantic(area, [rid], worker_context)
                        memory = output.get(rid)
                        if not memory:
                            return {"error": "refresh returned no memory"}
                        assessment = assess_scene(memory) if kind == "scene" else assess_semantic(memory)
                        if assessment.state != "fresh":
                            return {"error": "; ".join(assessment.reasons), "invalid_memory": memory}
                        return {"memory": memory}
                    except Exception as exc:
                        return {"error": f"{type(exc).__name__}: {exc}"}

                # The extractor already performs its constrained JSON retry. Journal the complete
                # per-RPS extraction so completed calls are not repeated after a graph-write crash.
                outcome, cached, key = journal.execute(f"{area}/{kind}/{rid}", {
                    "revision": input_revision, "source": source, "images": images,
                    "model": asdict(deployment) if deployment else "deterministic_rules",
                    "route": {"rules_only": worker_context["rules_only"]},
                }, compute)
                journal_hits += int(cached)
                memory = outcome.get("memory")
                if memory:
                    meta = memory.get("_meta") or {}
                    payload = {k: v for k, v in memory.items() if k != "_meta"}
                    append = repository.append_scene_memory if kind == "scene" else repository.append_semantic_memory
                    append(area, rid, payload, memory_id=f"runtime-{key}",
                           base_model=meta.get("base_model") or (deployment.base_model if deployment else "deterministic_rules"),
                           adapter_version=meta.get("adapter_version") or (deployment.adapter_version if deployment else "none"),
                           evidence_source=meta.get("evidence_source") or f"runtime_{kind}_refresh",
                           source_images=image_paths,
                           generated_at=meta.get("generated_at"))
                refresh_events.append({"rps_id": rid, "layer": kind, "action": "refresh_attempt",
                                       "result": "written" if memory else "failed", "journal_reused": cached,
                                       "execution_key": key, "reason": outcome.get("error", "")})
            records, events = repository.load_progressive_context(area, ids) if ids else ({}, [])

        pending = [e for e in events if e["layer"] == "scene" and e["action"] == "refresh_scene"]
        for event in pending:
            event["source_images"] = row_by_id[event["rps_id"]].get("source_images", [])
            route = next((e["decision"] for e in reversed(refresh_events)
                          if e["rps_id"] == event["rps_id"] and e["layer"] == "scene" and "decision" in e), None)
            if route:
                event["route_reasons"] = route["reasons"]
        stop = [e for e in events if e["layer"] == "scene" and e["action"] == "stop"]
        decisions = [{"area": area, "rps_id": e["rps_id"],
                      "outcome": "manual_review_required" if e in pending else "physically_infeasible",
                      "reason": "; ".join(e.get("reasons") or []),
                      "rules": ["scene_memory_quality_gate" if e in pending else "scene_physical_deployment_conditions"],
                      "fact_ids": [], "fact_kinds": ["scene"]} for e in pending + stop]
        repository.append_decisions(run_id, decisions)
        summary = {
            "scene_reused": sum(e["layer"] == "scene" and e["state"] == "fresh" for e in events),
            "scene_refresh": len(targets_by_layer["scene"]), "early_stopped": len(stop),
            "semantic_reused": sum(e["layer"] == "semantic" and e["action"] == "reuse" for e in events),
            "semantic_refresh": len(targets_by_layer["semantic"]), "manual_review": len(pending),
            "execution_receipts_reused": journal_hits,
        }
        out = path_output_environment_perception_agent()
        out.mkdir(parents=True, exist_ok=True)
        (out / f"progressive_retrieval_{area}_{run_id}.json").write_text(
            json.dumps({"events": refresh_events + events, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"workflow_status": "perception_completed", "retrieval_events": refresh_events + events,
                "retrieval_summary": summary, "pending_scene_reviews": pending}
