# -*- coding: utf-8 -*-
"""Neo4j repository for Scene–Semantic graph long-term memory."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from rcpp_core.build_scene_semantic_memory import _get_driver, _session_kwargs
from rcpp_core.memory_lifecycle import classify_memory
from rcpp_core.memory_schema import quality_score, validate_scene_memory, validate_semantic_memory


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("\x1f".join(_canonical(p) for p in parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def normalize_scene(memory: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(memory)
    for name in ("has_existing_RCP", "is_functional_zone", "has_ground_obstacle", "clearance_visual_assessment"):
        if name in out:
            converted = _bool(out[name])
            if converted is not None:
                out[name] = converted
    return out


@dataclass(frozen=True)
class SceneAssessment:
    state: str
    allows_semantic: bool
    reasons: tuple[str, ...]
    quality_score: float


@dataclass(frozen=True)
class MemoryAssessment:
    state: str
    reasons: tuple[str, ...]
    quality_score: float


def assess_scene(memory: Mapping[str, Any], *, declared_status: str | None = None) -> SceneAssessment:
    normalized = normalize_scene(memory)
    errors = validate_scene_memory(normalized)
    status = str(declared_status or normalized.get("status") or "active").lower()
    if status == "conflicted" or errors:
        state = "conflicted" if status == "conflicted" or any("requires" in e for e in errors) else "partial"
        return SceneAssessment(state, False, tuple(errors or ["memory status is conflicted"]), quality_score("scene", errors, normalized))
    lifecycle = classify_memory(normalized)
    if lifecycle in {"expired", "superseded"}:
        return SceneAssessment("stale", False, (f"memory lifecycle is {lifecycle}",), quality_score("scene", [], normalized))
    conditions = {
        "clearance_visual_assessment=true": normalized.get("clearance_visual_assessment") is True,
        "has_existing_RCP=false": normalized.get("has_existing_RCP") is False,
        "is_functional_zone=false": normalized.get("is_functional_zone") is False,
    }
    failed = tuple(name for name, passed in conditions.items() if not passed)
    return SceneAssessment("fresh", not failed, failed, quality_score("scene", [], normalized))


def assess_semantic(memory: Mapping[str, Any], *, declared_status: str | None = None) -> MemoryAssessment:
    errors = validate_semantic_memory(memory)
    status = str(declared_status or (memory.get("_meta") or {}).get("status") or "active").lower()
    if status == "conflicted":
        return MemoryAssessment("conflicted", tuple(errors or ["multiple or conflicting current versions"]),
                                quality_score("semantic", errors, memory))
    if errors or status == "partial":
        return MemoryAssessment("partial", tuple(errors or ["memory status is partial"]),
                                quality_score("semantic", errors, memory))
    lifecycle = classify_memory(memory)
    if lifecycle in {"expired", "superseded"}:
        return MemoryAssessment("stale", (f"memory lifecycle is {lifecycle}",),
                                quality_score("semantic", [], memory))
    return MemoryAssessment("fresh", (), quality_score("semantic", [], memory))


class GraphMemoryRepository:
    """Single read/write boundary for durable planning memory."""

    def __init__(self, driver: Any = None):
        self._driver = driver

    @property
    def driver(self) -> Any:
        return self._driver or _get_driver()

    def _session(self) -> Any:
        return self.driver.session(**_session_kwargs())

    def ensure_schema(self) -> None:
        statements = (
            "CREATE CONSTRAINT rps_identity IF NOT EXISTS FOR (n:RPS) REQUIRE (n.area_slug, n.rps_id) IS UNIQUE",
            "CREATE CONSTRAINT scene_memory_identity IF NOT EXISTS FOR (n:SceneMemory) REQUIRE n.memory_id IS UNIQUE",
            "CREATE CONSTRAINT semantic_memory_identity IF NOT EXISTS FOR (n:SemanticMemory) REQUIRE n.memory_id IS UNIQUE",
            "CREATE CONSTRAINT scene_fact_identity IF NOT EXISTS FOR (n:SceneFact) REQUIRE n.fact_id IS UNIQUE",
            "CREATE CONSTRAINT semantic_fact_identity IF NOT EXISTS FOR (n:SemanticFact) REQUIRE n.fact_id IS UNIQUE",
            "CREATE CONSTRAINT planning_run_identity IF NOT EXISTS FOR (n:PlanningRun) REQUIRE n.run_id IS UNIQUE",
            "CREATE CONSTRAINT decision_identity IF NOT EXISTS FOR (n:Decision) REQUIRE n.decision_id IS UNIQUE",
        )
        with self._session() as session:
            for statement in statements:
                session.run(statement).consume()

    def upsert_rps(self, area: str, records: Iterable[Mapping[str, Any]]) -> None:
        rows = []
        for item in records:
            coords = item.get("coordinates") or []
            rows.append({
                "area": area.lower(), "rps_id": str(item["rps_id"]),
                "longitude": coords[0] if len(coords) > 0 else None,
                "latitude": coords[1] if len(coords) > 1 else None,
                "street_name": str(item.get("street_name") or ""),
                "source_images": [str(x) for x in (item.get("source_images") or [])],
            })
        if not rows:
            return
        with self._session() as session:
            session.run("""
            UNWIND $rows AS row
            MERGE (r:RPS {area_slug: row.area, rps_id: row.rps_id})
            SET r.longitude=row.longitude, r.latitude=row.latitude,
                r.street_name=row.street_name, r.source_images=row.source_images,
                r.updated_at=datetime()
            """, rows=rows).consume()

    def append_scene_memory(self, area: str, rps_id: str, memory: Mapping[str, Any], **metadata: Any) -> str:
        return self._append_memory("scene", area, rps_id, memory, metadata)

    def append_semantic_memory(self, area: str, rps_id: str, memory: Mapping[str, Any], **metadata: Any) -> str:
        return self._append_memory("semantic", area, rps_id, memory, metadata)

    def _append_memory(self, kind: str, area: str, rps_id: str, memory: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
        payload = normalize_scene(memory) if kind == "scene" else dict(memory)
        errors = validate_scene_memory(payload) if kind == "scene" else validate_semantic_memory(payload)
        requested_status = str(metadata.get("status") or "active")
        status = "conflicted" if requested_status == "conflicted" else "partial" if errors else requested_status
        generated_at = str(metadata.get("generated_at") or _now())
        memory_id = str(metadata.get("memory_id") or stable_id(kind, area.lower(), rps_id, payload, generated_at))
        quality_input = dict(payload)
        if metadata.get("source_image") or metadata.get("source_images"):
            quality_input["source_image"] = True
        if metadata.get("evidence_source"):
            quality_input["evidence_source"] = metadata["evidence_source"]
        label, relation, fact_label = (
            ("SceneMemory", "HAS_SCENE_MEMORY", "SceneFact") if kind == "scene"
            else ("SemanticMemory", "HAS_SEMANTIC_MEMORY", "SemanticFact")
        )
        meta = {
            "memory_id": memory_id, "area": area.lower(), "rps_id": str(rps_id),
            "payload": _canonical(payload), "status": status,
            "base_model": str(metadata.get("base_model") or "legacy"),
            "adapter_version": str(metadata.get("adapter_version") or "legacy"),
            "schema_version": str(metadata.get("schema_version") or "2.0"),
            "generated_at": generated_at,
            "expires_at": str(metadata.get("expires_at") or ""),
            "quality_score": float(metadata.get("quality_score", quality_score(kind, errors, quality_input))),
            "evidence_source": str(metadata.get("evidence_source") or "migration"),
        }
        facts = self._facts(kind, memory_id, payload, metadata)
        source_images = metadata.get("source_images") or ([metadata["source_image"]] if metadata.get("source_image") else [])
        def write(tx):
            result = tx.run(f"""
            MERGE (r:RPS {{area_slug:$area, rps_id:$rps_id}})
            SET r._memory_write_lock=coalesce(r._memory_write_lock,0)+1
            MERGE (m:{label} {{memory_id:$memory_id}})
            ON CREATE SET m.created_at=datetime(), m.payload_json=$payload,
              m.area_slug=$area, m.rps_id=$rps_id, m.status=$status,
              m.base_model=$base_model, m.adapter_version=$adapter_version,
              m.schema_version=$schema_version, m.generated_at=$generated_at,
              m.expires_at=$expires_at, m.quality_score=$quality_score,
              m.evidence_source=$evidence_source, m.is_current=true
            MERGE (r)-[:{relation}]->(m)
            WITH r,m
            WHERE m.payload_json=$payload AND m.area_slug=$area AND m.rps_id=$rps_id
            OPTIONAL MATCH (r)-[:{relation}]->(old:{label})
            WHERE m.is_current=true AND old.is_current=true AND old.memory_id<>m.memory_id
            WITH m, collect(old) AS previous
            FOREACH (old IN previous | SET old.is_current=false, old.status='superseded')
            FOREACH (old IN previous | MERGE (m)-[:SUPERSEDES]->(old))
            RETURN m.memory_id AS memory_id
            """, **meta).single()
            if result is None:
                raise ValueError("memory_id already exists with a different payload or RPS identity")
            if facts:
                tx.run(f"""
                MATCH (m:{label} {{memory_id:$memory_id}})
                UNWIND $facts AS fact
                MERGE (f:{fact_label} {{fact_id:fact.fact_id}})
                SET f.field=fact.field, f.value_json=fact.value_json, f.unit=fact.unit,
                    f.raw_text=fact.raw_text, f.evidence_source=fact.evidence_source
                MERGE (m)-[:CONTAINS_FACT]->(f)
                """, memory_id=memory_id, facts=facts).consume()
            if kind == "scene" and source_images:
                tx.run("""
                MATCH (m:SceneMemory {memory_id:$memory_id})
                UNWIND $images AS path
                MERGE (s:SourceImage {source_id:path}) SET s.path=path
                MERGE (m)-[:DERIVED_FROM]->(s)
                """, memory_id=memory_id, images=[str(p) for p in source_images]).consume()
            return str(result["memory_id"])
        # Version replacement, facts and image provenance commit atomically. A retry uses
        # the same identity and cannot demote a newer version when replaying an old receipt.
        with self._session() as session:
            return session.execute_write(write)

    @staticmethod
    def _facts(kind: str, memory_id: str, payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[dict[str, str]]:
        units = metadata.get("units") if isinstance(metadata.get("units"), Mapping) else {}
        raw = str(metadata.get("raw_text") or payload.get("raw_text") or "")
        evidence = str(metadata.get("evidence_source") or "migration")
        flat: list[tuple[str, Any]] = []

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, Mapping):
                for child, child_value in value.items():
                    visit(f"{prefix}.{child}" if prefix else str(child), child_value)
            else:
                flat.append((prefix, value))

        for field, value in payload.items():
            if not str(field).startswith("_"):
                visit(str(field), value)
        return [{
            "fact_id": stable_id("fact", memory_id, field), "field": str(field),
            "value_json": _canonical(value), "unit": str(units.get(field) or ""),
            "raw_text": raw, "evidence_source": evidence,
        } for field, value in flat]

    def _get_current(self, kind: str, area: str, rps_ids: Sequence[str] | None) -> dict[str, dict[str, Any]]:
        label, relation = ("SceneMemory", "HAS_SCENE_MEMORY") if kind == "scene" else ("SemanticMemory", "HAS_SEMANTIC_MEMORY")
        ids = [str(x) for x in (rps_ids or [])]
        with self._session() as session:
            rows = list(session.run(f"""
            MATCH (r:RPS {{area_slug:$area}})-[:{relation}]->(m:{label})
            WHERE m.is_current=true AND (size($ids)=0 OR r.rps_id IN $ids)
            WITH r, collect(m) AS current
            RETURN r.rps_id AS rps_id, current
            """, area=area.lower(), ids=ids))
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            nodes = row.get("current") or []
            if not nodes:
                continue
            node = dict(nodes[0])
            payload = json.loads(node.get("payload_json") or "{}")
            payload["_meta"] = {k: node.get(k) for k in (
                "memory_id", "status", "base_model", "adapter_version", "schema_version",
                "generated_at", "expires_at", "quality_score", "evidence_source", "is_current",
            )}
            if len(nodes) > 1:
                payload["_meta"]["status"] = "conflicted"
                payload["_meta"]["conflicting_current_count"] = len(nodes)
            out[str(row["rps_id"])] = payload
        return out

    def get_current_scene(self, area: str, rps_ids: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
        return self._get_current("scene", area, rps_ids)

    def get_current_semantic(self, area: str, rps_ids: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
        return self._get_current("semantic", area, rps_ids)

    def list_rps(self, area: str) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.run("""
            MATCH (r:RPS {area_slug:$area}) RETURN r.rps_id AS rps_id,
              r.longitude AS longitude, r.latitude AS latitude, r.street_name AS street_name,
              r.source_images AS source_images
            ORDER BY r.rps_id
            """, area=area.lower())
            return [dict(row) for row in rows]

    def load_progressive_context(self, area: str, rps_ids: Sequence[str] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rps_rows = self.list_rps(area)
        rps_meta = {row["rps_id"]: row for row in rps_rows}
        requested = [str(x) for x in rps_ids] if rps_ids else list(rps_meta)
        scene = self.get_current_scene(area, requested)
        passed: list[str] = []
        events: list[dict[str, Any]] = []
        for rps_id in requested:
            payload = scene.get(rps_id)
            if payload is None:
                events.append({"rps_id": rps_id, "layer": "scene", "state": "miss",
                               "action": "refresh_scene", "reasons": ["current Scene memory is missing"]})
                continue
            meta = payload.get("_meta") or {}
            assessment = assess_scene(payload, declared_status=meta.get("status"))
            action = "semantic_lookup" if assessment.allows_semantic else "stop" if assessment.state == "fresh" else "refresh_scene"
            events.append({"rps_id": rps_id, "layer": "scene", "state": assessment.state,
                           "action": action, "reasons": list(assessment.reasons)})
            if assessment.allows_semantic:
                passed.append(rps_id)
        semantic = self.get_current_semantic(area, passed) if passed else {}
        records: dict[str, Any] = {}
        for rps_id in requested:
            scene_payload = scene.get(rps_id)
            scene_clean = {k: v for k, v in (scene_payload or {}).items() if k != "_meta"}
            semantic_payload = semantic.get(rps_id)
            records[rps_id] = {
                "RPS_id": rps_id,
                "coordinates": [rps_meta.get(rps_id, {}).get("longitude"), rps_meta.get(rps_id, {}).get("latitude")],
                "street_name": rps_meta.get(rps_id, {}).get("street_name", ""),
                "memory_node": {**({"scene_memory": scene_clean} if scene_payload else {}),
                                **({"semantic_memory": {k:v for k,v in semantic_payload.items() if k != "_meta"}} if semantic_payload else {})},
            }
        for rps_id in passed:
            payload = semantic.get(rps_id)
            if payload is None:
                events.append({"rps_id": rps_id, "layer": "semantic", "state": "miss",
                               "action": "refresh_semantic", "reasons": ["current Semantic memory is missing"]})
                continue
            assessment = assess_semantic(payload, declared_status=(payload.get("_meta") or {}).get("status"))
            events.append({"rps_id": rps_id, "layer": "semantic", "state": assessment.state,
                           "action": "reuse" if assessment.state == "fresh" else "refresh_semantic",
                           "reasons": list(assessment.reasons)})
        return records, events

    def append_decisions(self, run_id: str, decisions: Iterable[Mapping[str, Any]]) -> None:
        rows = []
        for decision in decisions:
            rid = str(decision["rps_id"])
            did = str(decision.get("decision_id") or stable_id("decision", run_id, rid, decision))
            rows.append({"decision_id": did, "run_id": run_id, "area": str(decision["area"]).lower(),
                         "rps_id": rid, "outcome": str(decision.get("outcome") or ""),
                         "reason": str(decision.get("reason") or ""),
                         "rules": [str(x) for x in decision.get("rules", [])],
                         "fact_ids": [str(x) for x in decision.get("fact_ids", [])],
                         "fact_kinds": [str(x) for x in decision.get("fact_kinds", ["scene", "semantic"])]})
        if not rows:
            return
        with self._session() as session:
            session.run("""
            UNWIND $rows AS row
            MERGE (run:PlanningRun {run_id:row.run_id}) ON CREATE SET run.created_at=datetime()
            MATCH (r:RPS {area_slug:row.area, rps_id:row.rps_id})
            MERGE (d:Decision {decision_id:row.decision_id})
            SET d.outcome=row.outcome, d.reason=row.reason, d.created_at=datetime()
            MERGE (d)-[:ABOUT]->(r) MERGE (d)-[:PRODUCED_IN]->(run)
            WITH d,row UNWIND row.rules AS rule_name
            MERGE (rule:PlanningRule {name:rule_name}) MERGE (d)-[:APPLIES_RULE]->(rule)
            """, rows=rows).consume()
            session.run("""
            UNWIND $rows AS row MATCH (d:Decision {decision_id:row.decision_id})
            UNWIND row.fact_ids AS fact_id
            MATCH (f) WHERE (f:SceneFact OR f:SemanticFact) AND f.fact_id=fact_id
            MERGE (d)-[:SUPPORTED_BY]->(f)
            """, rows=rows).consume()
            session.run("""
            UNWIND $rows AS row
            WITH row WHERE size(row.fact_ids)=0
            MATCH (d:Decision {decision_id:row.decision_id})-[:ABOUT]->(r:RPS)
            MATCH (r)-[:HAS_SCENE_MEMORY|HAS_SEMANTIC_MEMORY]->(m)
            WHERE m.is_current=true AND
              ((m:SceneMemory AND 'scene' IN row.fact_kinds) OR
               (m:SemanticMemory AND 'semantic' IN row.fact_kinds))
            MATCH (m)-[:CONTAINS_FACT]->(f)
            MERGE (d)-[:SUPPORTED_BY]->(f)
            """, rows=rows).consume()

    def get_memory_history(self, area: str, rps_id: str, kind: str) -> list[dict[str, Any]]:
        label, relation = ("SceneMemory", "HAS_SCENE_MEMORY") if kind == "scene" else ("SemanticMemory", "HAS_SEMANTIC_MEMORY")
        with self._session() as session:
            rows = session.run(f"""
            MATCH (:RPS {{area_slug:$area,rps_id:$rps_id}})-[:{relation}]->(m:{label})
            OPTIONAL MATCH (m)-[:SUPERSEDES]->(old:{label})
            RETURN properties(m) AS memory, collect(old.memory_id) AS supersedes
            ORDER BY m.generated_at DESC
            """, area=area.lower(), rps_id=rps_id)
            return [{"memory": dict(row["memory"]), "supersedes": list(row["supersedes"] or [])} for row in rows]

    def get_evidence_path(self, run_id: str, rps_id: str) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.run("""
            MATCH (run:PlanningRun {run_id:$run_id})<-[:PRODUCED_IN]-(d:Decision)-[:ABOUT]->(r:RPS {rps_id:$rps_id})
            OPTIONAL MATCH (d)-[:APPLIES_RULE]->(rule:PlanningRule)
            OPTIONAL MATCH (d)-[:SUPPORTED_BY]->(fact)
            OPTIONAL MATCH (memory)-[:CONTAINS_FACT]->(fact)
            OPTIONAL MATCH (memory)-[:DERIVED_FROM]->(source:SourceImage)
            RETURN properties(d) AS decision, r.area_slug AS area,
              collect(DISTINCT rule.name) AS rules, collect(DISTINCT properties(fact)) AS facts,
              collect(DISTINCT properties(memory)) AS memories, collect(DISTINCT properties(source)) AS sources
            """, run_id=run_id, rps_id=rps_id)
            return [dict(row) for row in rows]

    def get_area_counts(self, area: str) -> dict[str, int]:
        with self._session() as session:
            row = session.run("""
            MATCH (r:RPS {area_slug:$area})
            OPTIONAL MATCH (r)-[:HAS_SCENE_MEMORY]->(s:SceneMemory) WHERE s.is_current=true
            WITH count(DISTINCT r) AS rps, count(DISTINCT s) AS scene
            MATCH (r2:RPS {area_slug:$area})
            OPTIONAL MATCH (r2)-[:HAS_SEMANTIC_MEMORY]->(m:SemanticMemory) WHERE m.is_current=true
            RETURN rps, scene, count(DISTINCT m) AS semantic
            """, area=area.lower()).single()
            return {"rps": int(row["rps"]), "scene": int(row["scene"]),
                    "semantic": int(row["semantic"])} if row else {"rps": 0, "scene": 0, "semantic": 0}

    def delete_legacy_aggregate(self, area: str) -> int:
        with self._session() as session:
            row = session.run("""
            MATCH (n:GlobalSceneSemanticMemory {area_slug:$area}) WITH collect(n) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n) RETURN size(nodes) AS deleted
            """, area=area.lower()).single()
            return int(row["deleted"] if row else 0)
