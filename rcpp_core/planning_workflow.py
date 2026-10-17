"""Explicit planning edges and the human review boundary."""
from __future__ import annotations

from typing import Any, Callable, Mapping
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from rcpp_core.execution_journal import fingerprint
from rcpp_core.graph_memory_repository import assess_scene

WORKFLOW_VERSION = 2


def validate_review(payload: Mapping[str, Any], pending: list[dict[str, Any]],
                    token: str) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or payload.get("review_token") != token:
        raise ValueError("review_token does not match the pending review")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer is required")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not all(isinstance(d, dict) for d in decisions):
        raise ValueError("decisions must be a list of objects")
    expected = {str(e["rps_id"]) for e in pending}
    ids = [str(d.get("rps_id") or "") for d in decisions]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Provide exactly one decision for every pending RPS")
    for decision in decisions:
        if not isinstance(decision.get("note"), str) or not decision["note"].strip():
            raise ValueError("Every decision requires a note explaining its evidence")
        if decision.get("action") not in {"exclude", "correct_scene"}:
            raise ValueError("action must be exclude or correct_scene")
        if decision["action"] == "correct_scene":
            memory = decision.get("scene_memory")
            if not isinstance(memory, dict) or assess_scene(memory).state != "fresh":
                raise ValueError("correct_scene requires a complete and consistent Scene memory")
            images = decision.get("source_images")
            if not isinstance(images, list) or not images or not all(isinstance(x, str) and x.strip() for x in images):
                raise ValueError("correct_scene requires source_images")
    return decisions


def make_human_review_node(repository: Any):
    def node(state):
        pending = state.get("pending_scene_reviews") or []
        if not pending:
            return {}
        instruction = state["global_instruction"]
        run_id = instruction["run_id"]
        area = state["spatialite_context"]["area"]
        token = fingerprint({"run_id": run_id, "area": area, "pending": pending})
        payload = interrupt({"kind": "scene_review", "run_id": run_id, "area": area,
                             "review_token": token, "pending": pending,
                             "actions": ["exclude", "correct_scene"]})
        decisions = validate_review(payload, pending, token)
        # Validate the whole submission before performing any side effects.
        excluded = set(instruction.get("excluded_rps") or [])
        rows = []
        for decision in decisions:
            rid = decision["rps_id"]
            identity = fingerprint({"run_id": run_id, "decision": decision, "reviewer": payload["reviewer"]})
            if decision["action"] == "exclude":
                excluded.add(rid)
            else:
                repository.append_scene_memory(
                    area, rid, decision["scene_memory"], memory_id=f"manual-{identity}",
                    evidence_source=f"human_review:{payload['reviewer']}:{decision['note']}",
                    source_images=decision["source_images"], base_model="human_review", adapter_version="none",
                )
            rows.append({"area": area, "rps_id": rid, "decision_id": f"review-{identity}",
                         "outcome": decision["action"], "reason": f"{payload['reviewer']}: {decision['note']}",
                         "rules": ["manual_scene_review"], "fact_ids": [], "fact_kinds": ["scene"]})
        repository.append_decisions(run_id, rows)
        return {"global_instruction": {**instruction, "excluded_rps": sorted(excluded)},
                "pending_scene_reviews": [], "workflow_status": "scene_review_resolved"}
    return node


def compile_planning_graph(state_type: Any, nodes: Mapping[str, Callable], *, checkpointer: Any = None):
    graph = StateGraph(state_type)
    for name, node in nodes.items():
        graph.add_node(name, node)
    graph.add_edge(START, "TaskOrchestration")
    graph.add_edge("TaskOrchestration", "EnvironmentPerception")
    graph.add_conditional_edges("EnvironmentPerception",
        lambda state: "HumanReview" if state.get("pending_scene_reviews") else "SuitabilityAssessment",
        {name: name for name in ("HumanReview", "SuitabilityAssessment")})
    graph.add_edge("HumanReview", "EnvironmentPerception")
    for left, right in (
        ("SuitabilityAssessment", "MultiScenarioEvaluation"),
        ("MultiScenarioEvaluation", "PhasedDecisionMaking"),
        ("PhasedDecisionMaking", "MemoryUpdate"),
        ("MemoryUpdate", "Review"),
    ):
        graph.add_edge(left, right)
    graph.add_conditional_edges("Review",
        lambda state: "MultiScenarioEvaluation" if state.get("should_readjust_weights") else END,
        {"MultiScenarioEvaluation": "MultiScenarioEvaluation", END: END})
    return graph.compile(checkpointer=checkpointer)
