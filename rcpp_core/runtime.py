"""
RCPP-Agent: multi-agent roadside charging pile (RCP) planning with LangGraph orchestration.

It chains task orchestration, environment perception, suitability assessment, scenario
weighting (MCDA), phased decision-making, Neo4j-backed memory update, and review.
"""

from __future__ import annotations

import copy
import logging
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict, cast

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import PlanningPhase
from configs.data_config import (
    RCPP_AGENT_MAX_WORKFLOW_ITERATIONS,
    default_checkpoint_db_path,
    path_output_multi_scenario_evaluation_agent,
    path_output_phased_decision_making_agent,
    path_output_review_agent,
    path_output_suitability_assessment_agent,
)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from tools.rcpp_paths import default_poi_dir
from langgraph.store.base import BaseStore
from rcpp_core.neo4j_scene_semantic_store import (
    SCENE_SEMANTIC_KEY_GLOBAL,
    Neo4jGlobalSceneSemanticStore,
    area_slug_from_spatialite_context,
    scene_semantic_namespace,
)
from src.task_orchestration_agent import TaskOrchestrationAgent
from src.environment_perception_agent import EnvironmentPerceptionAgent
from src.suitability_assessment_agent import (
    SuitabilityAssessmentAgent,
    export_suitability_shapefiles,
    guess_rps_id_column,
    load_rps_gdf_for_export,
)
from src.multi_scenario_evaluation_agent import MultiScenarioEvaluationAgent
from src.phased_decision_making_agent import (
    PhasedDecisionMakingAgent,
    export_phase_rps_shapefile,
    write_json_path,
)
from src.review_agent import ReviewAgent, write_review_metrics_json

logger = logging.getLogger(__name__)

_MEMORY_EVENTS_CLEAR_SOURCE = "__memory_events_clear__"
_PHASED_PRIORITY_SOURCE = "PhasedDecisionMakingAgent"
_PHASED_PRIORITY_FIELD = "priority_score"

# Memory update event schema
class MemoryUpdateEvent(TypedDict):
    rps_id: str
    source: str
    field: str
    value: Any


def _apply_memory_update_events_to_memory(
    memory: Dict[str, Any],
    events: List[MemoryUpdateEvent],
) -> int:
    """
    Merge update events into memory.
    """
    applied = 0
    for evt in events:
        rps_id = str(evt.get("rps_id") or "")
        if not rps_id:
            continue
        if rps_id not in memory:
            continue
        node_data = memory[rps_id]
        if not isinstance(node_data, dict):
            continue
        annotations = node_data.setdefault("agent_annotations", {})
        source = str(evt.get("source") or "")
        field = str(evt.get("field") or "")
        if not source or not field:
            continue
        annotations.setdefault(source, {})[field] = evt.get("value")
        applied += 1
    return applied


# Merge or clear memory update events using a reducer.
def _memory_update_events_reducer(
    existing: Optional[List[MemoryUpdateEvent]],
    update: Optional[List[MemoryUpdateEvent]],
) -> List[MemoryUpdateEvent]:
    left = existing or []
    if not update:
        return left
    if (
        len(update) == 1
        and str(update[0].get("source", "")) == _MEMORY_EVENTS_CLEAR_SOURCE
    ):
        return []
    return left + update


# Full LangGraph state shared across supervisor and worker agents.
class RCPPAgentState(TypedDict):

    # Task Orchestration Agent
    global_instruction: Dict[str, Any]
    spatialite_context: Dict[str, Any]
    scenario: str
    time_horizon: Any
    workspace_dir: Optional[str]
    year: int

    # Suitability Assessment Agent
    suitable_rps: List[Dict[str, Any]]
    suitability_evaluations: List[Any]

    # Multi-Scenario Evaluation Agent
    scenario_mcda_weights: Dict[str, float]

    # Phased Decision-Making Agent
    mcda_weights: Dict[str, float]
    phased_strategy: Dict[str, Any]
    scored_candidates: List[Dict[str, Any]]

    # Review Agent
    review_result: Optional[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]

    # Structured review feedback into the next Multi-Scenario Evaluation Agent round.
    review_feedback: str

    # Memory update event channel: append via reducer; memory update node clears with sentinel.
    memory_update_events: Annotated[List[MemoryUpdateEvent], _memory_update_events_reducer]

    # Control flow
    iteration_count: int
    should_readjust_weights: bool
    workflow_status: str


# Partial state patch returned by graph nodes before merging into full state.
class RCPPAgentStateUpdate(TypedDict, total=False):

    global_instruction: Dict[str, Any]
    spatialite_context: Dict[str, Any]
    scenario: str
    time_horizon: Any
    workspace_dir: Optional[str]
    year: int
    suitable_rps: List[Dict[str, Any]]
    suitability_evaluations: List[Any]
    scenario_mcda_weights: Dict[str, float]
    mcda_weights: Dict[str, float]
    phased_strategy: Dict[str, Any]
    scored_candidates: List[Dict[str, Any]]
    review_result: Optional[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]
    review_feedback: str
    memory_update_events: List[MemoryUpdateEvent]
    iteration_count: int
    should_readjust_weights: bool
    workflow_status: str


# Build the Task Orchestration supervisor graph node callable.
def _make_task_orchestration_node(orchestrator: TaskOrchestrationAgent):
    """
    Supervisor node.

    On first entry it parses the global instruction and produces the orchestration result. 
    On every subsequent visit it acts as a routing hub and returns an empty state update; the downstream
    conditional_edges dispatches to the next agent based on workflow_status.
    """
    # Run orchestration once on init, then return empty updates for routing only.
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Supervisor] Task Orchestration ---")
        status = state.get("workflow_status", "init")
        if status != "init":
            logger.info("Supervisor routing (workflow_status=%s).", status)
            return {}

        global_instruction = state.get("global_instruction", {})
        orchestration_result = orchestrator.orchestrate(global_instruction)

        time_horizon = orchestration_result.get("time_horizon")
        year = global_instruction.get("year")
        if year is None and isinstance(time_horizon, (list, tuple)) and len(time_horizon) > 0:
            try:
                year = int(time_horizon[0])
            except (TypeError, ValueError):
                year = state.get("year", 2025)
        elif year is None:
            year = state.get("year", 2025)

        return {
            "spatialite_context": orchestration_result.get("spatialite_context", {}),
            "scenario": orchestration_result.get("scenario", state.get("scenario", "balance_oriented")),
            "time_horizon": time_horizon,
            "year": year,
            "workspace_dir": orchestration_result.get("workspace_dir")
            or global_instruction.get("workspace_dir"),
            "workflow_status": "task_orchestrated",
            "iteration_count": state.get("iteration_count", 0),
        }

    return node


# Build the Environment Perception graph node callable.
def _make_environment_perception_node(agent: EnvironmentPerceptionAgent, store: BaseStore):
    # Load or build scene-semantic memory from spatialite context and instruction.
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Environment Perception Agent ---")
        spatialite_context = state.get("spatialite_context", {})
        global_instruction = state.get("global_instruction") or {}
        return cast(
            RCPPAgentStateUpdate,
            agent.perceive(spatialite_context, global_instruction, store=store),
        )

    return node


# Build the Suitability Assessment graph node callable.
def _make_suitability_assessment_node(agent: SuitabilityAssessmentAgent, store: BaseStore):
    # Filter RPS sites, emit suitability memory events, and set workflow status.
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Suitability Assessment Agent ---")
        ctx = state.get("spatialite_context") or {}
        area = area_slug_from_spatialite_context(ctx)
        ns = scene_semantic_namespace(area)
        mem_item = store.get(ns, SCENE_SEMANTIC_KEY_GLOBAL)
        memories: Dict[str, Any] = (
            dict(mem_item.value)
            if mem_item is not None and isinstance(mem_item.value, dict)
            else {}
        )

        suitable_rps, evaluations = agent.assess_suitability(memories)

        events: List[MemoryUpdateEvent] = []
        for ev in evaluations:
            if not ev.is_suitable:
                events.append({
                    "rps_id": ev.rps_id,
                    "source": "SuitabilityAssessmentAgent",
                    "field": "suitability_rejected",
                    "value": {
                        "violated_rules": ev.violated_rules,
                        "overall_score": ev.overall_score,
                    },
                })
            else:
                events.append({
                    "rps_id": ev.rps_id,
                    "source": "SuitabilityAssessmentAgent",
                    "field": "suitability_score",
                    "value": ev.overall_score,
                })

        memory_out = copy.deepcopy(memories)
        applied = _apply_memory_update_events_to_memory(memory_out, events)
        store.put(ns, SCENE_SEMANTIC_KEY_GLOBAL, memory_out)
        logger.info(
            "Suitability: persisted %s suitability annotation events to scene-semantic store.",
            applied,
        )

        gdf = load_rps_gdf_for_export(ctx)
        if gdf is not None:
            try:
                export_suitability_shapefiles(
                    gdf,
                    evaluations,
                    path_output_suitability_assessment_agent(),
                    area,
                )
            except Exception as exc:
                logger.warning("Suitability shapefile export skipped: %s", exc)
        else:
            logger.warning(
                "RPS geometry unavailable (no usable rps_shp_path and no SpatiaLite rps layer); "
                "suitability SHP export skipped."
            )

        return {
            "suitable_rps": suitable_rps,
            "suitability_evaluations": evaluations,
            "workflow_status": "suitability_assessed",
        }

    return node


# Build the Multi-Scenario Evaluation graph node callable.
def _make_multi_scenario_evaluation_node(agent: MultiScenarioEvaluationAgent, store: BaseStore):
    """
    Load scene-semantic store after suitability write; compute MCDA weights via LLM-AHP.
    """
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Multi-Scenario Evaluation Agent ---")
        scenario = state.get("scenario", "equity_oriented")
        iteration = state.get("iteration_count", 0)

        ctx = state.get("spatialite_context") or {}
        area = area_slug_from_spatialite_context(ctx)
        ns = scene_semantic_namespace(area)
        mem_item = store.get(ns, SCENE_SEMANTIC_KEY_GLOBAL)
        memories: Dict[str, Any] = (
            dict(mem_item.value)
            if mem_item is not None and isinstance(mem_item.value, dict)
            else {}
        )
        logger.info(
            "Multi-Scenario Evaluation: loaded scene-semantic store (%s RPS keys).",
            len(memories),
        )

        review_fb = state.get("review_feedback", "")
        if state.get("should_readjust_weights") and review_fb:
            logger.info("Feeding review diagnostics into LLM-AHP weight revision (iteration %s).", iteration)
            revision_note = f"Round {iteration}. Review feedback: {review_fb}"
        else:
            revision_note = str(iteration) if iteration else ""

        weights_info = agent.determine_weights(
            scenario,
            weight_revision=iteration,
            revision_note=revision_note,
            scene_semantic_memory=memories,
        )

        mse_dir = path_output_multi_scenario_evaluation_agent()
        mse_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_json_path(
                mse_dir / f"llm_ahp_scenario_{scenario}_{area}_iter{iteration}.json",
                {
                    "scenario": scenario,
                    "area": area,
                    "iteration": iteration,
                    "revision_note": revision_note,
                    "llm_ahp_scenario": weights_info.get("llm_ahp_scenario"),
                },
            )
        except OSError as exc:
            logger.warning("Could not write llm_ahp_scenario JSON: %s", exc)

        return {
            "scenario_mcda_weights": weights_info["scenario_mcda_weights"],
            "workflow_status": "weights_calculated",
        }

    return node


# Build the Phased Decision-Making graph node callable.
def _make_phased_decision_making_node(agent: PhasedDecisionMakingAgent, store: BaseStore):
    # Score candidates, select quota, emit priority memory events, and set status.
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Phased Decision-Making Agent ---")
        suitable_rps = state.get("suitable_rps", [])
        ctx = state.get("spatialite_context") or {}
        area = area_slug_from_spatialite_context(ctx)
        ns = scene_semantic_namespace(area)
        mem_item = store.get(ns, SCENE_SEMANTIC_KEY_GLOBAL)
        memories: Dict[str, Any] = (
            dict(mem_item.value)
            if mem_item is not None and isinstance(mem_item.value, dict)
            else {}
        )
        scenario_mcda_weights = state.get("scenario_mcda_weights", {})
        scenario = state.get("scenario", "equity_oriented")
        time_horizon = state.get("time_horizon")
        year = int(state.get("year", 2025))
        planning_phase = (
            PlanningPhase.INITIAL_EXPLORATION
            if year <= 2026
            else PlanningPhase.RAPID_DEVELOPMENT
            if year <= 2028
            else PlanningPhase.MATURITY_OPTIMIZATION
        )
        logger.info(
            "Instruction year=%s -> planning_phase=%s. Formulating phase-by-phase plan.",
            year,
            planning_phase.value,
        )

        decision_result = agent.formulate_phase_planning(
            suitable_rps,
            memories,
            scenario_mcda_weights,
            scenario,
            time_horizon=time_horizon,
            spatialite_context=state.get("spatialite_context"),
            weight_revision=state.get("iteration_count", 0),
            instruction_year=year,
        )

        phased_dir = path_output_phased_decision_making_agent()
        phased_dir.mkdir(parents=True, exist_ok=True)
        iter_n = int(state.get("iteration_count", 0))
        try:
            write_json_path(
                phased_dir / f"scenario_mcda_weights_{scenario}_{area}.json",
                scenario_mcda_weights,
            )
        except OSError as exc:
            logger.warning("Could not write scenario_mcda_weights JSON: %s", exc)
        try:
            write_json_path(
                phased_dir / f"llm_ahp_phase_{scenario}_{area}_iter{iter_n}.json",
                {
                    "scenario": scenario,
                    "area": area,
                    "iteration": iter_n,
                    "phases": [
                        {
                            "phase_tag": p.get("phase_tag"),
                            "skipped": bool(
                                (p.get("phased_strategy") or {}).get("skipped")
                            ),
                            "llm_ahp_phase": p.get("llm_ahp_phase"),
                        }
                        for p in decision_result.get("phase_round", [])
                    ],
                },
            )
        except OSError as exc:
            logger.warning("Could not write llm_ahp_phase JSON: %s", exc)

        gdf = load_rps_gdf_for_export(ctx)
        id_col: Optional[str] = None
        if gdf is not None:
            id_col = guess_rps_id_column(gdf)
            if id_col is None:
                logger.warning(
                    "Could not guess RPS id column on RPS GeoDataFrame; phased SHP export may be skipped."
                )

        for pr in decision_result.get("phase_round", []):
            if pr.get("phased_strategy", {}).get("skipped"):
                continue
            tag = str(pr.get("phase_tag", "phase"))
            ids = pr.get("phased_strategy", {}).get("selected_rps_ids", [])
            if gdf is not None and id_col and ids:
                try:
                    export_phase_rps_shapefile(
                        gdf,
                        ids,
                        phased_dir / f"{scenario}_{tag}_{area}.shp",
                        id_column=id_col,
                        memories=memories,
                    )
                except Exception as exc:
                    logger.warning("Phase SHP export failed (%s): %s", tag, exc)

        cps = decision_result.get("combined_phased_strategy") or {}
        itag = str(cps.get("instruction_phase_tag", "phase1"))
        for pr in decision_result.get("phase_round", []):
            if pr.get("phase_tag") != itag or pr.get("phased_strategy", {}).get("skipped"):
                continue
            ids = pr.get("phased_strategy", {}).get("selected_rps_ids", [])
            if gdf is not None and id_col and ids:
                try:
                    export_phase_rps_shapefile(
                        gdf,
                        ids,
                        phased_dir / f"{scenario}_instruction_{itag}_{area}.shp",
                        id_column=id_col,
                        memories=memories,
                    )
                except Exception as exc:
                    logger.warning("Instruction-phase SHP export failed (%s): %s", itag, exc)
            break

        mem_cands = decision_result.get("scored_candidates_memory", [])
        events: List[MemoryUpdateEvent] = []
        for cand in mem_cands:
            events.append({
                "rps_id": cand["rps_id"],
                "source": "PhasedDecisionMakingAgent",
                "field": "priority_score",
                "value": cand.get("overall_priority_score", 0.0),
            })

        return {
            "phased_strategy": decision_result["combined_phased_strategy"],
            "scored_candidates": mem_cands,
            "mcda_weights": decision_result.get("mcda_weights", {}),
            "memory_update_events": events,
            "workflow_status": "decision_made",
        }

    return node


# Build the Memory Update graph node callable.
def _make_memory_update_node(store: BaseStore):
    """
    Updates phased priority score events to scene-semantic memory.
    """
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Memory Update ---")
        raw = state.get("memory_update_events", [])
        events = [
            e
            for e in raw
            if str(e.get("source", "")) == _PHASED_PRIORITY_SOURCE
            and str(e.get("field", "")) == _PHASED_PRIORITY_FIELD
        ]
        if not events:
            state_patch: RCPPAgentStateUpdate = {"workflow_status": "memory_updated"}
            if raw:
                logger.warning(
                    "Memory update: no phased priority_score events in channel (%s raw); clearing.",
                    len(raw),
                )
                state_patch["memory_update_events"] = [
                    {
                        "rps_id": "",
                        "source": _MEMORY_EVENTS_CLEAR_SOURCE,
                        "field": "",
                        "value": None,
                    }
                ]
            return state_patch

        ctx = state.get("spatialite_context") or {}
        area = area_slug_from_spatialite_context(ctx)
        ns = scene_semantic_namespace(area)
        mem_item = store.get(ns, SCENE_SEMANTIC_KEY_GLOBAL)
        memory = copy.deepcopy(
            dict(mem_item.value)
            if mem_item is not None and isinstance(mem_item.value, dict)
            else {}
        )

        applied = _apply_memory_update_events_to_memory(memory, events)

        logger.info(
            "Memory update: %s phased priority events (of %s raw), %s applied to %s RPS nodes.",
            len(events), len(raw), applied, len(memory),
        )

        store.put(ns, SCENE_SEMANTIC_KEY_GLOBAL, memory)

        return {
            "memory_update_events": [
                {
                    "rps_id": "",
                    "source": _MEMORY_EVENTS_CLEAR_SOURCE,
                    "field": "",
                    "value": None,
                }
            ],
            "workflow_status": "memory_updated",
        }

    return node


# Build the Review graph node callable with iteration capacity for weight readjust.
def _make_review_node(agent: ReviewAgent, max_iterations: int, store: BaseStore):
    # Run metrics-based review, set retry flags, and bump iteration count.
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Review Agent ---")
        strategy = state.get("phased_strategy", {})
        candidates = state.get("scored_candidates", [])
        iteration_count = state.get("iteration_count", 0)

        ctx = state.get("spatialite_context") or {}
        area = area_slug_from_spatialite_context(ctx)
        ns = scene_semantic_namespace(area)
        mem_item = store.get(ns, SCENE_SEMANTIC_KEY_GLOBAL)
        memories: Dict[str, Any] = (
            dict(mem_item.value)
            if mem_item is not None and isinstance(mem_item.value, dict)
            else {}
        )
        poi_raw = ctx.get("poi_dir")
        if poi_raw:
            poi_dir = Path(poi_raw)
        else:
            poi_dir = default_poi_dir(state.get("workspace_dir"))
        existing_rcp_raw = ctx.get("existing_rcp")
        existing_rcp = Path(existing_rcp_raw) if existing_rcp_raw else None
        review_result = agent.evaluate_outcomes(
            strategy,
            candidates,
            scene_semantic_memory=memories,
            poi_dir=poi_dir,
            existing_rcp=existing_rcp,
            phased_planning_dir=path_output_phased_decision_making_agent(),
            phased_planning_area=area,
            review_attempt=iteration_count,
            max_readjust_rounds=max_iterations,
        )

        scenario = state.get("scenario", "equity_oriented")
        try:
            rdir = path_output_review_agent()
            rdir.mkdir(parents=True, exist_ok=True)
            write_review_metrics_json(
                rdir / f"review_metrics_{scenario}_{area}_round{iteration_count + 1}.json",
                review_result,
            )
        except OSError as exc:
            logger.warning("Review metrics JSON export failed: %s", exc)

        passed = review_result["passed"]
        should_retry = not passed and iteration_count < max_iterations

        review_feedback = ""
        if should_retry:
            metrics = review_result.get("metrics", {})
            threshold = float(review_result.get("threshold", 0.6))
            comp = float(review_result.get("comprehensive_score", 0.0))
            scr = float(metrics.get("scr", 0.0))
            dcr = float(metrics.get("dcr", 0.0))
            nle = float(metrics.get("nle", 0.0))
            eb = float(metrics.get("eb", 0.0))
            review_feedback = (
                f"Comprehensive weighted score={comp:.4f} (threshold={threshold:.4f}). "
                f"Component scores: SCR={scr:.4f}, DCR={dcr:.4f}, NLE={nle:.4f}, EB={eb:.4f}."
            )

        return {
            "review_result": review_result,
            "review_feedback": review_feedback,
            "should_readjust_weights": should_retry,
            "iteration_count": iteration_count + 1,
            "workflow_status": "review_completed" if passed or not should_retry else "review_failed_retry",
        }

    return node


# Supervisor routing: Task Orchestration hub decides the next worker agent based on the latest workflow_status (and readjust signal after Review).
_SUPERVISOR_ROUTE_MAP: Dict[str, str] = {
    "task_orchestrated": "EnvironmentPerception",
    "perception_completed": "SuitabilityAssessment",
    "suitability_assessed": "MultiScenarioEvaluation",
    "weights_calculated": "PhasedDecisionMaking",
    "decision_made": "MemoryUpdate",
    "memory_updated": "Review",
}


# Map workflow_status to the next worker node name or end the run after Review.
def supervisor_route(state: RCPPAgentState) -> str:
    status = state.get("workflow_status", "init")
    if status in ("review_completed", "review_failed_retry"):
        if state.get("should_readjust_weights"):
            return "MultiScenarioEvaluation"
        return END
    target = _SUPERVISOR_ROUTE_MAP.get(status)
    if target is None:
        logger.warning("Supervisor received unknown workflow_status=%s; ending workflow.", status)
        return END
    return target


# Graph construction
def create_rcpp_workflow(
    task_orchestration_agent: TaskOrchestrationAgent,
    environment_perception_agent: EnvironmentPerceptionAgent,
    suitability_assessment_agent: SuitabilityAssessmentAgent,
    multi_scenario_evaluation_agent: MultiScenarioEvaluationAgent,
    phased_decision_making_agent: PhasedDecisionMakingAgent,
    review_agent: ReviewAgent,
    max_iterations: int,
    store: BaseStore,
    checkpointer: Any = None,
) -> Any:
    """
    Build and compile the workflow graph as a supervisor topology: the Task
    Orchestration Agent is the central hub; every worker agent returns to it,
    and the supervisor routes to the next worker by workflow_status.
    """
    workflow = StateGraph(RCPPAgentState)

    workflow.add_node("TaskOrchestration", _make_task_orchestration_node(task_orchestration_agent))
    workflow.add_node(
        "EnvironmentPerception",
        _make_environment_perception_node(environment_perception_agent, store),
    )
    workflow.add_node(
        "SuitabilityAssessment",
        _make_suitability_assessment_node(suitability_assessment_agent, store),
    )
    workflow.add_node(
        "MultiScenarioEvaluation",
        _make_multi_scenario_evaluation_node(multi_scenario_evaluation_agent, store),
    )
    workflow.add_node(
        "PhasedDecisionMaking",
        _make_phased_decision_making_node(phased_decision_making_agent, store),
    )
    workflow.add_node("MemoryUpdate", _make_memory_update_node(store))
    workflow.add_node("Review", _make_review_node(review_agent, max_iterations, store))

    workflow.set_entry_point("TaskOrchestration")

    workflow.add_conditional_edges(
        "TaskOrchestration",
        supervisor_route,
        {
            "EnvironmentPerception": "EnvironmentPerception",
            "SuitabilityAssessment": "SuitabilityAssessment",
            "MultiScenarioEvaluation": "MultiScenarioEvaluation",
            "PhasedDecisionMaking": "PhasedDecisionMaking",
            "MemoryUpdate": "MemoryUpdate",
            "Review": "Review",
            END: END,
        },
    )

    for worker in (
        "EnvironmentPerception",
        "SuitabilityAssessment",
        "MultiScenarioEvaluation",
        "PhasedDecisionMaking",
        "MemoryUpdate",
        "Review",
    ):
        workflow.add_edge(worker, "TaskOrchestration")

    return workflow.compile(checkpointer=checkpointer, store=store)


# Public workflow runtime used by the CLI.
class RCPPAgent:

    # Wire agents, checkpointing, and the compiled LangGraph workflow.
    def __init__(self, max_iterations: Optional[int] = None, checkpointer: Any = None):
        
        if checkpointer is not None:
            self._checkpointer = checkpointer
        else:
            db_path = default_checkpoint_db_path()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._checkpointer = SqliteSaver(conn)

        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else RCPP_AGENT_MAX_WORKFLOW_ITERATIONS
        )
        self.task_orchestration_agent = TaskOrchestrationAgent()
        self.environment_perception_agent = EnvironmentPerceptionAgent()
        self.suitability_assessment_agent = SuitabilityAssessmentAgent()
        self.multi_scenario_evaluation_agent = MultiScenarioEvaluationAgent()
        self.phased_decision_making_agent = PhasedDecisionMakingAgent()
        self.review_agent = ReviewAgent()
        self._long_term_store: BaseStore = Neo4jGlobalSceneSemanticStore()

        self.workflow = create_rcpp_workflow(
            self.task_orchestration_agent,
            self.environment_perception_agent,
            self.suitability_assessment_agent,
            self.multi_scenario_evaluation_agent,
            self.phased_decision_making_agent,
            self.review_agent,
            self.max_iterations,
            self._long_term_store,
            checkpointer=self._checkpointer,
        )
        logger.info("RCPP-Agent multi-agent architecture initialized.")

    _TERMINAL_STATUSES = {"review_completed", "finished"}

    def run(
        self,
        global_instruction: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the RCPP-Agent.

        If *thread_id* refers to an existing, non-terminal checkpoint the
        workflow resumes from where it left off.  Otherwise a
        fresh run is started from *global_instruction*.
        """
        tid = thread_id or uuid.uuid4().hex
        config = {"configurable": {"thread_id": tid}}

        existing = self.workflow.get_state(config)
        can_resume = (
            existing is not None
            and existing.values
            and existing.values.get("workflow_status") not in self._TERMINAL_STATUSES
        )

        if can_resume:
            logger.info("Resuming RCPP-Agent from checkpoint; thread_id=%s", tid)
            result_state = self.workflow.invoke(None, config=config)
        else:
            if global_instruction is None:
                raise ValueError(
                    "global_instruction is required when starting a new run "
                    "(no resumable checkpoint found for thread_id=%s)." % tid
                )
            logger.info("Starting the RCPP-Agent; thread_id=%s instruction=%s", tid, global_instruction)

            initial_state = {
                "global_instruction": global_instruction,
                "scenario": global_instruction.get("scenario", "equity_oriented"),
                "time_horizon": global_instruction.get("time_horizon"),
                "workspace_dir": global_instruction.get("workspace_dir"),
                "spatialite_context": {},
                "year": global_instruction.get("year", 2025),
                "suitable_rps": [],
                "suitability_evaluations": [],
                "scenario_mcda_weights": {},
                "mcda_weights": {},
                "phased_strategy": {},
                "scored_candidates": [],
                "review_result": None,
                "final_output": None,
                "review_feedback": "",
                "memory_update_events": [],
                "should_readjust_weights": False,
                "iteration_count": 0,
                "workflow_status": "init",
            }
            result_state = self.workflow.invoke(initial_state, config=config)

        final_output = {
            "status": result_state.get("workflow_status"),
            "iterations": result_state.get("iteration_count"),
            "phased_strategy": result_state.get("phased_strategy"),
            "scored_candidates": result_state.get("scored_candidates"),
            "mcda_weights": result_state.get("mcda_weights"),
            "review_result": result_state.get("review_result"),
            "thread_id": tid,
        }

        logger.info("RCPP-Agent finished.")
        return final_output

    def get_run_state(self, thread_id: str) -> Dict[str, Any]:
        """
        Return the latest checkpoint state for thread_id.
        """
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.workflow.get_state(config)
        return snapshot.values if snapshot else {}

    def get_run_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Return all checkpoint snapshots for *thread_id*.
        """
        config = {"configurable": {"thread_id": thread_id}}
        return list(self.workflow.get_state_history(config))
