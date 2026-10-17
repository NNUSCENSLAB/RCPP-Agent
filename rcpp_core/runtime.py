"""
RCPP-Agent: multi-agent roadside charging pile (RCP) planning with LangGraph orchestration.

It chains task orchestration, environment perception, suitability assessment, scenario
weighting (MCDA), phased decision-making, Neo4j-backed memory update, and review.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict, cast, TYPE_CHECKING

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

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from tools.rcpp_paths import default_poi_dir
from rcpp_core.graph_memory_repository import GraphMemoryRepository
from rcpp_core.execution_journal import exclusive_file, run_directory
from rcpp_core.planning_workflow import WORKFLOW_VERSION, compile_planning_graph, make_human_review_node

if TYPE_CHECKING:
    from src.task_orchestration_agent import TaskOrchestrationAgent
    from src.environment_perception_agent import EnvironmentPerceptionAgent
    from src.suitability_assessment_agent import SuitabilityAssessmentAgent
    from src.multi_scenario_evaluation_agent import MultiScenarioEvaluationAgent
    from src.phased_decision_making_agent import PhasedDecisionMakingAgent
    from src.review_agent import ReviewAgent

logger = logging.getLogger(__name__)

_MEMORY_EVENTS_CLEAR_SOURCE = "__memory_events_clear__"
_PHASED_PRIORITY_SOURCE = "PhasedDecisionMakingAgent"
_PHASED_PRIORITY_FIELD = "priority_score"


def _area_slug(spatialite_context: Dict[str, Any]) -> str:
    area = str((spatialite_context or {}).get("area") or "").strip().lower()
    if not area:
        raise ValueError("spatialite_context['area'] is required")
    return area


def _load_progressive_memory(repository: GraphMemoryRepository, area: str,
                             excluded: List[str] | None = None) -> Dict[str, Any]:
    ids = [r["rps_id"] for r in repository.list_rps(area) if r["rps_id"] not in set(excluded or [])]
    if not ids:
        return {}
    memory, events = repository.load_progressive_context(area, ids)
    ready = {event["rps_id"] for event in events
             if event.get("layer") == "semantic" and event.get("action") == "reuse"}
    return {rps_id: record for rps_id, record in memory.items() if rps_id in ready}

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
    retrieval_events: List[Dict[str, Any]]
    retrieval_summary: Dict[str, int]
    pending_scene_reviews: List[Dict[str, Any]]
    workflow_version: int

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
    retrieval_events: List[Dict[str, Any]]
    retrieval_summary: Dict[str, int]
    pending_scene_reviews: List[Dict[str, Any]]
    workflow_version: int
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
    """Parse and initialize once; explicit graph edges determine subsequent stages."""
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Task Orchestration ---")
        status = state.get("workflow_status", "init")
        if status != "init":
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
def _make_environment_perception_node(agent: EnvironmentPerceptionAgent, repository: GraphMemoryRepository):
    # Load or build scene-semantic memory from spatialite context and instruction.
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Environment Perception Agent ---")
        spatialite_context = state.get("spatialite_context", {})
        global_instruction = state.get("global_instruction") or {}
        global_instruction = {**global_instruction,
                              "excluded_rps": global_instruction.get("excluded_rps", [])}
        return cast(
            RCPPAgentStateUpdate,
            agent.perceive(spatialite_context, global_instruction, repository=repository),
        )

    return node


# Build the Suitability Assessment graph node callable.
def _make_suitability_assessment_node(agent: SuitabilityAssessmentAgent, repository: GraphMemoryRepository):
    # Filter RPS sites, emit suitability memory events, and set workflow status.
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Suitability Assessment Agent ---")
        from src.suitability_assessment_agent import load_rps_gdf_for_export, export_suitability_shapefiles
        ctx = state.get("spatialite_context") or {}
        area = _area_slug(ctx)
        memories = _load_progressive_memory(repository, area, (state.get("global_instruction") or {}).get("excluded_rps"))

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

        repository.append_decisions(str((state.get("global_instruction") or {}).get("run_id") or "unknown"), [
            {"area": area, "rps_id": event["rps_id"],
             "outcome": event["field"], "reason": str(event["value"]),
             "rules": ["suitability_assessment"], "fact_ids": []}
            for event in events
        ])

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
def _make_multi_scenario_evaluation_node(agent: MultiScenarioEvaluationAgent, repository: GraphMemoryRepository):
    """
    Load scene-semantic store after suitability write; compute MCDA weights via LLM-AHP.
    """
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Multi-Scenario Evaluation Agent ---")
        from src.phased_decision_making_agent import write_json_path
        scenario = state.get("scenario", "equity_oriented")
        iteration = state.get("iteration_count", 0)

        ctx = state.get("spatialite_context") or {}
        area = _area_slug(ctx)
        memories = _load_progressive_memory(repository, area, (state.get("global_instruction") or {}).get("excluded_rps"))
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
def _make_phased_decision_making_node(agent: PhasedDecisionMakingAgent, repository: GraphMemoryRepository):
    # Score candidates, select quota, emit priority memory events, and set status.
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Phased Decision-Making Agent ---")
        from src.suitability_assessment_agent import load_rps_gdf_for_export, guess_rps_id_column
        from src.phased_decision_making_agent import write_json_path, export_phase_rps_shapefile
        suitable_rps = state.get("suitable_rps", [])
        ctx = state.get("spatialite_context") or {}
        area = _area_slug(ctx)
        memories = _load_progressive_memory(repository, area, (state.get("global_instruction") or {}).get("excluded_rps"))
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
def _make_memory_update_node(repository: GraphMemoryRepository):
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
        area = _area_slug(ctx)
        run_id = str((state.get("global_instruction") or {}).get("run_id") or "unknown")
        repository.append_decisions(run_id, [
            {"area": area, "rps_id": event["rps_id"], "outcome": "priority_scored",
             "reason": str(event["value"]), "rules": ["phased_priority"], "fact_ids": []}
            for event in events
        ])
        logger.info("Memory update: persisted %s phased decisions (of %s raw).", len(events), len(raw))

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
def _make_review_node(agent: ReviewAgent, max_iterations: int, repository: GraphMemoryRepository):
    # Run metrics-based review, set retry flags, and bump iteration count.
    def node(state: RCPPAgentState) -> RCPPAgentStateUpdate:
        logger.info("--- [Node] Review Agent ---")
        from src.review_agent import write_review_metrics_json
        strategy = state.get("phased_strategy", {})
        candidates = state.get("scored_candidates", [])
        iteration_count = state.get("iteration_count", 0)

        ctx = state.get("spatialite_context") or {}
        area = _area_slug(ctx)
        memories = _load_progressive_memory(repository, area, (state.get("global_instruction") or {}).get("excluded_rps"))
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


# Graph construction
def create_rcpp_workflow(
    task_orchestration_agent: TaskOrchestrationAgent,
    environment_perception_agent: EnvironmentPerceptionAgent,
    suitability_assessment_agent: SuitabilityAssessmentAgent,
    multi_scenario_evaluation_agent: MultiScenarioEvaluationAgent,
    phased_decision_making_agent: PhasedDecisionMakingAgent,
    review_agent: ReviewAgent,
    max_iterations: int,
    repository: GraphMemoryRepository,
    checkpointer: Any = None,
) -> Any:
    """Compile direct stage edges; branch only for human review and outcome revision."""
    return compile_planning_graph(RCPPAgentState, {
        "TaskOrchestration": _make_task_orchestration_node(task_orchestration_agent),
        "EnvironmentPerception": _make_environment_perception_node(environment_perception_agent, repository),
        "HumanReview": make_human_review_node(repository),
        "SuitabilityAssessment": _make_suitability_assessment_node(suitability_assessment_agent, repository),
        "MultiScenarioEvaluation": _make_multi_scenario_evaluation_node(multi_scenario_evaluation_agent, repository),
        "PhasedDecisionMaking": _make_phased_decision_making_node(phased_decision_making_agent, repository),
        "MemoryUpdate": _make_memory_update_node(repository),
        "Review": _make_review_node(review_agent, max_iterations, repository),
    }, checkpointer=checkpointer)


# Public workflow runtime used by the CLI.
class RCPPAgent:

    # Wire agents, checkpointing, and the compiled LangGraph workflow.
    def __init__(self, max_iterations: Optional[int] = None, checkpointer: Any = None):
        from src.task_orchestration_agent import TaskOrchestrationAgent
        from src.environment_perception_agent import EnvironmentPerceptionAgent
        from src.suitability_assessment_agent import SuitabilityAssessmentAgent
        from src.multi_scenario_evaluation_agent import MultiScenarioEvaluationAgent
        from src.phased_decision_making_agent import PhasedDecisionMakingAgent
        from src.review_agent import ReviewAgent
        if checkpointer is not None:
            self._checkpointer = checkpointer
        else:
            db_path = default_checkpoint_db_path()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._checkpointer = SqliteSaver(self._checkpoint_conn)

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
        self._memory_repository = GraphMemoryRepository()

        self.workflow = create_rcpp_workflow(
            self.task_orchestration_agent,
            self.environment_perception_agent,
            self.suitability_assessment_agent,
            self.multi_scenario_evaluation_agent,
            self.phased_decision_making_agent,
            self.review_agent,
            self.max_iterations,
            self._memory_repository,
            checkpointer=self._checkpointer,
        )
        logger.info("RCPP-Agent multi-agent architecture initialized.")

    _TERMINAL_STATUSES = {"review_completed", "finished"}

    def run(
        self,
        global_instruction: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        review_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the RCPP-Agent.

        If *thread_id* refers to an existing, non-terminal checkpoint the
        workflow resumes from where it left off.  Otherwise a
        fresh run is started from *global_instruction*.
        """
        tid = thread_id or uuid.uuid4().hex
        with exclusive_file(run_directory(tid) / "runtime.lock"):
            return self._run_locked(global_instruction, tid, review_response)

    def _run_locked(self, global_instruction, tid, review_response):
        config = {"configurable": {"thread_id": tid}}
        config["recursion_limit"] = max(100, 20 * (self.max_iterations + 1))

        existing = self.workflow.get_state(config)
        can_resume = (
            existing is not None
            and existing.values
            and existing.values.get("workflow_status") not in self._TERMINAL_STATUSES
        )

        if can_resume:
            if existing.values.get("workflow_version") != WORKFLOW_VERSION:
                raise ValueError("Checkpoint uses the previous graph topology; start a new run_id")
            logger.info("Resuming RCPP-Agent from checkpoint; thread_id=%s", tid)
            result_state = self.workflow.invoke(
                Command(resume=review_response) if review_response is not None else None, config=config)
        else:
            if review_response is not None:
                raise ValueError("No pending workflow is available for this review response")
            if global_instruction is None:
                raise ValueError(
                    "global_instruction is required when starting a new run "
                    "(no resumable checkpoint found for thread_id=%s)." % tid
                )
            global_instruction = dict(global_instruction)
            global_instruction.setdefault("run_id", tid)
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
                "retrieval_events": [],
                "retrieval_summary": {},
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
                "workflow_version": WORKFLOW_VERSION,
                "pending_scene_reviews": [],
            }
            result_state = self.workflow.invoke(initial_state, config=config)

        snapshot = self.workflow.get_state(config)
        pending_interrupts = [i.value for task in snapshot.tasks for i in task.interrupts]
        final_output = {
            "status": "awaiting_review" if pending_interrupts else result_state.get("workflow_status"),
            "pending_reviews": pending_interrupts,
            "iterations": result_state.get("iteration_count"),
            "phased_strategy": result_state.get("phased_strategy"),
            "scored_candidates": result_state.get("scored_candidates"),
            "mcda_weights": result_state.get("mcda_weights"),
            "review_result": result_state.get("review_result"),
            "retrieval_events": result_state.get("retrieval_events", []),
            "retrieval_summary": result_state.get("retrieval_summary", {}),
            "thread_id": tid,
        }

        logger.info("RCPP-Agent finished.")
        return final_output

    def close(self):
        connection = getattr(self, "_checkpoint_conn", None)
        if connection is not None:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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
