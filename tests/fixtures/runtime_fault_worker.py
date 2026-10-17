"""Killable worker exercising the actual LangGraph and receipt integration."""
from pathlib import Path
from contextlib import closing
import os
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
directory = Path(sys.argv[1])
os.environ["RCPP_RUN_ROOT"] = str(directory / "runs")
os.environ["RCPP_ROUTING_PROFILES"] = ""
os.environ["RCPPAGENT_WORKSPACE"] = str(directory)

from langgraph.checkpoint.sqlite import SqliteSaver
from rcpp_core.runtime import RCPPAgentState
from rcpp_core.planning_workflow import compile_planning_graph, make_human_review_node
from src.environment_perception_agent import EnvironmentPerceptionAgent
from runtime_support import FakeRepository

repo = FakeRepository(directory / "fake_domain.sqlite", crash=sys.argv[2] == "crash")
agent = EnvironmentPerceptionAgent(scene_refresher=repo.worker("scene"), semantic_refresher=repo.worker("semantic"))
nodes = {name: (lambda state: {}) for name in (
    "SuitabilityAssessment", "MultiScenarioEvaluation", "PhasedDecisionMaking", "MemoryUpdate")}
nodes.update({
    "TaskOrchestration": lambda state: {"spatialite_context": {"area": "test"}},
    "EnvironmentPerception": lambda state: agent.perceive(state["spatialite_context"], state["global_instruction"], repository=repo),
    "HumanReview": make_human_review_node(repo),
    "Review": lambda state: {"should_readjust_weights": False, "workflow_status": "review_completed"},
})
with closing(sqlite3.connect(directory / "checkpoints.sqlite", check_same_thread=False)) as conn:
    workflow = compile_planning_graph(RCPPAgentState, nodes, checkpointer=SqliteSaver(conn))
    config = {"configurable": {"thread_id": "fault-test"}}
    state = workflow.get_state(config)
    initial = None if state.values else {"global_instruction": {"run_id": "fault-test"}}
    result = workflow.invoke(initial, config=config)
    assert result["workflow_status"] == "review_completed"
