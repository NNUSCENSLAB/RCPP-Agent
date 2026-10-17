import importlib.util
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rcpp_core.execution_journal import ExecutionJournal
from runtime_support import FakeRepository, scene, semantic


class JournalTests(unittest.TestCase):
    def test_committed_result_survives_reopen_and_changed_input_invalidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "journal.sqlite"
            calls = []
            compute = lambda: calls.append(1) or {"answer": len(calls)}
            ExecutionJournal(path).execute("scene/A", {"image_sha256": "one"}, compute)
            result, cached, _ = ExecutionJournal(path).execute("scene/A", {"image_sha256": "one"}, compute)
            self.assertTrue(cached)
            self.assertEqual(1, result["answer"])
            ExecutionJournal(path).execute("scene/A", {"image_sha256": "two"}, compute)
            self.assertEqual(2, len(calls))


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "requires LangGraph runtime dependencies")
class DurablePlanningTests(unittest.TestCase):
    def setUp(self):
        from rcpp_core.runtime import RCPPAgent, RCPPAgentState
        from rcpp_core.planning_workflow import compile_planning_graph, make_human_review_node
        self.RCPPAgent = RCPPAgent
        self.State = RCPPAgentState
        self.compile = compile_planning_graph
        self.human_node = make_human_review_node

    def test_hard_process_crash_reuses_inference_receipt_before_graph_write(self):
        root = Path(__file__).resolve().parents[1]
        worker = root / "tests" / "fixtures" / "runtime_fault_worker.py"
        with tempfile.TemporaryDirectory() as temporary:
            command = [sys.executable, str(worker), temporary]
            crashed = subprocess.run(command + ["crash"], cwd=root, capture_output=True, timeout=45)
            self.assertEqual(73, crashed.returncode, crashed.stderr.decode(errors="replace"))
            resumed = subprocess.run(command + ["resume"], cwd=root, capture_output=True, timeout=45)
            self.assertEqual(0, resumed.returncode, resumed.stderr.decode(errors="replace"))
            with closing(sqlite3.connect(Path(temporary) / "fake_domain.sqlite")) as conn:
                calls = conn.execute("SELECT kind,rid,n FROM calls ORDER BY kind,rid").fetchall()
            self.assertEqual(6, len(calls))
            self.assertTrue(all(n == 1 for _, _, n in calls), calls)

    def test_actual_sqlite_interrupt_reopen_review_and_resume(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        from src.environment_perception_agent import EnvironmentPerceptionAgent
        from rcpp_core.planning_workflow import validate_review
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "RCPP_RUN_ROOT": str(Path(temporary) / "runs"), "RCPP_ROUTING_PROFILES": "",
        }):
            repo = FakeRepository(Path(temporary) / "domain.sqlite")
            for rid in ("A", "C"):
                repo.append_scene_memory("test", rid, scene())
                repo.append_semantic_memory("test", rid, semantic(rid))
            agent = EnvironmentPerceptionAgent(scene_refresher=lambda *args: {},
                                               semantic_refresher=repo.worker("semantic"))
            seen = []
            def stage(name, extra=None):
                def node(state):
                    seen.append(name)
                    return extra or {}
                return node
            nodes = {name: stage(name) for name in (
                "SuitabilityAssessment", "MultiScenarioEvaluation", "PhasedDecisionMaking", "MemoryUpdate")}
            nodes.update({
                "TaskOrchestration": stage("TaskOrchestration", {"spatialite_context": {"area": "test"},
                                            "workflow_status": "task_orchestrated"}),
                "EnvironmentPerception": lambda state: agent.perceive(
                    state["spatialite_context"], state["global_instruction"], repository=repo),
                "HumanReview": self.human_node(repo),
                "Review": stage("Review", {"should_readjust_weights": False, "workflow_status": "review_completed"}),
            })
            path = Path(temporary) / "checkpoints.sqlite"
            def runtime(conn):
                instance = self.RCPPAgent.__new__(self.RCPPAgent)
                instance.max_iterations = 3
                instance.workflow = self.compile(self.State, nodes, checkpointer=SqliteSaver(conn))
                return instance
            with closing(sqlite3.connect(path, check_same_thread=False)) as conn:
                result = runtime(conn).run({"run_id": "review-run"}, thread_id="review-run")
            self.assertEqual("awaiting_review", result["status"])
            self.assertNotIn("SuitabilityAssessment", seen)
            pending = result["pending_reviews"][0]
            response = {"review_token": pending["review_token"], "reviewer": "tester",
                        "decisions": [{"rps_id": "B", "action": "exclude", "note": "image unavailable"}]}
            with self.assertRaises(ValueError):
                validate_review({**response, "review_token": "stale"}, pending["pending"], pending["review_token"])
            with closing(sqlite3.connect(path, check_same_thread=False)) as conn:
                result = runtime(conn).run(thread_id="review-run", review_response=response)
            self.assertEqual("review_completed", result["status"])
            self.assertEqual(1, seen.count("TaskOrchestration"))
            self.assertEqual(1, seen.count("SuitabilityAssessment"))
            with closing(sqlite3.connect(repo.path)) as conn:
                queries = conn.execute("SELECT ids FROM queries WHERE kind='semantic'").fetchall()
            self.assertTrue(all("B" not in json.loads(ids) for (ids,) in queries))

    def test_review_loop_revises_weights_without_repeating_upstream(self):
        from langgraph.checkpoint.memory import InMemorySaver
        seen = []
        def node(name):
            def call(state):
                seen.append(name)
                if name == "Review":
                    n = state.get("iteration_count", 0) + 1
                    return {"iteration_count": n, "should_readjust_weights": n < 3}
                return {}
            return call
        names = ("TaskOrchestration", "EnvironmentPerception", "HumanReview", "SuitabilityAssessment",
                 "MultiScenarioEvaluation", "PhasedDecisionMaking", "MemoryUpdate", "Review")
        graph = self.compile(self.State, {name: node(name) for name in names}, checkpointer=InMemorySaver())
        result = graph.invoke({"iteration_count": 0}, {"configurable": {"thread_id": "loop"}})
        self.assertEqual(3, result["iteration_count"])
        self.assertEqual(1, seen.count("TaskOrchestration"))
        self.assertEqual(1, seen.count("EnvironmentPerception"))
        self.assertEqual(1, seen.count("SuitabilityAssessment"))
        self.assertEqual(3, seen.count("MultiScenarioEvaluation"))

    def test_manual_scene_correction_rejects_partial_evidence(self):
        from rcpp_core.planning_workflow import validate_review
        pending = [{"rps_id": "A"}]
        response = {"review_token": "token", "reviewer": "tester", "decisions": [
            {"rps_id": "A", "action": "correct_scene", "note": "reviewed", "scene_memory": scene()}]}
        with self.assertRaises(ValueError):
            validate_review(response, pending, "token")
        response["decisions"][0]["source_images"] = ["photo.jpg"]
        self.assertEqual(1, len(validate_review(response, pending, "token")))

    def test_corrected_scene_is_versioned_then_semantic_is_retrieved(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import StateGraph, END
        from langgraph.types import Command
        from src.environment_perception_agent import EnvironmentPerceptionAgent
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "RCPP_RUN_ROOT": str(Path(temporary) / "runs"), "RCPP_ROUTING_PROFILES": "",
        }):
            repo = FakeRepository(Path(temporary) / "domain.sqlite")
            graph = StateGraph(self.State)
            graph.add_node("HumanReview", self.human_node(repo))
            graph.set_entry_point("HumanReview")
            graph.add_edge("HumanReview", END)
            workflow = graph.compile(checkpointer=InMemorySaver())
            config = {"configurable": {"thread_id": "correction"}}
            state = {"global_instruction": {"run_id": "correction"}, "spatialite_context": {"area": "test"},
                     "pending_scene_reviews": [{"rps_id": "A", "reasons": ["invalid scene"]}]}
            result = workflow.invoke(state, config)
            pending = result["__interrupt__"][0].value
            response = {"review_token": pending["review_token"], "reviewer": "test", "decisions": [
                {"rps_id": "A", "action": "correct_scene", "note": "photo checked", "scene_memory": scene(),
                 "source_images": ["A.jpg"]}]}
            result = workflow.invoke(Command(resume=response), config)
            current = repo.get_current_scene("test", ["A"])["A"]
            self.assertTrue(current["_meta"]["memory_id"].startswith("manual-"))
            agent = EnvironmentPerceptionAgent(scene_refresher=repo.worker("scene"),
                                               semantic_refresher=repo.worker("semantic"))
            agent.perceive({"area": "test"}, result["global_instruction"], repository=repo)
            self.assertIn("A", repo.get_current_semantic("test", ["A"]))
