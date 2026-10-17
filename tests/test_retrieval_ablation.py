import tempfile
from pathlib import Path
import unittest

from evaluation.retrieval_ablation import (
    JsonMemorySource,
    TokenCounter,
    cases_from_plan,
    run_experiment,
    run_policy,
)
from rcpp_core.memory_migration import MigrationPlan


def scene(**changes):
    value = {
        "has_existing_RCP": False,
        "is_functional_zone": False,
        "functional_zone_type": "none",
        "has_ground_obstacle": False,
        "ground_obstacle_types": "none",
        "ground_obstacle_count": 0,
        "visual_distractors_noted": ["none"],
        "clearance_visual_assessment": True,
        "scene_reasoning": "clear",
        "confidence_score": 5,
    }
    value.update(changes)
    return value


def semantic(rps_id):
    return {
        "rps_id": rps_id,
        "functional_zone_type": "mixed",
        "commuting_flow": "high",
        "sensitive_constraints": "no risk",
        "grid_accessibility": "good",
    }


class RetrievalAblationTests(unittest.TestCase):
    def setUp(self):
        self.plan = MigrationPlan(
            area="gulou",
            rps=[
                {"rps_id": "R1", "street_name": "A"},
                {"rps_id": "R2", "street_name": "A"},
                {"rps_id": "R3", "street_name": "B"},
            ],
            scene=[
                {"rps_id": "R1", "memory": scene(), "status": "active"},
                {"rps_id": "R2", "memory": scene(has_existing_RCP=True), "status": "active"},
                {"rps_id": "R3", "memory": scene(has_ground_obstacle=True,
                                                     ground_obstacle_types="tree_trunk",
                                                     ground_obstacle_count=1), "status": "active"},
            ],
            semantic=[
                {"rps_id": rps_id, "memory": semantic(rps_id), "status": "active"}
                for rps_id in ("R1", "R2", "R3")
            ],
            audit={},
        )
        self.source = JsonMemorySource(self.plan)
        self.cases = cases_from_plan(self.plan, "detailed-active")
        self.counter = TokenCounter()

    def test_progressive_skips_only_scene_failure(self):
        eager, _ = run_policy(policy="eager", repeat=1, source=self.source,
                              area="gulou", cases=self.cases, counter=self.counter)
        progressive, rows = run_policy(policy="progressive", repeat=1, source=self.source,
                                       area="gulou", cases=self.cases, counter=self.counter)
        self.assertEqual(eager.semantic_request_count, 3)
        self.assertEqual(progressive.semantic_request_count, 2)
        self.assertEqual(progressive.unnecessary_semantic_count, 0)
        self.assertEqual(progressive.semantic_route_precision, 1.0)
        self.assertEqual(progressive.semantic_route_recall, 1.0)
        self.assertFalse(next(row for row in rows if row["rps_id"] == "R2")["semantic_requested"])

    def test_experiment_writes_reproducible_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(
                source=self.source, cases=self.cases, area="gulou",
                output_dir=Path(directory), repeats=2, warmup=0,
                bootstrap_samples=20,
            )
            self.assertAlmostEqual(result["comparison"]["semantic_request_reduction"], 1 / 3)
            for name in ("rps_metrics.jsonl", "run_metrics.jsonl", "summary.json", "report.md"):
                self.assertTrue((Path(directory) / name).is_file())


if __name__ == "__main__":
    unittest.main()
