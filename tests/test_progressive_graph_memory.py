import json
from pathlib import Path
import tempfile
import unittest

from rcpp_core.graph_memory_repository import GraphMemoryRepository, assess_scene, assess_semantic, stable_id
from rcpp_core.memory_migration import parse_semantic_facts, prepare_migration


def valid_scene(**overrides):
    value = {
        "has_existing_RCP": False, "is_functional_zone": False,
        "functional_zone_type": "none", "has_ground_obstacle": False,
        "ground_obstacle_types": "none", "ground_obstacle_count": 0,
        "visual_distractors_noted": ["none"], "clearance_visual_assessment": True,
        "scene_reasoning": "clear", "confidence_score": 5,
    }
    value.update(overrides)
    return value


class InMemoryProgressiveRepository(GraphMemoryRepository):
    def __init__(self, scenes, semantics):
        self.scenes = scenes
        self.semantics = semantics
        self.semantic_queries = []

    def get_current_scene(self, area, rps_ids=None):
        return {k: v for k, v in self.scenes.items() if not rps_ids or k in rps_ids}

    def get_current_semantic(self, area, rps_ids=None):
        self.semantic_queries.append(list(rps_ids or []))
        return {k: v for k, v in self.semantics.items() if not rps_ids or k in rps_ids}

    def list_rps(self, area):
        return [{"rps_id": key, "longitude": 1, "latitude": 2, "street_name": "road"} for key in self.scenes]


class ProgressiveGraphMemoryTests(unittest.TestCase):
    def test_scene_conditions_allow_obstacles_when_clearance_is_true(self):
        result = assess_scene(valid_scene(has_ground_obstacle=True,
                                          ground_obstacle_types="tree_trunk",
                                          ground_obstacle_count=1))
        self.assertTrue(result.allows_semantic)

    def test_scene_failure_is_not_sent_to_semantic_query(self):
        repo = InMemoryProgressiveRepository(
            {"PASS": valid_scene(), "FAIL": valid_scene(has_existing_RCP=True)},
            {"PASS": {"functional_zone_type": "x"}},
        )
        _, events = repo.load_progressive_context("gulou")
        self.assertEqual(repo.semantic_queries, [["PASS"]])
        self.assertTrue(any(x["rps_id"] == "FAIL" and x["action"] == "stop" for x in events))

    def test_missing_scene_field_never_passes(self):
        scene = valid_scene()
        del scene["is_functional_zone"]
        result = assess_scene(scene)
        self.assertEqual(result.state, "partial")
        self.assertFalse(result.allows_semantic)

    def test_memory_identity_uses_payload_not_adapter(self):
        first = stable_id("scene", "gulou", "A", valid_scene())
        second = stable_id("scene", "gulou", "A", valid_scene(scene_reasoning="changed"))
        self.assertNotEqual(first, second)

    def test_partial_semantic_is_refreshed(self):
        result = assess_semantic({"rps_id": "R1", "functional_zone_type": "x"})
        self.assertEqual(result.state, "partial")

    def test_semantic_numbers_are_parsed_from_original_text(self):
        parsed = parse_semantic_facts({
            "functional_zone_type": "P_{res}=2, P_{comm}=3",
            "commuting_flow": "Total: 105294, Inflow: 40357, Outflow: 64937",
            "sensitive_constraints": "Risk Warning: No Risk.",
            "grid_accessibility": "Dist=7174.0m",
        }, "R1")
        self.assertEqual(parsed["traffic_total"], 105294)
        self.assertEqual(parsed["grid_distance_m"], 7174.0)
        self.assertEqual(parsed["residential_poi_count"], 2)

    def test_migration_uses_image_name_and_keeps_unmatched_in_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = [
                {"image": "/raw/a.jpg", "conversations": [{"from": "gpt", "value": json.dumps(valid_scene())}]},
                {"image": "/raw/unmatched.jpg", "conversations": [{"from": "gpt", "value": json.dumps(valid_scene())}]},
            ]
            combined = [
                {"RPS_id": "R1", "coordinates": [1, 2], "street_view_images": ["/other/a.jpg"],
                 "memory_node": {"scene_memory": {"clearance_visual_assessment": "True", "scene_reasoning": "x"},
                                 "semantic_memory": {"functional_zone_type": "x", "commuting_flow": "Total: 1, Inflow: 1, Outflow: 0", "sensitive_constraints": "Risk Warning: No Risk.", "grid_accessibility": "Dist=100m"}}},
                {"RPS_id": "R2", "coordinates": [3, 4], "street_view_images": ["/other/b.jpg"],
                 "memory_node": {"scene_memory": {"clearance_visual_assessment": "False", "scene_reasoning": "x"},
                                 "semantic_memory": {"functional_zone_type": "x", "commuting_flow": "Total: 1, Inflow: 1, Outflow: 0", "sensitive_constraints": "Risk Warning: No Risk.", "grid_accessibility": "Dist=100m"}}},
            ]
            scene_path, combined_path = root / "scene.json", root / "combined.json"
            scene_path.write_text(json.dumps(raw), encoding="utf-8")
            combined_path.write_text(json.dumps(combined), encoding="utf-8")
            plan = prepare_migration("gulou", scene_path, combined_path)
            self.assertEqual(plan.audit["detailed_scene_count"], 1)
            self.assertEqual(plan.audit["partial_scene_count"], 1)
            self.assertEqual(plan.audit["unmatched_raw_scene_count"], 1)

    def test_runtime_has_no_legacy_aggregate_access(self):
        root = Path(__file__).resolve().parents[1]
        runtime = (root / "rcpp_core" / "runtime.py").read_text(encoding="utf-8")
        perception = (root / "src" / "environment_perception_agent.py").read_text(encoding="utf-8")
        self.assertNotIn("GlobalSceneSemanticMemory", runtime + perception)
        self.assertNotIn("SCENE_SEMANTIC_KEY_GLOBAL", runtime + perception)

    def test_repository_cypher_enforces_current_replacement_and_idempotent_merge(self):
        source = (Path(__file__).resolve().parents[1] / "rcpp_core" /
                  "graph_memory_repository.py").read_text(encoding="utf-8")
        self.assertIn("MERGE (m:{label} {{memory_id:$memory_id}})", source)
        self.assertIn("old.is_current=false", source)
        self.assertIn("MERGE (m)-[:SUPERSEDES]->(old)", source)


if __name__ == "__main__":
    unittest.main()
