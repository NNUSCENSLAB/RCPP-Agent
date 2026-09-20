import unittest

from evaluation.evaluate_memory import canonical_semantic, dataset_primary


class EvaluationMetricTests(unittest.TestCase):
    def test_legacy_label_aliases_match_canonical_labels(self):
        self.assertEqual("none", canonical_semantic({"sensitive_constraints": "No Risk"}, "sensitive_constraints"))
        self.assertEqual("moderate", canonical_semantic({"grid_accessibility": "General: Dist=1200m"}, "grid_accessibility"))

    def test_perfect_scene_primary_is_one(self):
        gold = [{"ground_obstacle_types": "none", "clearance_visual_assessment": True}]
        self.assertEqual(1.0, dataset_primary("scene", gold, list(gold)))


if __name__ == "__main__":
    unittest.main()
