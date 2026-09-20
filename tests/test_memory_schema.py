import unittest

from rcpp_core.memory_schema import validate_scene_memory, validate_semantic_memory


class MemorySchemaTests(unittest.TestCase):
    def test_scene_detects_fixed_or_invalid_confidence(self):
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
            "confidence_score": 6,
        }
        self.assertIn("confidence_score must be between 1 and 5", validate_scene_memory(value))
        value["confidence_score"] = 4
        self.assertEqual([], validate_scene_memory(value))

    def test_semantic_rejects_null_placeholder(self):
        value = {
            "rps_id": "1",
            "functional_zone_type": "null",
            "commuting_flow": "Low Traffic",
            "sensitive_constraints": "No Risk",
            "grid_accessibility": "Good",
        }
        self.assertTrue(validate_semantic_memory(value))


if __name__ == "__main__":
    unittest.main()
