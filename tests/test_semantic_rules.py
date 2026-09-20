import unittest

from rcpp_core.semantic_rules import (
    deterministic_semantic_memory,
    derive_semantic_facts,
    validate_semantic_against_rules,
)


def summary(grid_km=0.2, residential=10, commercial=2, inflow=20_000, outflow=20_000, sensitive=0):
    return {
        "rps_id": "RPS-1",
        "poi_counts_1km": {"residential": residential, "commercial": commercial},
        "poi_counts_sensitive_0.1km": {"sensitive": sensitive},
        "traffic_flow_stats": {"total_inflow": inflow, "total_outflow": outflow},
        "infrastructure_dist": {"grid_distance_km": grid_km},
        "bsv_image": "one.jpg",
    }


class SemanticRuleTests(unittest.TestCase):
    def test_grid_boundaries(self):
        expected = {
            0.499: "excellent",
            0.5: "good",
            0.999: "good",
            1.0: "moderate",
            1.999: "moderate",
            2.0: "poor",
            4.999: "poor",
            5.0: "very_poor",
            -1: "unknown",
        }
        for distance, label in expected.items():
            with self.subTest(distance=distance):
                facts = derive_semantic_facts(summary(grid_km=distance))
                self.assertEqual(label, facts["labels"]["grid_accessibility"])

    def test_flow_boundaries(self):
        self.assertEqual("low", derive_semantic_facts(summary(inflow=25_000, outflow=24_999))["labels"]["commuting_flow"])
        self.assertEqual("medium", derive_semantic_facts(summary(inflow=25_000, outflow=25_000))["labels"]["commuting_flow"])
        self.assertEqual("medium", derive_semantic_facts(summary(inflow=50_000, outflow=50_000))["labels"]["commuting_flow"])
        self.assertEqual("high", derive_semantic_facts(summary(inflow=50_001, outflow=50_000))["labels"]["commuting_flow"])

    def test_sensitive_boundaries(self):
        self.assertEqual("none", derive_semantic_facts(summary(sensitive=0))["labels"]["sensitive_risk"])
        self.assertEqual("low", derive_semantic_facts(summary(sensitive=1))["labels"]["sensitive_risk"])
        self.assertEqual("high", derive_semantic_facts(summary(sensitive=2))["labels"]["sensitive_risk"])

    def test_deterministic_fallback_is_rule_consistent(self):
        source = summary()
        memory = deterministic_semantic_memory(source)
        self.assertEqual([], validate_semantic_against_rules(memory, source))
        self.assertIn("Dist=200m", memory["grid_accessibility"])

    def test_ungrounded_number_is_rejected(self):
        source = summary()
        memory = deterministic_semantic_memory(source)
        memory["grid_accessibility"] += " Estimated cost 999 yuan."
        self.assertTrue(
            any("ungrounded number 999" in error for error in validate_semantic_against_rules(memory, source))
        )


if __name__ == "__main__":
    unittest.main()
