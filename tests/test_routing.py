import unittest

from rcpp_core.routing import MemoryAwareRouter, RoutingContext


class RoutingTests(unittest.TestCase):
    def test_fresh_high_quality_memory_bypasses_model(self):
        decision = MemoryAwareRouter().decide(
            RoutingContext("run-1", "semantic", memory_state="fresh", memory_quality=0.95)
        )
        self.assertEqual("reuse_memory", decision.action)
        self.assertIsNone(decision.production_model)

    def test_partial_memory_uses_incremental_inference(self):
        decision = MemoryAwareRouter().decide(
            RoutingContext("run-2", "semantic", memory_state="partial", memory_quality=0.7)
        )
        self.assertEqual("incremental_inference", decision.action)

    def test_scene_route_forbids_text_fallback(self):
        decision = MemoryAwareRouter().decide(
            RoutingContext("run-3", "scene", memory_state="conflicted", request_risk="high")
        )
        self.assertIn("no_text_only_fallback", decision.safeguards)
        self.assertIn("human_review_on_second_failure", decision.safeguards)


if __name__ == "__main__":
    unittest.main()
