from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rcpp_core.model_registry import ModelDeployment
from rcpp_core.routing import MemoryAwareRouter, ModelProfile, RoutingContext


class BudgetRoutingTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "cheap": ModelDeployment("cheap", "scene", "vendor/cheap", stage="production"),
            "fast": ModelDeployment("fast", "scene", "vendor/fast", stage="production"),
            "shadow": ModelDeployment("shadow", "scene", "vendor/shadow", stage="shadow"),
            "text": ModelDeployment("text", "semantic", "vendor/text", stage="production"),
        }
        self.profiles = {
            "cheap": ModelProfile(5000, 0.01, 0.97, 100, "2026-01-01T00:00:00Z", True),
            "fast": ModelProfile(1000, 0.02, 0.96, 100, "2026-01-01T00:00:00Z", True),
            "shadow": ModelProfile(100, 0.001, 0.99, 100, "2026-01-01T00:00:00Z", True),
            "text": ModelProfile(100, 0.001, 0.99, 100, "2026-01-01T00:00:00Z", True),
        }
        self.profiles = {key: replace(profile, base_model=self.registry[key].base_model,
                                       adapter_version=self.registry[key].adapter_version)
                         for key, profile in self.profiles.items()}

    def test_latency_budget_changes_executed_model(self):
        router = MemoryAwareRouter(self.profiles, self.registry)
        relaxed = router.decide(RoutingContext("r", "scene", latency_budget_ms=6000))
        tight = router.decide(RoutingContext("r", "scene", latency_budget_ms=2000))
        self.assertEqual("cheap", relaxed.production_model)
        self.assertEqual("fast", tight.production_model)
        self.assertTrue(any("P95 latency exceeds budget" in c["reasons"] for c in tight.candidates))

    def test_unreachable_budget_escalates_and_never_selects_shadow_or_text(self):
        route = MemoryAwareRouter(self.profiles, self.registry).decide(
            RoutingContext("r", "scene", cost_budget=0.001))
        self.assertEqual("human_review", route.action)
        self.assertIsNone(route.production_model)
        self.assertNotIn("text", [c["model"] for c in route.candidates])

    def test_missing_measurements_use_safe_fallback(self):
        router = MemoryAwareRouter({}, self.registry)
        self.assertEqual("human_review", router.decide(RoutingContext("r", "scene", cost_budget=1)).action)
        self.assertEqual("rule_fallback", router.decide(RoutingContext("r", "semantic", cost_budget=1)).action)

    def test_risk_threshold_and_unknown_quality_cannot_be_bought_away(self):
        profiles = {key: replace(p, quality=0.92) for key, p in self.profiles.items()}
        route = MemoryAwareRouter(profiles, self.registry).decide(
            RoutingContext("r", "scene", request_risk="high", cost_budget=100))
        self.assertEqual("human_review", route.action)

    def test_invalid_budgets_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RoutingContext("r", "scene", cost_budget=value)

    def test_profile_for_another_adapter_is_rejected(self):
        profiles = {key: replace(profile, adapter_version="old-adapter")
                    for key, profile in self.profiles.items()}
        route = MemoryAwareRouter(profiles, self.registry).decide(RoutingContext("r", "scene", cost_budget=1))
        self.assertEqual("human_review", route.action)

    def test_fresh_memory_avoids_model_and_shadow_even_under_zero_budget(self):
        route = MemoryAwareRouter(self.profiles, self.registry).decide(
            RoutingContext("r", "scene", memory_state="fresh", memory_quality=1,
                           cost_budget=0, shadow_enabled=True))
        self.assertEqual("reuse_memory", route.action)
        self.assertIsNone(route.production_model)
        self.assertIsNone(route.shadow_model)

    def test_selected_model_reaches_refresh_worker_with_shadow_disabled(self):
        from src.environment_perception_agent import EnvironmentPerceptionAgent
        from runtime_support import FakeRepository
        from rcpp_core.model_registry import MODEL_REGISTRY
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
            "RCPP_RUN_ROOT": str(Path(temporary) / "runs"), "RCPP_SHADOW_ENABLED": "true",
        }), patch.dict(MODEL_REGISTRY, self.registry):
            repo = FakeRepository(Path(temporary) / "domain.sqlite")
            contexts = []
            def refresh(area, ids, context):
                contexts.append(context)
                return repo.worker("scene")(area, ids, context)
            agent = EnvironmentPerceptionAgent(scene_refresher=refresh,
                semantic_refresher=repo.worker("semantic"),
                router=MemoryAwareRouter(self.profiles, self.registry))
            agent.perceive({"area": "test"}, {"run_id": "budgeted", "latency_budget_ms": 2000}, repository=repo)
            self.assertEqual(["fast"] * 3, [c["model_key"] for c in contexts])
            self.assertTrue(all(c["shadow_enabled"] is False for c in contexts))

    def test_inference_tool_respects_explicit_shadow_disable(self):
        from tools.memory_construction_tool.memory_inference_tool import MemoryInferenceTool
        module = "tools.memory_construction_tool.memory_inference_tool"
        with patch.dict(os.environ, {"RCPP_SHADOW_ENABLED": "true"}), \
                patch(module + ".run_scene_inference_file") as infer, \
                patch(module + ".shadow_deployment") as shadow:
            MemoryInferenceTool().run(kind="scene", input_file="input.json", output_file="output.json",
                                      shadow_enabled=False)
            infer.assert_called_once()
            shadow.assert_not_called()

    def test_custom_weights_cannot_reuse_default_model_profile(self):
        from src.environment_perception_agent import EnvironmentPerceptionAgent
        from rcpp_core.model_registry import get_deployment
        default = get_deployment("scene")
        with patch.dict(os.environ, {"RCPP_ROUTING_PROFILES": ""}):
            agent = EnvironmentPerceptionAgent(scene_base_model="vendor/custom", scene_adapter_path="")
        agent.router.profiles = {default.key: ModelProfile(
            1000, 0.01, 0.99, 100, "2026-01-01T00:00:00Z", True,
            default.base_model, default.adapter_version)}
        route = agent.router.decide(RoutingContext("r", "scene", cost_budget=1))
        self.assertEqual("human_review", route.action)
