import json
from pathlib import Path
import tempfile
import unittest

from evaluation.qwen3_route_ablation import run_experiment


class Qwen3RouteAblationTests(unittest.TestCase):
    def test_progressive_policy_skips_semantic_for_failed_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = root / "scene_calls.jsonl"
            combined = root / "combined.json"
            output = root / "output"
            records = [
                {
                    "rps_id": "RPS-1", "street_name": "A", "success": True,
                    "prediction": {
                        "clearance_visual_assessment": True,
                        "has_existing_RCP": False,
                        "is_functional_zone": False,
                    },
                },
                {
                    "rps_id": "RPS-2", "street_name": "B", "success": True,
                    "prediction": {
                        "clearance_visual_assessment": False,
                        "has_existing_RCP": False,
                        "is_functional_zone": False,
                    },
                },
            ]
            calls.write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
            combined.write_text(json.dumps([
                {
                    "RPS_id": "RPS-1", "street_name": "A",
                    "memory_node": {
                        "scene_memory": {"clearance_visual_assessment": "True"},
                        "semantic_memory": {"commuting_flow": "high"},
                    },
                },
                {
                    "RPS_id": "RPS-2", "street_name": "B",
                    "memory_node": {
                        "scene_memory": {"clearance_visual_assessment": "False"},
                        "semantic_memory": {"commuting_flow": "low"},
                    },
                },
            ]), encoding="utf-8")
            (root / "summary.json").write_text(json.dumps({"metrics": {
                "selected_case_count": 2, "success_count": 2, "failure_count": 0,
                "json_schema_success_rate": 1.0, "request_count": 2,
                "total_tokens": 100, "latency_ms_p50": 5, "latency_ms_p95": 6,
            }}), encoding="utf-8")

            result = run_experiment(
                calls_file=calls, combined_file=combined, output_dir=output,
                bootstrap_samples=10,
            )

            route = result["route_ablation"]
            self.assertEqual(2, route["eager_semantic_request_count"])
            self.assertEqual(1, route["progressive_semantic_request_count"])
            self.assertEqual(0.5, route["semantic_request_reduction"])
            self.assertTrue((output / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
