import tempfile
import unittest
from pathlib import Path

from evaluation.live_semantic_ablation import (
    ApiCall,
    SiliconFlowClient,
    _aggregate,
    select_cases,
)
from evaluation.retrieval_ablation import RpsCase


class LiveSemanticAblationTests(unittest.TestCase):
    def test_stratified_selection_contains_pass_and_stop(self):
        cases = [RpsCase(f"R{i}", "road", i < 7) for i in range(10)]
        selected = select_cases(cases, max_calls=4, seed=42)
        self.assertEqual(len(selected), 4)
        self.assertTrue(any(item.reference_needs_semantic for item in selected))
        self.assertTrue(any(not item.reference_needs_semantic for item in selected))

    def test_client_captures_usage_latency_and_trace_without_logging_key(self):
        observed = {}

        def transport(url, headers, payload, timeout):
            observed.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return (
                {
                    "choices": [{"message": {"content": '{"rps_id":"R1"}'}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                {"x-siliconcloud-trace-id": "trace-1"},
            )

        client = SiliconFlowClient(
            api_key="test-secret", model="test/model", max_retries=0, transport=transport
        )
        call = client.complete(rps_id="R1", semantic={"traffic_total": 12}, scene_passed=True)
        self.assertTrue(call.success)
        self.assertEqual(call.total_tokens, 120)
        self.assertEqual(call.trace_id, "trace-1")
        self.assertNotIn("test-secret", call.response)
        self.assertEqual(observed["payload"]["model"], "test/model")

    def test_progressive_aggregation_removes_stopped_calls(self):
        calls = [
            ApiCall("pass", True, True, 100, 10, 110, 50, "a", "{}"),
            ApiCall("stop", False, True, 90, 10, 100, 40, "b", "{}"),
        ]
        eager = _aggregate(calls, progressive=False)
        progressive = _aggregate(calls, progressive=True)
        self.assertEqual(eager["semantic_request_count"], 2)
        self.assertEqual(progressive["semantic_request_count"], 1)
        self.assertEqual(progressive["prompt_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
