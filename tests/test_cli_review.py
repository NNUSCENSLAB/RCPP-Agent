from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rcpp_cli.app import main


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "requires LangGraph runtime dependencies")
class CliReviewTests(unittest.TestCase):
    def test_review_cli_resumes_and_updates_manifest(self):
        from rcpp_core.runtime import RCPPAgent
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "run1"
            directory.mkdir()
            pending = {"kind": "scene_review", "review_token": "t", "pending": [{"rps_id": "A"}]}
            (directory / "manifest.json").write_text(json.dumps({
                "status": "awaiting_review", "weight_source": "expert", "shadow": False,
            }), encoding="utf-8")
            (directory / "result.json").write_text(json.dumps({
                "status": "awaiting_review", "pending_reviews": [pending],
            }), encoding="utf-8")
            response = {"review_token": "t", "reviewer": "user", "decisions": [
                {"rps_id": "A", "action": "exclude", "note": "verified blocked"}]}
            path = root / "review.json"
            path.write_text(json.dumps(response), encoding="utf-8")
            with patch("rcpp_cli.app.RUN_ROOT", root), patch.dict(os.environ, {"RCPP_RUN_ROOT": str(root)}), \
                    patch("rcpp_core.runtime.RCPPAgent") as runtime, redirect_stdout(io.StringIO()):
                runtime.return_value.__enter__.return_value.run.return_value = {
                    "status": "review_completed", "retrieval_events": [], "retrieval_summary": {},
                }
                self.assertEqual(0, main(["plan", "review", "run1", "--file", str(path)]))
                runtime.return_value.__enter__.return_value.run.assert_called_once_with(
                    thread_id="run1", review_response=response)
                self.assertEqual("expert", os.environ["RCPP_WEIGHT_SOURCE"])
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", manifest["status"])
            self.assertIn("completed_at", manifest)

    def test_waiting_is_not_reported_as_completed(self):
        from rcpp_cli.app import _save_plan_result
        with tempfile.TemporaryDirectory() as temporary, patch("rcpp_cli.app.RUN_ROOT", Path(temporary)):
            manifest = {}
            _save_plan_result("waiting", manifest, {"status": "awaiting_review", "pending_reviews": []})
            self.assertEqual("awaiting_review", manifest["status"])
            self.assertNotIn("completed_at", manifest)
