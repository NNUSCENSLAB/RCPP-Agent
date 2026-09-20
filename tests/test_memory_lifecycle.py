import unittest
from datetime import datetime, timezone

from rcpp_core.memory_lifecycle import classify_memory, reusable


class MemoryLifecycleTests(unittest.TestCase):
    def test_high_quality_active_memory_is_reusable(self):
        memory = {"_meta": {"status": "active", "quality_score": 0.95}}
        self.assertTrue(reusable(memory))

    def test_expired_memory_is_not_reusable(self):
        memory = {
            "_meta": {
                "status": "active",
                "quality_score": 1.0,
                "expires_at": "2025-01-01T00:00:00+00:00",
            }
        }
        self.assertEqual("expired", classify_memory(memory, now=datetime(2026, 1, 1, tzinfo=timezone.utc)))
        self.assertFalse(reusable(memory))

    def test_invalid_expiry_is_conservatively_conflicted(self):
        self.assertEqual("conflicted", classify_memory({"_meta": {"expires_at": "bad-date"}}))


if __name__ == "__main__":
    unittest.main()
