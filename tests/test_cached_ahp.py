import unittest

from rcpp_core.ahp_policy import cached_ahp_weights


class CachedAHPTests(unittest.TestCase):
    def test_cached_scenario_weights_need_no_api_key(self):
        result = cached_ahp_weights("scenario", "balance_oriented")
        self.assertAlmostEqual(1.0, sum(result["weights"].values()))
        self.assertEqual("cached", result["source"])

    def test_cached_phase_weights_need_no_api_key(self):
        result = cached_ahp_weights(
            "phase",
            "balance_oriented",
            {key: 0.2 for key in ("technical", "economic", "social", "traffic", "policy")},
            "INITIAL_EXPLORATION",
        )
        self.assertAlmostEqual(1.0, sum(result["weights"].values()))


if __name__ == "__main__":
    unittest.main()
