import json
import unittest

from rcpp_core.semantic_rules import deterministic_semantic_memory
from tools.memory_construction_tool.merge_memory_tool import parse_semantic_memory


class MergeCompatibilityTests(unittest.TestCase):
    def test_negative_coordinates_and_metadata_survive(self):
        summary = {
            "rps_id": "RPS-A",
            "coordinates": {"lng": -1.25, "lat": 31.2, "angle": -15},
            "poi_counts_1km": {"residential": 8, "commercial": 8},
            "poi_counts_sensitive_0.1km": {"sensitive": 0},
            "traffic_flow_stats": {"total_inflow": 1, "total_outflow": 2},
            "infrastructure_dist": {"grid_distance_km": 0.7},
            "bsv_image": "a.jpg",
        }
        memory = deterministic_semantic_memory(summary)
        memory["_meta"] = {"schema_version": "2.0"}
        messages = [
            {"role": "user", "content": "【Statistical Summary】\n" + json.dumps(summary)},
            {"role": "assistant", "content": json.dumps(memory)},
        ]
        parsed = parse_semantic_memory(messages)
        self.assertEqual(-1.25, parsed["lng"])
        self.assertEqual("2.0", parsed["semantic_memory"]["_meta"]["schema_version"])


if __name__ == "__main__":
    unittest.main()
