import unittest

from rcpp_core.build_scene_semantic_memory import _save_versioned_memory_nodes


class FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))


class Neo4jVersioningTests(unittest.TestCase):
    def test_scene_and_semantic_nodes_are_written_separately(self):
        session = FakeSession()
        memory = {
            "RPS-1": {
                "RPS_id": "RPS-1",
                "coordinates": [118.1, 32.1],
                "memory_node": {
                    "scene_memory": {"_meta": {"adapter_version": "scene-v2"}},
                    "semantic_memory": {"_meta": {"adapter_version": "semantic-v2"}},
                },
            }
        }
        _save_versioned_memory_nodes(session, "gulou", memory)
        queries = "\n".join(query for query, _ in session.calls)
        self.assertIn("SceneMemory", queries)
        self.assertIn("SemanticMemory", queries)
        self.assertIn("HAS_SCENE_MEMORY", queries)
        self.assertIn("HAS_SEMANTIC_MEMORY", queries)
        self.assertIn("SUPERSEDES", queries)
        self.assertIn("superseded", queries)


if __name__ == "__main__":
    unittest.main()
