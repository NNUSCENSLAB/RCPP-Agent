import unittest

from pathlib import Path


class Neo4jVersioningTests(unittest.TestCase):
    def test_scene_and_semantic_nodes_are_written_separately(self):
        source = Path(__file__).resolve().parents[1] / "rcpp_core" / "graph_memory_repository.py"
        queries = source.read_text(encoding="utf-8")
        self.assertIn("SceneMemory", queries)
        self.assertIn("SemanticMemory", queries)
        self.assertIn("HAS_SCENE_MEMORY", queries)
        self.assertIn("HAS_SEMANTIC_MEMORY", queries)
        self.assertIn("SUPERSEDES", queries)
        self.assertIn("superseded", queries)


if __name__ == "__main__":
    unittest.main()
