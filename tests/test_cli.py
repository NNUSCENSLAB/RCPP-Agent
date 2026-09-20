import ast
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from rcpp_cli.app import main


class CliTests(unittest.TestCase):
    def test_help_is_available(self):
        with self.assertRaises(SystemExit) as raised, redirect_stdout(io.StringIO()):
            main(["--help"])
        self.assertEqual(0, raised.exception.code)

    def test_old_python_entrypoint_is_removed(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / "src" / "rcpp_agent.py").exists())
        tree = ast.parse((root / "rcpp_core" / "runtime.py").read_text(encoding="utf-8"))
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("main", names)

    def test_structured_router_explain_needs_no_api_key(self):
        with redirect_stdout(io.StringIO()):
            code = main(["router", "explain", "--kind", "semantic", "--memory-state", "fresh", "--memory-quality", "0.95"])
        self.assertEqual(0, code)


if __name__ == "__main__":
    unittest.main()
