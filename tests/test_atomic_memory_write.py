from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from rcpp_core.graph_memory_repository import GraphMemoryRepository
from rcpp_core.memory_refresh import TargetedMemoryRefresher
from runtime_support import scene


class FakeResult:
    def __init__(self, row):
        self.row = row

    def single(self):
        return self.row

    def consume(self):
        pass


class Transaction:
    def __init__(self, fail_facts=False, collision=False):
        self.statements = []
        self.fail_facts = fail_facts
        self.collision = collision

    def run(self, query, **params):
        self.statements.append(query)
        if self.fail_facts and "UNWIND $facts" in query:
            raise RuntimeError("injected fact write failure")
        return FakeResult(None if self.collision else {"memory_id": params.get("memory_id")})


class Session:
    def __init__(self, tx):
        self.tx = tx
        self.outcome = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute_write(self, callback):
        try:
            result = callback(self.tx)
        except Exception:
            self.outcome = "rollback"
            raise
        self.outcome = "commit"
        return result


class AtomicMemoryWriteTests(unittest.TestCase):
    def repository(self, tx):
        session = Session(tx)
        repo = GraphMemoryRepository()
        repo._session = lambda: session
        return repo, session

    def test_version_facts_and_images_share_one_transaction(self):
        tx = Transaction()
        repo, session = self.repository(tx)
        repo.append_scene_memory("test", "A", scene(), memory_id="one", source_images=["A.jpg"])
        self.assertEqual("commit", session.outcome)
        self.assertEqual(3, len(tx.statements))
        self.assertIn("m.is_current=true AND old.is_current=true", tx.statements[0])
        self.assertIn("DERIVED_FROM", tx.statements[2])

    def test_fact_failure_rolls_back_version_transaction(self):
        repo, session = self.repository(Transaction(fail_facts=True))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            repo.append_scene_memory("test", "A", scene(), memory_id="one")
        self.assertEqual("rollback", session.outcome)

    def test_identity_collision_does_not_continue_writing_facts(self):
        tx = Transaction(collision=True)
        repo, session = self.repository(tx)
        with self.assertRaisesRegex(ValueError, "different payload"):
            repo.append_scene_memory("test", "A", scene(), memory_id="collision")
        self.assertEqual("rollback", session.outcome)
        self.assertEqual(1, len(tx.statements))

    def test_historical_image_name_resolves_to_actual_local_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.sqlite"
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("CREATE TABLE bsv_image(image_path TEXT)")
                conn.execute("INSERT INTO bsv_image VALUES(?)", (str(Path(directory) / "A.jpg"),))
            resolved = TargetedMemoryRefresher.resolve_image_sources({"db_path": str(path)})
            self.assertEqual(str(Path(directory) / "A.jpg"), resolved["A.jpg"])
