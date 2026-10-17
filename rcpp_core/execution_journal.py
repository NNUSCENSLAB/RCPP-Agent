"""Run-scoped execution receipts, separate from Neo4j domain memory."""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Mapping


def run_directory(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("run_id must contain only letters, numbers, _, - or .")
    return Path(os.getenv("RCPP_RUN_ROOT", ".rcpp/runs")) / run_id


def fingerprint(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {"path": str(source), "missing": True}
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source.resolve()), "sha256": digest.hexdigest()}


@contextmanager
def exclusive_file(path: Path):
    """Fail fast on concurrent execution; OS releases the lock after a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is executing this run: {path.parent.name}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ExecutionJournal:
    """Commit each returned result before graph writes; replay the receipt on resume.

    A crash between a provider accepting a request and this commit can still repeat
    that request. Exactly-once billing requires provider-side idempotency.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS receipts (
                key TEXT PRIMARY KEY, operation TEXT NOT NULL, input_hash TEXT NOT NULL,
                status TEXT NOT NULL, result_json TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""")

    def execute(self, operation: str, inputs: Mapping[str, Any],
                compute: Callable[[], Any]) -> tuple[Any, bool, str]:
        input_hash = fingerprint(inputs)
        key = fingerprint({"operation": operation, "input_hash": input_hash})
        with exclusive_file(self.path.with_suffix(".lock")):
            with closing(sqlite3.connect(self.path)) as conn, conn:
                row = conn.execute("SELECT status,result_json FROM receipts WHERE key=?", (key,)).fetchone()
                if row and row[0] == "completed":
                    return json.loads(row[1]), True, key
                conn.execute("""INSERT INTO receipts(key,operation,input_hash,status)
                    VALUES(?,?,?,'running') ON CONFLICT(key) DO UPDATE SET status='running',
                    updated_at=CURRENT_TIMESTAMP""", (key, operation, input_hash))
            result = compute()
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
            with closing(sqlite3.connect(self.path)) as conn, conn:
                conn.execute("""UPDATE receipts SET status='completed',result_json=?,
                    updated_at=CURRENT_TIMESTAMP WHERE key=?""", (encoded, key))
            return result, False, key
