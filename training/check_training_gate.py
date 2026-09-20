#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=Path)
    args = parser.parse_args()
    directory = args.train_jsonl.resolve().parent
    audit = json.loads((directory / "audit_report.json").read_text(encoding="utf-8"))
    approval_path = directory / "manual_review_approval.json"
    if not audit.get("gate_passed"):
        raise SystemExit("Automated dataset audit did not reach 95%")
    if not approval_path.exists():
        raise SystemExit("manual_review_approval.json is missing; review manual_review_sample.jsonl first")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if not approval.get("approved"):
        raise SystemExit("Manual structure/label review did not reach 95%")
    if args.train_jsonl.stat().st_size == 0:
        raise SystemExit("Training JSONL is empty")
    validation = directory / "validation.jsonl"
    if not validation.exists() or validation.stat().st_size == 0:
        raise SystemExit("validation.jsonl is missing or empty")
    if not (directory / "test_lock.json").exists():
        raise SystemExit("test_lock.json is missing")


if __name__ == "__main__":
    main()
