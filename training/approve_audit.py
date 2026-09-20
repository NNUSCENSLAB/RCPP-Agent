#!/usr/bin/env python3
"""Record the human review result required before QLoRA training."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed", type=int, required=True)
    parser.add_argument("--structure-correct", type=int, required=True)
    parser.add_argument("--label-correct", type=int, required=True)
    args = parser.parse_args()
    if args.reviewed <= 0:
        raise ValueError("reviewed must be positive")
    report = json.loads((args.dataset_dir / "audit_report.json").read_text(encoding="utf-8"))
    expected = int(report.get("manual_review_sample_size", 0))
    if args.reviewed != expected:
        raise ValueError(f"reviewed must equal manual review sample size ({expected})")
    if not 0 <= args.structure_correct <= args.reviewed:
        raise ValueError("structure-correct must be between 0 and reviewed")
    if not 0 <= args.label_correct <= args.reviewed:
        raise ValueError("label-correct must be between 0 and reviewed")
    structure_rate = args.structure_correct / args.reviewed
    label_rate = args.label_correct / args.reviewed
    approval = {
        "reviewer": args.reviewer,
        "reviewed": args.reviewed,
        "structure_correct": args.structure_correct,
        "label_correct": args.label_correct,
        "structure_rate": structure_rate,
        "label_rate": label_rate,
        "approved": structure_rate >= 0.95 and label_rate >= 0.95,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "audit_source": str((args.dataset_dir / "audit_report.json").resolve()),
    }
    path = args.dataset_dir / "manual_review_approval.json"
    path.write_text(json.dumps(approval, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(approval, ensure_ascii=False, indent=2))
    return 0 if approval["approved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
