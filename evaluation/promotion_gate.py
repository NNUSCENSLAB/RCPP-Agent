#!/usr/bin/env python3
"""Decide whether a QLoRA deployment is eligible to replace the current model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--kind", choices=("scene", "semantic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    metrics = report.get("variants", {}).get("new_qlora", {})
    gain = report.get("comparisons", {}).get("final_system_gain", {})
    reasons: list[str] = []

    def mean(name: str) -> float:
        return float((metrics.get(name) or {}).get("mean", 0.0))

    if mean("json_valid") < 0.99:
        reasons.append("JSON validity is below 99%")
    if args.kind == "semantic":
        if mean("numeric_copy_accuracy") < 1.0:
            reasons.append("semantic numeric copy accuracy is below 100%")
        if mean("conflict_free") < 1.0:
            reasons.append("semantic rule conflict rate is not zero")
    delta = float(gain.get("delta", 0.0))
    ci_low = float(gain.get("ci_low", 0.0))
    if not (ci_low > 0.0 or delta >= 0.03):
        reasons.append("primary metric gain is neither significant nor at least 3 percentage points")
    decision = {
        "promote": not reasons,
        "kind": args.kind,
        "candidate": f"{args.kind}_qwen3{'_vl_8b' if args.kind == 'scene' else '_8b'}_qlora",
        "reasons": reasons,
        "final_system_gain": gain,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["promote"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
