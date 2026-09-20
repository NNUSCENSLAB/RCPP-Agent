#!/usr/bin/env python3
"""Aggregate metric reports from seeds 42, 43 and 44."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    collected: dict[str, list[float]] = {}
    for report in reports:
        for variant, metrics in report["variants"].items():
            for metric, value in metrics.items():
                collected.setdefault(f"{variant}.{metric}", []).append(float(value["mean"]))
    result = {
        key: {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "seeds": len(values),
        }
        for key, values in sorted(collected.items())
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
