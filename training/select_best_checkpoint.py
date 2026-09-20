#!/usr/bin/env python3
"""Resolve the validation-loss winner without assuming the last checkpoint is best."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    candidates = []
    for state_path in args.output_dir.glob("checkpoint-*/trainer_state.json"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        losses = [
            float(entry["eval_loss"])
            for entry in state.get("log_history", [])
            if entry.get("eval_loss") is not None
        ]
        if losses:
            candidates.append((min(losses), state_path.parent))
    if not candidates:
        raise SystemExit("No checkpoint with eval_loss was found")
    loss, checkpoint = min(candidates, key=lambda item: item[0])
    result = {"checkpoint": str(checkpoint.resolve()), "eval_loss": loss}
    (args.output_dir / "best_checkpoint.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
