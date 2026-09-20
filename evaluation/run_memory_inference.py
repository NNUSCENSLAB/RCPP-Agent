#!/usr/bin/env python3
"""Run one registered model variant on an audited test file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcpp_core.model_registry import get_deployment
from rcpp_core.scene_semantic_inference import run_scene_inference_file, run_semantic_inference_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("scene", "semantic"))
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--adapter-path")
    args = parser.parse_args()
    deployment = get_deployment(args.kind, args.model_key)
    adapter = "" if args.base_only else args.adapter_path or deployment.adapter_path
    deployment = get_deployment(
        args.kind,
        args.model_key,
        base_model=deployment.base_model,
        adapter_path=adapter,
    )
    if args.kind == "scene":
        run_scene_inference_file(
            args.input,
            args.output,
            deployment.base_model,
            deployment.adapter_path,
            {},
            deployment=deployment,
        )
    else:
        run_semantic_inference_file(
            args.input,
            args.output,
            deployment.base_model,
            deployment.adapter_path,
            {},
            max_new_tokens=deployment.max_new_tokens,
            max_length=deployment.max_length,
            deployment=deployment,
            evaluation_mode=True,
        )


if __name__ == "__main__":
    main()
