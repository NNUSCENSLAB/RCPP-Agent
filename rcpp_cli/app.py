"""The only supported executable entrypoint for RCPP-Agent."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Sequence

from configs.config import get_provider_api_key
from configs.data_config import DEFAULT_WORKSPACE
from rcpp_core.model_registry import MODEL_REGISTRY
from rcpp_core.provider_registry import general_reasoning_deployment
from rcpp_core.routing import MemoryAwareRouter, RoutingContext
from rcpp_core.memory_lifecycle import classify_memory

RUN_ROOT = Path(os.getenv("RCPP_RUN_ROOT", ".rcpp/runs"))


def _json_dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _run_dir(run_id: str) -> Path:
    return RUN_ROOT / run_id


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _memory_route_state(path_value: str | None, kind: str) -> tuple[str, float]:
    if not path_value:
        return "miss", 0.0
    path = Path(path_value)
    if not path.exists():
        return "miss", 0.0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "conflicted", 0.0
    records = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    qualities: list[float] = []
    missing = False
    for record in records:
        if not isinstance(record, dict):
            missing = True
            continue
        node = record.get("memory_node", record)
        memory = node.get(f"{kind}_memory") if isinstance(node, dict) else None
        if not isinstance(memory, dict):
            missing = True
            continue
        state = classify_memory(memory)
        if state == "conflicted":
            return "conflicted", 0.0
        if state == "expired":
            return "stale", 0.0
        if state == "superseded":
            missing = True
        meta = memory.get("_meta") if isinstance(memory.get("_meta"), dict) else {}
        try:
            qualities.append(float(meta.get("quality_score", 0.0)))
        except (TypeError, ValueError):
            qualities.append(0.0)
    quality = min(qualities) if qualities else 0.0
    if not missing and qualities and quality >= 0.9:
        return "fresh", quality
    return "partial", quality


def command_doctor(_: argparse.Namespace) -> int:
    general = general_reasoning_deployment()
    models = {
        key: {
            "kind": item.kind,
            "stage": item.stage,
            "availability": item.availability,
            "base_model": item.base_model,
            "adapter": item.adapter_version,
        }
        for key, item in MODEL_REGISTRY.items()
    }
    checks = {
        "python": sys.version.split()[0],
        "workspace": {"path": str(DEFAULT_WORKSPACE), "exists": Path(DEFAULT_WORKSPACE).exists()},
        "models": models,
        "general_reasoning": {
            "provider": general.provider,
            "model": general.model,
            "required": general.required,
            "availability": "ready" if general.ready else "disabled",
        },
        "credentials": {
            "deepseek": bool(get_provider_api_key("deepseek")),
            "openai": bool(get_provider_api_key("openai")),
        },
    }
    _json_dump(checks)
    return 0


def _structured_instruction(args: argparse.Namespace) -> dict[str, Any]:
    if args.instruction:
        from src.task_orchestration_agent import TaskOrchestrationAgent

        return TaskOrchestrationAgent().parse_natural_language_to_global_instruction(
            args.instruction, workspace_dir=args.workspace
        )
    if not args.area:
        raise ValueError("--area is required unless --instruction is provided")
    return {
        "area": args.area,
        "scenario": args.scenario,
        "time_horizon": [args.start_year, args.end_year],
        "year": args.start_year,
        "workspace_dir": args.workspace,
        **(
            {"scene_semantic_memory_json_path": args.memory_json}
            if args.memory_json
            else {}
        ),
    }


def command_plan_run(args: argparse.Namespace) -> int:
    run_id = args.run_id or uuid.uuid4().hex
    os.environ["RCPP_WEIGHT_SOURCE"] = args.weight_source
    os.environ["RCPP_SHADOW_ENABLED"] = "true" if args.shadow else "false"
    instruction = _structured_instruction(args)
    started = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": started,
        "instruction": instruction,
        "weight_source": args.weight_source,
        "shadow": args.shadow,
    }
    _write_json(_run_dir(run_id) / "manifest.json", manifest)
    route_payload: dict[str, Any] = {}
    for kind in ("scene", "semantic"):
        memory_state, memory_quality = _memory_route_state(args.memory_json, kind)
        context = RoutingContext(
            run_id=run_id,
            task_type=kind,
            memory_state=memory_state,
            memory_quality=memory_quality,
            shadow_enabled=args.shadow,
        )
        route_payload[kind] = {
            "context": vars(context),
            "decision": MemoryAwareRouter().decide(context).to_dict(),
        }
    _write_json(_run_dir(run_id) / "route.json", route_payload)
    try:
        from rcpp_core.runtime import RCPPAgent

        result = RCPPAgent(max_iterations=args.max_iterations).run(instruction, thread_id=run_id)
        manifest.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            result=result,
        )
        _write_json(_run_dir(run_id) / "result.json", result)
        _write_json(_run_dir(run_id) / "manifest.json", manifest)
        _json_dump(result)
        return 0
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(_run_dir(run_id) / "manifest.json", manifest)
        print(f"RCPP run failed; run_id={run_id}: {exc}", file=sys.stderr)
        return 1


def command_plan_resume(args: argparse.Namespace) -> int:
    try:
        from rcpp_core.runtime import RCPPAgent

        result = RCPPAgent(max_iterations=args.max_iterations).run(thread_id=args.run_id)
        _write_json(_run_dir(args.run_id) / "result.json", result)
        _json_dump(result)
        return 0
    except Exception as exc:
        print(f"Resume failed for {args.run_id}: {exc}", file=sys.stderr)
        return 1


def command_plan_status(args: argparse.Namespace) -> int:
    manifest = _run_dir(args.run_id) / "manifest.json"
    if not manifest.exists():
        print(f"Unknown run_id: {args.run_id}", file=sys.stderr)
        return 2
    print(manifest.read_text(encoding="utf-8"))
    return 0


def command_router_explain(args: argparse.Namespace) -> int:
    if args.run_id:
        trace = _run_dir(args.run_id) / "route.json"
        if not trace.exists():
            print(f"No route trace for run_id: {args.run_id}", file=sys.stderr)
            return 2
        print(trace.read_text(encoding="utf-8"))
        return 0
    context = RoutingContext(
        run_id="explain",
        task_type=args.kind,
        memory_state=args.memory_state,
        memory_quality=args.memory_quality,
        request_risk=args.risk,
        shadow_enabled=args.shadow,
    )
    decision = MemoryAwareRouter().decide(context)
    _json_dump({"context": vars(context), "decision": decision.to_dict()})
    return 0


def command_router_stats(_: argparse.Namespace) -> int:
    statuses: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    if RUN_ROOT.exists():
        for manifest in RUN_ROOT.glob("*/manifest.json"):
            try:
                statuses[json.loads(manifest.read_text(encoding="utf-8")).get("status", "unknown")] += 1
            except (OSError, json.JSONDecodeError):
                statuses["invalid"] += 1
        for route in RUN_ROOT.glob("*/route.json"):
            try:
                payload = json.loads(route.read_text(encoding="utf-8"))
                if "decision" in payload:
                    routes[payload["decision"]["action"]] += 1
                else:
                    for item in payload.values():
                        routes[item["decision"]["action"]] += 1
            except (OSError, json.JSONDecodeError, KeyError):
                routes["invalid"] += 1
    _json_dump({"runs": dict(statuses), "route_actions": dict(routes)})
    return 0


def command_memory_inspect(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"Memory file not found: {path}", file=sys.stderr)
        return 2
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.values() if isinstance(raw, dict) else raw
    scene = semantic = invalid = 0
    for item in values:
        if not isinstance(item, dict):
            invalid += 1
            continue
        node = item.get("memory_node", item)
        scene += int(isinstance(node.get("scene_memory"), dict))
        semantic += int(isinstance(node.get("semantic_memory"), dict))
    _json_dump({"path": str(path.resolve()), "scene": scene, "semantic": semantic, "invalid": invalid})
    return 0


def command_memory_build(args: argparse.Namespace) -> int:
    from rcpp_core.build_scene_semantic_memory import load_scene_semantic_memory_json_file

    memory = load_scene_semantic_memory_json_file(Path(args.path))
    _json_dump({"validated_entries": len(memory), "source": str(Path(args.path).resolve())})
    return 0


def command_data_import(args: argparse.Namespace) -> int:
    from tools.data_flow_tool.import_spatialite_tool import main as import_main

    old = sys.argv
    try:
        sys.argv = [old[0], "--area", args.area]
        import_main()
        return 0
    finally:
        sys.argv = old


def command_eval(args: argparse.Namespace) -> int:
    if args.target == "router":
        return command_router_stats(args)
    print("Memory evaluation uses evaluation/evaluate_memory.py with a locked prediction and gold JSONL.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rcpp", description="RCPP-Agent command line runtime")
    sub = parser.add_subparsers(dest="group", required=True)
    doctor = sub.add_parser("doctor", help="check runtime, model and provider readiness")
    doctor.set_defaults(func=command_doctor)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="command", required=True)
    data_import = data_sub.add_parser("import")
    data_import.add_argument("--area", required=True)
    data_import.set_defaults(func=command_data_import)

    memory = sub.add_parser("memory")
    memory_sub = memory.add_subparsers(dest="command", required=True)
    for name, func in (("build", command_memory_build), ("inspect", command_memory_inspect)):
        cmd = memory_sub.add_parser(name)
        cmd.add_argument("path")
        cmd.set_defaults(func=func)

    plan = sub.add_parser("plan")
    plan_sub = plan.add_subparsers(dest="command", required=True)
    run = plan_sub.add_parser("run")
    run.add_argument("--area")
    run.add_argument("--scenario", choices=("efficiency_oriented", "equity_oriented", "balance_oriented"), default="balance_oriented")
    run.add_argument("--start-year", type=int, default=2025)
    run.add_argument("--end-year", type=int, default=2030)
    run.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    run.add_argument("--memory-json")
    run.add_argument("--instruction", help="optional natural-language path; requires configured provider")
    run.add_argument("--weight-source", choices=("cached", "expert", "llm"), default="cached")
    run.add_argument("--shadow", action="store_true")
    run.add_argument("--run-id")
    run.add_argument("--max-iterations", type=int)
    run.set_defaults(func=command_plan_run)
    resume = plan_sub.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--max-iterations", type=int)
    resume.set_defaults(func=command_plan_resume)
    status = plan_sub.add_parser("status")
    status.add_argument("run_id")
    status.set_defaults(func=command_plan_status)

    router = sub.add_parser("router")
    router_sub = router.add_subparsers(dest="command", required=True)
    explain = router_sub.add_parser("explain")
    explain.add_argument("run_id", nargs="?")
    explain.add_argument("--kind", choices=("scene", "semantic"), default="scene")
    explain.add_argument("--memory-state", choices=("miss", "fresh", "partial", "stale", "conflicted"), default="miss")
    explain.add_argument("--memory-quality", type=float, default=0.0)
    explain.add_argument("--risk", choices=("low", "medium", "high"), default="medium")
    explain.add_argument("--shadow", action="store_true")
    explain.set_defaults(func=command_router_explain)
    stats = router_sub.add_parser("stats")
    stats.set_defaults(func=command_router_stats)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("target", choices=("memory", "router"))
    evaluate.set_defaults(func=command_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
