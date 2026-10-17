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
from rcpp_core.execution_journal import run_directory

RUN_ROOT = Path(os.getenv("RCPP_RUN_ROOT", ".rcpp/runs"))


def _json_dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _run_dir(run_id: str) -> Path:
    # RUN_ROOT remains patchable for tests and callers.
    run_directory(run_id)  # validate the identifier before joining paths
    return RUN_ROOT / run_id


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


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
            "siliconflow": bool(get_provider_api_key("siliconflow")),
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
    }


def command_plan_run(args: argparse.Namespace) -> int:
    run_id = args.run_id or uuid.uuid4().hex
    if (_run_dir(run_id) / "manifest.json").exists():
        raise ValueError("This run_id already exists; use rcpp plan resume or select a new run_id")
    os.environ["RCPP_WEIGHT_SOURCE"] = args.weight_source
    os.environ["RCPP_SHADOW_ENABLED"] = "true" if args.shadow else "false"
    instruction = _structured_instruction(args)
    instruction["run_id"] = run_id
    instruction.update(latency_budget_ms=args.latency_budget_ms,
                       cost_budget_usd=args.cost_budget_usd, request_risk=args.risk)
    # Validate budgets even when all memory is fresh and no router call is needed.
    RoutingContext(run_id, "scene", latency_budget_ms=args.latency_budget_ms,
                   cost_budget=args.cost_budget_usd)
    profile_source = args.routing_profiles or os.getenv("RCPP_ROUTING_PROFILES")
    if profile_source:
        from rcpp_core.routing import load_profiles
        load_profiles(profile_source)
        profile_path = _run_dir(run_id) / "routing_profiles.json"
        _write_json(profile_path, json.loads(Path(profile_source).read_text(encoding="utf-8")))
        os.environ["RCPP_ROUTING_PROFILES"] = str(profile_path.resolve())
    started = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": run_id,
        "status": "running",
        "started_at": started,
        "instruction": instruction,
        "weight_source": args.weight_source,
        "shadow": args.shadow,
        "max_iterations": args.max_iterations,
        "routing_profiles": os.getenv("RCPP_ROUTING_PROFILES"),
    }
    _write_json(_run_dir(run_id) / "manifest.json", manifest)
    try:
        from rcpp_core.runtime import RCPPAgent

        with RCPPAgent(max_iterations=args.max_iterations) as runtime:
            result = runtime.run(instruction, thread_id=run_id)
        _save_plan_result(run_id, manifest, result)
        _json_dump(result)
        return 0
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(_run_dir(run_id) / "manifest.json", manifest)
        print(f"RCPP run failed; run_id={run_id}: {exc}", file=sys.stderr)
        return 1


def _save_plan_result(run_id: str, manifest: dict[str, Any], result: dict[str, Any]) -> None:
    waiting = result.get("status") == "awaiting_review"
    manifest.update(status="awaiting_review" if waiting else "completed", result=result,
                    updated_at=datetime.now(timezone.utc).isoformat())
    manifest.pop("error", None)
    if not waiting:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(_run_dir(run_id) / "route.json", {
        "events": result.get("retrieval_events", []), "summary": result.get("retrieval_summary", {})})
    _write_json(_run_dir(run_id) / "result.json", result)
    _write_json(_run_dir(run_id) / "manifest.json", manifest)


def _resume_plan(args: argparse.Namespace, review_response: dict[str, Any] | None = None) -> int:
    manifest_path = _run_dir(args.run_id) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Unknown run_id: {args.run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    os.environ["RCPP_WEIGHT_SOURCE"] = manifest.get("weight_source", "cached")
    os.environ["RCPP_SHADOW_ENABLED"] = "true" if manifest.get("shadow") else "false"
    if manifest.get("routing_profiles"):
        os.environ["RCPP_ROUTING_PROFILES"] = manifest["routing_profiles"]
    else:
        os.environ.pop("RCPP_ROUTING_PROFILES", None)
    try:
        from rcpp_core.runtime import RCPPAgent
        maximum = args.max_iterations if args.max_iterations is not None else manifest.get("max_iterations")
        with RCPPAgent(max_iterations=maximum) as runtime:
            result = runtime.run(thread_id=args.run_id, review_response=review_response)
        _save_plan_result(args.run_id, manifest, result)
        _json_dump(result)
        return 0
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(manifest_path, manifest)
        print(f"Resume failed for {args.run_id}: {exc}", file=sys.stderr)
        return 1


def command_plan_resume(args: argparse.Namespace) -> int:
    return _resume_plan(args)


def command_plan_reviews(args: argparse.Namespace) -> int:
    result = json.loads((_run_dir(args.run_id) / "result.json").read_text(encoding="utf-8"))
    _json_dump({"run_id": args.run_id, "pending_reviews": result.get("pending_reviews", [])})
    return 0


def command_plan_review(args: argparse.Namespace) -> int:
    from rcpp_core.planning_workflow import validate_review
    result = json.loads((_run_dir(args.run_id) / "result.json").read_text(encoding="utf-8"))
    pending = result.get("pending_reviews") or []
    if len(pending) != 1 or pending[0].get("kind") != "scene_review":
        raise ValueError("This run has no pending Scene review")
    payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
    validate_review(payload, pending[0]["pending"], pending[0]["review_token"])
    return _resume_plan(args, payload)


def command_plan_status(args: argparse.Namespace) -> int:
    manifest = _run_dir(args.run_id) / "manifest.json"
    if not manifest.exists():
        print(f"Unknown run_id: {args.run_id}", file=sys.stderr)
        return 2
    print(manifest.read_text(encoding="utf-8"))
    return 0


def command_router_explain(args: argparse.Namespace) -> int:
    if args.run_id:
        if args.rps:
            from rcpp_core.graph_memory_repository import GraphMemoryRepository
            trace = _run_dir(args.run_id) / "route.json"
            route_events: list[dict[str, Any]] = []
            if trace.exists():
                payload = json.loads(trace.read_text(encoding="utf-8"))
                route_events = [x for x in payload.get("events", []) if x.get("rps_id") == args.rps]
            evidence_error = None
            try:
                paths = GraphMemoryRepository().get_evidence_path(args.run_id, args.rps)
            except Exception as exc:
                paths = []
                evidence_error = f"{type(exc).__name__}: {exc}"
            _json_dump({"run_id": args.run_id, "rps_id": args.rps,
                        "route_events": route_events, "evidence_paths": paths,
                        **({"evidence_error": evidence_error} if evidence_error else {})})
            return 0 if paths or route_events else 2
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
        latency_budget_ms=args.latency_budget_ms,
        cost_budget=args.cost_budget_usd,
    )
    from rcpp_core.routing import load_profiles
    decision = MemoryAwareRouter(profiles=load_profiles(args.routing_profiles)).decide(context)
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
                elif "events" in payload:
                    for event in payload["events"]:
                        if isinstance(event.get("decision"), dict):
                            routes[event["decision"]["action"]] += 1
                else:
                    for item in payload.values():
                        routes[item["decision"]["action"]] += 1
            except (OSError, json.JSONDecodeError, KeyError):
                routes["invalid"] += 1
    _json_dump({"runs": dict(statuses), "route_actions": dict(routes)})
    return 0


def command_memory_inspect(args: argparse.Namespace) -> int:
    from rcpp_core.graph_memory_repository import GraphMemoryRepository, assess_scene
    repo = GraphMemoryRepository()
    scene = repo.get_current_scene(args.area, [args.rps]).get(args.rps)
    semantic = repo.get_current_semantic(args.area, [args.rps]).get(args.rps)
    assessment = assess_scene(scene, declared_status=(scene or {}).get("_meta", {}).get("status")) if scene else None
    _json_dump({"area": args.area, "rps_id": args.rps, "scene": scene, "semantic": semantic,
                "scene_assessment": vars(assessment) if assessment else {"state": "miss"}})
    return 0 if scene or semantic else 2


def command_memory_history(args: argparse.Namespace) -> int:
    from rcpp_core.graph_memory_repository import GraphMemoryRepository
    _json_dump(GraphMemoryRepository().get_memory_history(args.area, args.rps, args.kind))
    return 0


def command_memory_trace(args: argparse.Namespace) -> int:
    from rcpp_core.graph_memory_repository import GraphMemoryRepository
    paths = GraphMemoryRepository().get_evidence_path(args.run_id, args.rps)
    _json_dump({"run_id": args.run_id, "rps_id": args.rps, "evidence_paths": paths})
    return 0 if paths else 2


def command_memory_migrate(args: argparse.Namespace) -> int:
    from rcpp_core.graph_memory_repository import GraphMemoryRepository
    from rcpp_core.memory_migration import apply_migration, prepare_migration
    plan = prepare_migration(args.area, args.scene_file, args.combined_file)
    if args.apply:
        result = apply_migration(plan, GraphMemoryRepository())
    else:
        result = {k: v for k, v in plan.audit.items()
                  if k not in {"unmatched_raw_scene_images", "raw_anomaly_records", "repair_records"}}
        result.update(applied=False, mode="dry-run")
    if args.report:
        _write_json(Path(args.report), {**plan.audit, **result})
    _json_dump(result)
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
    if args.target == "retrieval":
        from evaluation.retrieval_ablation import run_from_args

        result = run_from_args(args)
        _json_dump({
            "output": str(Path(args.output).resolve()),
            "metadata": result["metadata"],
            "comparison": result["comparison"],
        })
        return 0
    if args.target == "retrieval-live":
        if not args.scene_file or not args.combined_file:
            raise ValueError("--scene-file and --combined-file are required")
        from evaluation.live_semantic_ablation import run_from_args

        _json_dump(run_from_args(args))
        return 0
    if args.target == "scene-api":
        if not args.image_dir or not args.combined_file:
            raise ValueError("--image-dir and --combined-file are required")
        from evaluation.scene_api_experiment import run_from_args

        _json_dump(run_from_args(args))
        return 0
    if args.target == "qwen3-route":
        if not args.calls_file or not args.combined_file:
            raise ValueError("--calls-file and --combined-file are required")
        from evaluation.qwen3_route_ablation import run_from_args

        _json_dump(run_from_args(args))
        return 0
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
    build = memory_sub.add_parser("build")
    build.add_argument("path")
    build.set_defaults(func=command_memory_build)
    inspect = memory_sub.add_parser("inspect")
    inspect.add_argument("--area", required=True)
    inspect.add_argument("--rps", required=True)
    inspect.set_defaults(func=command_memory_inspect)
    history = memory_sub.add_parser("history")
    history.add_argument("--area", required=True)
    history.add_argument("--rps", required=True)
    history.add_argument("--kind", choices=("scene", "semantic"), required=True)
    history.set_defaults(func=command_memory_history)
    trace = memory_sub.add_parser("trace")
    trace.add_argument("--run-id", required=True)
    trace.add_argument("--rps", required=True)
    trace.set_defaults(func=command_memory_trace)
    migrate = memory_sub.add_parser("migrate")
    migrate.add_argument("--area", required=True)
    migrate.add_argument("--scene-file", required=True)
    migrate.add_argument("--combined-file", required=True)
    mode = migrate.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    migrate.add_argument("--report")
    migrate.set_defaults(func=command_memory_migrate)

    plan = sub.add_parser("plan")
    plan_sub = plan.add_subparsers(dest="command", required=True)
    run = plan_sub.add_parser("run")
    run.add_argument("--area")
    run.add_argument("--scenario", choices=("efficiency_oriented", "equity_oriented", "balance_oriented"), default="balance_oriented")
    run.add_argument("--start-year", type=int, default=2025)
    run.add_argument("--end-year", type=int, default=2030)
    run.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    run.add_argument("--instruction", help="optional natural-language path; requires configured provider")
    run.add_argument("--weight-source", choices=("cached", "expert", "llm"), default="cached")
    run.add_argument("--shadow", action="store_true")
    run.add_argument("--run-id")
    run.add_argument("--max-iterations", type=int)
    run.add_argument("--latency-budget-ms", type=float)
    run.add_argument("--cost-budget-usd", type=float)
    run.add_argument("--routing-profiles")
    run.add_argument("--risk", choices=("low", "medium", "high"), default="medium")
    run.set_defaults(func=command_plan_run)
    resume = plan_sub.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--max-iterations", type=int)
    resume.set_defaults(func=command_plan_resume)
    status = plan_sub.add_parser("status")
    status.add_argument("run_id")
    status.set_defaults(func=command_plan_status)
    reviews = plan_sub.add_parser("reviews", help="show pending Scene review and its token")
    reviews.add_argument("run_id")
    reviews.set_defaults(func=command_plan_reviews)
    review = plan_sub.add_parser("review", help="submit Scene corrections/exclusions and resume")
    review.add_argument("run_id")
    review.add_argument("--file", required=True)
    review.add_argument("--max-iterations", type=int)
    review.set_defaults(func=command_plan_review)

    router = sub.add_parser("router")
    router_sub = router.add_subparsers(dest="command", required=True)
    explain = router_sub.add_parser("explain")
    explain.add_argument("run_id", nargs="?")
    explain.add_argument("--rps")
    explain.add_argument("--kind", choices=("scene", "semantic"), default="scene")
    explain.add_argument("--memory-state", choices=("miss", "fresh", "partial", "stale", "conflicted"), default="miss")
    explain.add_argument("--memory-quality", type=float, default=0.0)
    explain.add_argument("--risk", choices=("low", "medium", "high"), default="medium")
    explain.add_argument("--shadow", action="store_true")
    explain.add_argument("--latency-budget-ms", type=float)
    explain.add_argument("--cost-budget-usd", type=float)
    explain.add_argument("--routing-profiles")
    explain.set_defaults(func=command_router_explain)
    stats = router_sub.add_parser("stats")
    stats.set_defaults(func=command_router_stats)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument(
        "target", choices=(
            "memory", "router", "retrieval", "retrieval-live", "scene-api", "qwen3-route"
        )
    )
    evaluate.add_argument("--source", choices=("json", "neo4j"), default="json")
    evaluate.add_argument("--area", default="gulou")
    evaluate.add_argument("--scene-file")
    evaluate.add_argument("--combined-file")
    evaluate.add_argument("--image-dir")
    evaluate.add_argument("--old-lora-file")
    evaluate.add_argument("--calls-file", help="qwen3-route: scene_calls.jsonl from scene-api")
    evaluate.add_argument(
        "--historical-route-summary",
        help="qwen3-route: optional historical LoRA retrieval summary.json",
    )
    evaluate.add_argument("--cohort", choices=("detailed-active", "detailed-all", "all"), default="detailed-active")
    evaluate.add_argument("--warmup", type=int, default=2)
    evaluate.add_argument("--repeats", type=int, default=10)
    evaluate.add_argument("--tokenizer", help="Hugging Face tokenizer path or model id; omit for an explicitly labelled estimate")
    evaluate.add_argument("--bootstrap-samples", type=int, default=1000)
    evaluate.add_argument("--model", help="SiliconFlow model id for retrieval-live")
    evaluate.add_argument(
        "--max-calls", type=int, default=20,
        help="retrieval-live: 0 runs the full cohort; default 20",
    )
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--max-samples", type=int, default=20,
                          help="scene-api: 0 runs every available image")
    evaluate.add_argument("--workers", type=int, default=2)
    evaluate.add_argument("--image-detail", choices=("low", "high", "auto"), default="high")
    evaluate.add_argument("--retry-failures", action="store_true")
    evaluate.add_argument("--progress-every", type=int, default=10)
    evaluate.add_argument(
        "--execute", action="store_true",
        help="issue billable live API requests; omitted means dry-run",
    )
    evaluate.add_argument("--output", default="evaluation/results/retrieval_ablation")
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
