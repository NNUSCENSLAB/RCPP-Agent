"""
Task Orchestration Agent: reads the global planning instruction, breaks it into explicit
sub-tasks, and hands off spatial and orchestration context to downstream agents.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, SecretStr, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.config import (
    DEEPSEEK_API_BASE_URL,
    DEFAULT_INSTRUCTION_PARSE_TEMPERATURE,
    get_instruction_parse_model,
    get_provider_api_key,
    provider_for_model,
)
from configs.data_config import (
    DEFAULT_SPATIALITE_TABLES,
    REL_DEFAULT_DISTRICT_DATA,
    default_spatialite_db_path,
)
from rcpp_core.prompts import INSTRUCTION_PARSE_SYSTEM_PROMPT
from tools.rcpp_paths import DEFAULT_WORKSPACE, normalize_area_slug

logger = logging.getLogger(__name__)

DEFAULT_DB = str(default_spatialite_db_path())


def _json_object_from_llm_text(text: str) -> Dict[str, Any]:
    """
    Parse a single JSON object from model output.
    """
    raw = text.strip()
    try:
        out = json.loads(raw)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    block = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if block:
        out = json.loads(block.group(1))
        if isinstance(out, dict):
            return out
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        out = json.loads(raw[start : end + 1])
        if isinstance(out, dict):
            return out
    raise ValueError("No JSON object found in LLM response")


def _resolve_data_path(path: str | Path, workspace_dir: Path) -> str:
    p = Path(path).expanduser()
    return str((p if p.is_absolute() else workspace_dir / p).resolve())


class ParsedPlanningInstruction(BaseModel):
    """Structured extraction from a single-sentence user instruction."""

    area: str = Field(description="Study area as a short slug, e.g. gulou")
    time_horizon: List[int] = Field(
        description="Exactly two integers: [start_year, end_year], start_year <= end_year"
    )
    scenario: Literal["efficiency_oriented", "equity_oriented", "balance_oriented"] = Field(
        description="Decision scenario slug"
    )

    @field_validator("time_horizon")
    @classmethod
    def exactly_two_years(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("time_horizon must contain exactly two integers")
        if int(v[0]) > int(v[1]):
            raise ValueError("time_horizon start_year must be <= end_year")
        return [int(v[0]), int(v[1])]


class TaskOrchestrationAgent:
    def __init__(self, data_registry: Optional[Dict[str, Dict[str, Any]]] = None):
        self.data_registry = data_registry or {
            "gulou": {
                "spatialite_db": DEFAULT_DB,
                "tables": {},
                "area_data_root": REL_DEFAULT_DISTRICT_DATA.as_posix(),
            }
        }

    def _resolve_area_entry(self, raw_area: str) -> Tuple[Dict[str, Any], str]:
        slug = normalize_area_slug(raw_area)
        if slug in self.data_registry:
            return self.data_registry[slug], slug
        if raw_area in self.data_registry:
            return self.data_registry[raw_area], normalize_area_slug(raw_area)
        return {}, slug

    def parse_natural_language_to_global_instruction(
        self,
        text: str,
        *,
        workspace_dir: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse one natural-language sentence into a global_instruction dict for the RCPP-Agent run.
        """
        stripped = (text or "").strip()
        if not stripped:
            raise ValueError("Natural language instruction is empty.")

        model_name = model or get_instruction_parse_model()
        provider = provider_for_model(model_name)
        api_key = get_provider_api_key(provider)
        if not api_key:
            raise RuntimeError(
                f"API key is not set for provider {provider!r}; cannot parse natural language instruction."
            )
        registry_slugs = list(self.data_registry.keys())
        areas_line = ", ".join(registry_slugs) if registry_slugs else "gulou"

        system = INSTRUCTION_PARSE_SYSTEM_PROMPT.format(known_area_slugs=areas_line)

        llm_kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": DEFAULT_INSTRUCTION_PARSE_TEMPERATURE,
            "api_key": SecretStr(api_key),
        }
        if provider == "deepseek":
            llm_kwargs["base_url"] = DEEPSEEK_API_BASE_URL
        llm = ChatOpenAI(**llm_kwargs)
        try:
            msg = llm.invoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=stripped),
                ]
            )
            raw_content = msg.content
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                pieces: List[str] = []
                for p in raw_content:
                    if isinstance(p, str):
                        pieces.append(p)
                    elif isinstance(p, dict) and isinstance(p.get("text"), str):
                        pieces.append(p["text"])
                    else:
                        pieces.append(str(p))
                content = "".join(pieces)
            else:
                content = str(raw_content)
            obj = _json_object_from_llm_text(content)
            parsed = ParsedPlanningInstruction.model_validate(obj)
        except Exception as e:
            raise RuntimeError(f"Failed to parse planning instruction: {e}") from e

        area_slug = normalize_area_slug(parsed.area)
        th = [int(parsed.time_horizon[0]), int(parsed.time_horizon[1])]
        year = th[0]
        ws = str(workspace_dir or DEFAULT_WORKSPACE)

        return {
            "area": area_slug,
            "time_horizon": th,
            "scenario": parsed.scenario,
            "workspace_dir": ws,
            "year": year,
        }

    def orchestrate(self, global_instruction: Dict[str, Any]) -> Dict[str, Any]:
        print("Task Orchestration Agent running...", flush=True)
        logger.info("Task Orchestration Agent received global planning instruction: %s", global_instruction)

        raw_area = global_instruction.get("area", "gulou")
        area_entry, area_slug = self._resolve_area_entry(str(raw_area))

        scenario = global_instruction.get("scenario", "equity_oriented")
        time_horizon = global_instruction.get("time_horizon")
        workspace_dir = str(
            global_instruction.get("workspace_dir") or DEFAULT_WORKSPACE
        )
        workspace_path = Path(workspace_dir)

        if not area_entry:
            logger.warning("No data registration found for area %s (slug=%s); using default database path.", raw_area, area_slug)
            area_entry = {"spatialite_db": DEFAULT_DB, "tables": {}}

        db_raw = area_entry.get("spatialite_db", DEFAULT_DB)
        db_p = Path(str(db_raw)).expanduser()
        if not db_p.is_absolute():
            db_p = (workspace_path / db_p).resolve()
        else:
            db_p = db_p.resolve()
        tables = {**DEFAULT_SPATIALITE_TABLES, **area_entry.get("tables", {})}

        spatialite_context: Dict[str, Any] = {
            "db_path": str(db_p),
            "area": area_slug,
            "tables": tables,
            "scenario": scenario,
            "time_horizon": time_horizon,
            "workspace_dir": workspace_dir,
        }
        rsp = area_entry.get("rps_shp_path")
        if rsp:
            spatialite_context["rps_shp_path"] = rsp

        adr = area_entry.get("area_data_root") or area_entry.get("district_data_root")
        adr_resolved: Optional[str] = None
        if adr:
            adr_resolved = _resolve_data_path(adr, workspace_path)
            spatialite_context["area_data_root"] = adr_resolved

        poi_dir = area_entry.get("poi_dir")
        if poi_dir:
            spatialite_context["poi_dir"] = _resolve_data_path(poi_dir, workspace_path)
        elif adr_resolved:
            spatialite_context["poi_dir"] = str(Path(adr_resolved) / "POI")

        existing_rcp = area_entry.get("existing_rcp")
        if existing_rcp:
            spatialite_context["existing_rcp"] = _resolve_data_path(existing_rcp, workspace_path)

        road_network_shp = area_entry.get("road_network_shp")
        if road_network_shp:
            spatialite_context["road_network_shp"] = _resolve_data_path(
                road_network_shp, workspace_path
            )

        logger.info(
            "Task decomposition completed: area=%s scenario=%s time_horizon=%s -> spatialite_context is ready",
            area_slug,
            scenario,
            time_horizon,
        )

        return {
            "spatialite_context": spatialite_context,
            "scenario": scenario,
            "time_horizon": time_horizon,
            "workspace_dir": workspace_dir,
            "workflow_status": "task_orchestrated",
        }
