"""
LLM-based phased RCP allocation.
Assigns phase1, phase2, and phase3 RPS count shares for a given total of feasible sites.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from configs.config import (
    DEEPSEEK_API_BASE_URL,
    DEFAULT_LLM_AHP_TEMPERATURE,
    get_llm_ahp_model,
    get_openai_api_key,
)
from rcpp_core.prompts import phased_allocation_human_prompt
from tools.multi_scenario_evaluation_tool.ahp_common import parse_json_response

logger = logging.getLogger(__name__)


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


@dataclass
class PhasedAllocationTool:
    """
    Uses an LLM to choose proportions for phase1, phase2, and phase3 that sum to total_rps_count.
    """

    llm_provider: str = "openai"
    model_name: str = ""
    api_key: Optional[str] = None
    temperature: float = DEFAULT_LLM_AHP_TEMPERATURE

    def _get_llm(self) -> ChatOpenAI:
        name = (self.model_name or get_llm_ahp_model()).strip()
        key = (self.api_key or get_openai_api_key() or "").strip()
        if not key:
            raise RuntimeError("API key not set for phased allocation: set DEEPSEEK_API_KEY or OPENAI_API_KEY.")
        if self.llm_provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        if "deepseek" in name.lower():
            return ChatOpenAI.model_validate(
                {
                    "model": name,
                    "base_url": DEEPSEEK_API_BASE_URL,
                    "api_key": SecretStr(key),
                    "temperature": self.temperature,
                }
            )
        return ChatOpenAI.model_validate(
            {
                "model": name,
                "api_key": SecretStr(key),
                "temperature": self.temperature,
            }
        )

    def decide_allocation(self, scenario: str, total_rps_count: int) -> Dict[str, Any]:
        logger.info("LLM phased allocation: scenario=%s total=%s", scenario, total_rps_count)
        prompt = phased_allocation_human_prompt(scenario, total_rps_count)
        try:
            llm = self._get_llm()
            resp = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are an expert in refined roadside charging point RCP planning and "
                            "phased deployment under multi-scenario analysis. Output JSON only."
                        )
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            text = _stringify_message_content(resp.content)
            result = cast(Dict[str, Any], parse_json_response(text))
            allocation = cast(Dict[str, Any], result.get("allocation") or {})
            p1 = float(allocation.get("phase1", 0))
            p2 = float(allocation.get("phase2", 0))
            p3 = float(allocation.get("phase3", 0))
            total_ratio = p1 + p2 + p3
            if abs(total_ratio - 1.0) > 0.02 and total_ratio > 0:
                logger.warning("Phased allocation ratios sum to %s; normalizing.", total_ratio)
                p1, p2, p3 = p1 / total_ratio, p2 / total_ratio, p3 / total_ratio
                allocation = {"phase1": p1, "phase2": p2, "phase3": p3}
                result["allocation"] = allocation

            n = int(total_rps_count)
            allocation_counts = {
                "phase1": int(round(p1 * n)) if n else 0,
                "phase2": int(round(p2 * n)) if n else 0,
                "phase3": int(round(p3 * n)) if n else 0,
            }
            diff = n - sum(allocation_counts.values())
            if diff != 0 and allocation:
                max_phase = max(allocation, key=lambda k: allocation.get(k, 0))
                allocation_counts[str(max_phase)] += diff
            result["allocation_counts"] = allocation_counts
            result["scenario"] = scenario
            result["total_rps_count"] = n
            logger.info(
                "Phased allocation done: %s / counts %s",
                allocation,
                allocation_counts,
            )
            return result
        except Exception as e:
            logger.error("LLM phased allocation failed: %s", e)
            n = int(total_rps_count)
            default_allocation = {"phase1": 0.33, "phase2": 0.34, "phase3": 0.33}
            allocation_counts = {
                "phase1": int(round(0.33 * n)),
                "phase2": int(round(0.34 * n)),
                "phase3": int(round(0.33 * n)),
            }
            allocation_counts["phase2"] += n - sum(allocation_counts.values())
            return {
                "scenario": scenario,
                "total_rps_count": n,
                "allocation": default_allocation,
                "allocation_counts": allocation_counts,
                "reasoning": f"LLM decision failed; using default balanced split. Error: {e!s}",
            }
