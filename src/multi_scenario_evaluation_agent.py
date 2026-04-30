"""
Multi-Scenario Evaluation Agent: runs MCDA with an LLM-AHP tool to produce dynamic criteria
weights for SDG-aligned scenarios.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional, cast

from tools.llm_ahp_tool import LLMAHPTool

logger = logging.getLogger(__name__)


def _digest_scene_semantic_memory(memories: Optional[Dict[str, Any]]) -> str:
    """
    Short deterministic summary of the global scene-semantic dict for LLM context and cache keys.
    """
    if not memories:
        return "EMPTY"
    scored = 0
    rejected = 0
    for _rid, node in memories.items():
        if not isinstance(node, dict):
            continue
        ann = node.get("agent_annotations")
        if not isinstance(ann, dict):
            continue
        sa = ann.get("SuitabilityAssessmentAgent")
        if not isinstance(sa, dict):
            continue
        if "suitability_score" in sa:
            scored += 1
        elif "suitability_rejected" in sa:
            rejected += 1
    return (
        f"rps_entries={len(memories)} suitability_scored={scored} suitability_rejected={rejected}"
    )


def _scenario_weight_cache_key(
    scenario: str,
    weight_revision: int,
    revision_note: str,
    memory_digest: str = "",
) -> str:
    """
    Build the cache label from scenario name, review round index, and any review feedback text.
    """
    base = f"{scenario}_r{int(weight_revision)}"
    note = (revision_note or "").strip()
    if not note:
        key = base
    else:
        digest = hashlib.sha256(note.encode("utf-8")).hexdigest()[:16]
        key = f"{base}_n{digest}"
    if (memory_digest or "").strip() and memory_digest != "EMPTY":
        h = hashlib.sha256(memory_digest.encode("utf-8")).hexdigest()[:12]
        key = f"{key}_mem{h}"
    return key


class MultiScenarioEvaluationAgent:
    """
    Computes scenario-level MCDA weights for the phased planning so downstream
    agents can rank RPSs in line with efficiency, equity, or balance-oriented scenarios.
    """

    def __init__(self, llm_ahp_tool: Optional[LLMAHPTool] = None) -> None:
        self._llm_ahp_tool = llm_ahp_tool
        self._llm_ahp_scenario_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def llm_ahp_tool(self) -> LLMAHPTool:
        """
        Weighting tool used to run the LLM-AHP step.
        """
        if self._llm_ahp_tool is None:
            self._llm_ahp_tool = LLMAHPTool()
        return self._llm_ahp_tool

    def _cached_llm_ahp_scenario(
        self,
        scenario: str,
        weight_revision: int,
        revision_note: str,
        memory_digest: str,
    ) -> Dict[str, Any]:
        """
        Serve LLM-AHP scenario output from cache when the same context was already computed;
        otherwise invoke the tool once and cache weights, reasoning, matrix, and CR.
        """
        cache_key = _scenario_weight_cache_key(
            scenario, weight_revision, revision_note, memory_digest
        )
        if cache_key in self._llm_ahp_scenario_cache:
            return dict(self._llm_ahp_scenario_cache[cache_key])

        logger.info(
            "LLM-AHP subgraph invoke (scenario=%s, revision=%s).",
            scenario,
            weight_revision,
        )

        rev_note = revision_note or (str(weight_revision) if weight_revision else "")
        digest_for_llm = memory_digest if memory_digest != "EMPTY" else ""
        result = self.llm_ahp_tool.run(
            kind="scenario",
            scenario=scenario,
            weight_revision=rev_note,
            scene_semantic_context=digest_for_llm,
        )
        snapshot: Dict[str, Any] = {
            "weights": cast(Dict[str, float], result.get("weights") or {}),
            "reasoning": str(result.get("reasoning", "")).strip(),
            "pairwise_matrix": result.get("pairwise_matrix"),
            "consistency_ratio": result.get("consistency_ratio"),
        }
        self._llm_ahp_scenario_cache[cache_key] = dict(snapshot)
        return snapshot

    def determine_weights(
        self,
        scenario: str,
        *,
        weight_revision: int = 0,
        revision_note: str = "",
        scene_semantic_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        MCDA dimension weights via LLM-AHP subgraph.
        """
        print("Multi-Scenario Evaluation Agent running...", flush=True)
        memory_digest = _digest_scene_semantic_memory(scene_semantic_memory)
        logger.info(
            "Multi-Scenario Evaluation Agent computing scenario weights (scenario=%s, revision=%s, store=%s).",
            scenario,
            weight_revision,
            memory_digest,
        )

        llm_block = self._cached_llm_ahp_scenario(
            scenario, weight_revision, revision_note, memory_digest
        )
        weights = cast(Dict[str, float], llm_block["weights"])
        reasoning = str(llm_block.get("reasoning", "")).strip()
        logger.info("Scenario weights calculation completed: %s", weights)
        return {
            "scenario_mcda_weights": weights,
            "scenario": scenario,
            "reasoning": reasoning,
            "pairwise_matrix": llm_block.get("pairwise_matrix"),
            "consistency_ratio": llm_block.get("consistency_ratio"),
            "llm_ahp_scenario": llm_block,
        }
