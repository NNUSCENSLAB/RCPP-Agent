# -*- coding: utf-8 -*-
"""
LLM-AHP LangGraph subgraph shared by the Multi-Scenario Evaluation Agent and the
Phased Decision-Making Agent.

The subgraph takes a ``kind`` field ("scenario" or "phase") to dispatch between
two weighting modes:

* ``kind="scenario"``: derive scenario-level MCDA weights from a scenario slug.
* ``kind="phase"``: derive phase-specific criterion weights from the active
  scenario, scenario-level weights, time horizon, and deployment phase.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from configs.config import (
    LLM_AHP_MAX_RETRIES,
    MCDA_DIMENSIONS,
    PLANNING_PHASE,
    PlanningPhase,
)
from rcpp_core.prompts import (
    LLM_AHP_PHASE_PREAMBLE,
    LLM_AHP_PRINCIPLES_PHASE,
    LLM_AHP_PRINCIPLES_SCENARIO,
    LLM_AHP_PROMPT_INTRO,
    LLM_AHP_SAATY_BLOCK,
    LLM_AHP_SCENARIO_PREAMBLE,
    MCDA_INDICATOR_DESCRIPTION_PROMPT,
    PHASE_OVERVIEW,
    PHASE_PROMPT,
    all_scenarios_reference_block,
    llm_ahp_output_format_block_phase,
    llm_ahp_output_format_block_scenario,
    llm_ahp_reasoning_requirements_phase,
    llm_ahp_reasoning_requirements_scenario,
    llm_ahp_special_considerations_phase,
    llm_ahp_special_considerations_scenario,
    scenario_narrative,
)
from tools.base import BaseTool, ToolCategory, ToolContext
from tools.multi_scenario_evaluation_tool.ahp_common import (
    ahp_weights_from_matrix,
    build_chat_openai,
    equal_dimension_weights,
    normalize_dimension_weights,
    parse_json_response,
)
from tools.registry import register_tool

logger = logging.getLogger(__name__)


def _stringify_message_content(content: Any) -> str:
    """
    Convert a LangChain message content object to a string.
    """
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


LLMAHPKind = Literal["scenario", "phase"]

_SYSTEM_MESSAGE = (
    "You are a senior expert in fine-grained roadside charging point (RCP) planning. "
    "Respond with one JSON object only (no markdown, no prose outside JSON). "
    "Keys must be exactly: pairwise_matrix (5x5 array of JSON numbers, Saaty 1-9 scale, reciprocal, diagonal 1) "
    "and reasoning (string). The user prompt specifies criterion order for rows/columns."
)


class LLMAHPState(TypedDict, total=False):
    kind: LLMAHPKind
    scenario: str
    weight_revision: str
    scenario_weights: Dict[str, float]
    time_horizon: Any
    phase: Union[PlanningPhase, str]
    scene_semantic_context: str

    attempts: int
    pairwise_matrix: Any
    llm_reasoning: str
    last_error: str

    weights: Dict[str, float]
    consistency_ratio: Optional[float]
    reasoning: str


def _coerce_planning_phase(phase: Union[str, PlanningPhase]) -> PlanningPhase:
    if isinstance(phase, PlanningPhase):
        return phase
    if isinstance(phase, str):
        try:
            return PlanningPhase[phase]
        except KeyError:
            for member in PlanningPhase:
                if member.value == phase or member.name == phase:
                    return member
    raise TypeError(f"phase must be PlanningPhase or str, got {type(phase)!r}")


def _format_time_horizon(time_horizon: Any) -> str:
    if time_horizon is None:
        return "(not specified)"
    if isinstance(time_horizon, (list, tuple)) and len(time_horizon) >= 2:
        return f"{time_horizon[0]}\u2013{time_horizon[1]}"
    return str(time_horizon)


def _scenario_weights_block(scenario_weights: Dict[str, float]) -> str:
    lines: List[str] = []
    for k in MCDA_DIMENSIONS:
        v = float(scenario_weights.get(k, 0.0))
        lines.append(f"- {k}: {v:.6f}")
    return "\n".join(lines)


def _build_scenario_prompt(
    scenario: str,
    weight_revision: str,
    scene_semantic_context: str = "",
) -> str:
    dims = ", ".join(MCDA_DIMENSIONS.keys())
    pool_block = ""
    if (scene_semantic_context or "").strip():
        pool_block = f"""
## II-a. Site pool (scene-semantic store after suitability — qualitative context only)

{scene_semantic_context.strip()}

Do not invent site IDs or counts beyond this summary; use it only to calibrate criterion emphasis if relevant.
"""
    return f"""{LLM_AHP_SCENARIO_PREAMBLE}{LLM_AHP_PROMPT_INTRO}

{LLM_AHP_SAATY_BLOCK}

{LLM_AHP_PRINCIPLES_SCENARIO}

{llm_ahp_special_considerations_scenario(scenario)}

# RCP multi-scenario MCDA-AHP task

Derive criterion importance weights for **fine-grained roadside charging pile (RCP) planning** under multi-scenario analysis.

## I. All policy scenarios (reference—contrast primary goals, SDGs, and strategic focus)

{all_scenarios_reference_block()}

## II. Selected scenario (the pairwise matrix must reflect **only** this scenario's goals)

{scenario_narrative(scenario)}
{pool_block}
## III. Decision requirements

1. Build a 5*5 **pairwise comparison matrix** for the five criteria, **for Section II only** (do not blend other scenarios into the numerics).
2. Apply Saaty's 1-9 scale and reciprocity rules from the **Saaty's 1-9 scale** section above (row vs column criterion dominance).
3. The matrix must be **reciprocal** (a_ij = 1 / a_ji) with diagonal entries 1.
4. Row and column order must be exactly: {dims}.

## Weight revision note (optional adjustment)

{weight_revision.strip() if weight_revision else "(none)"}

{llm_ahp_reasoning_requirements_scenario(scenario)}

{llm_ahp_output_format_block_scenario(dims)}
"""


def _build_phase_prompt(
    scenario: str,
    scenario_weights: Dict[str, float],
    time_horizon: Any,
    phase: PlanningPhase,
    weight_revision: str,
) -> str:
    phase_key = phase.value
    phase_title = PLANNING_PHASE.get(phase_key, phase_key)
    phase_detail = PHASE_PROMPT.get(phase_key, "")
    dims = ", ".join(MCDA_DIMENSIONS.keys())
    return f"""{LLM_AHP_PHASE_PREAMBLE}{LLM_AHP_PROMPT_INTRO}

{LLM_AHP_SAATY_BLOCK}

{LLM_AHP_PRINCIPLES_PHASE}

{llm_ahp_special_considerations_phase(scenario, phase_title)}

# RCP phased MCDA-AHP task (scenario-level weights \u2192 phase-specific criterion weights)

{PHASE_OVERVIEW}

Multi-scenario evaluation has already produced **five-dimensional MCDA weights** for each policy scenario. The pipeline now runs **once per (scenario, phase)**. Your job: produce **final** criterion weights for **this** scenario in **this** deployment phase by building a new pairwise matrix that **respects the numeric scenario prior** and **shifts emphasis** according to the phase's positioning, objectives, and weight-consideration logic (below).

## I. All policy scenarios (reference\u2014three-scenario context)

{all_scenarios_reference_block()}

## II. Active scenario for this call (matrix must be consistent with this slug and its goals)

- **Scenario slug**: `{scenario}`
- **Narrative**:
{scenario_narrative(scenario)}

## III. Scenario-level MCDA weights (numeric prior from multi-scenario evaluation for **this** scenario)

These values summarize how the active scenario already trades off **technical, economic, social, traffic, policy**. You must **integrate** them with the phase detail in Section V\u2014**not** ignore or replace them arbitrarily; the final matrix should be a reasoned refinement for the current phase.

{_scenario_weights_block(scenario_weights)}

## IV. Planning time horizon (calendar years, inclusive range)

{_format_time_horizon(time_horizon)}

## V. Current deployment phase (full framing from the phased RCP framework)

- **Phase** (enum value): `{phase_key}`
- **Phase with calendar band**: **{phase_title}**

{phase_detail}

## VI. Five criteria \u2014 detailed definitions and weight considerations (legacy RCP reference)

{MCDA_INDICATOR_DESCRIPTION_PROMPT}

## VII. Criteria (order for the matrix)

{dims}

## VIII. Weight revision note (optional)

{weight_revision.strip() if weight_revision else "(none)"}

## IX. Task

Build a 5*5 **pairwise comparison matrix** that yields **final** weights for RCP site prioritization under **the active scenario** and **this phase**, informed by Sections III, V, and VI. Apply Saaty's 1-9 scale from the dedicated section above; matrix reciprocal with diagonal 1; criterion order as in Section VII.

{llm_ahp_reasoning_requirements_phase(scenario, phase_title)}

{llm_ahp_output_format_block_phase(dims)}
"""


def _build_prompt(state: LLMAHPState) -> str:
    raw_kind = state.get("kind")
    if raw_kind not in ("scenario", "phase"):
        raise ValueError(f"Unsupported or missing LLM-AHP kind: {raw_kind!r}")
    kind: LLMAHPKind = raw_kind
    revision = state.get("weight_revision", "") or ""
    if kind == "scenario":
        scenario = state.get("scenario")
        if scenario is None:
            raise ValueError("LLM-AHP scenario mode requires 'scenario' in state")
        ctx = str(state.get("scene_semantic_context") or "")
        return _build_scenario_prompt(scenario, revision, ctx)
    phase_raw = state.get("phase")
    if phase_raw is None:
        raise ValueError("LLM-AHP phase mode requires 'phase' in state")
    phase = _coerce_planning_phase(phase_raw)
    prior = normalize_dimension_weights(state.get("scenario_weights") or {})
    scenario = state.get("scenario")
    if scenario is None:
        raise ValueError("LLM-AHP phase mode requires 'scenario' in state")
    return _build_phase_prompt(
        scenario,
        prior,
        state.get("time_horizon"),
        phase,
        revision,
    )


def _invoke_llm_node(state: LLMAHPState) -> Dict[str, Any]:
    """
    Build the prompt, call the LLM, parse pairwise_matrix and reasoning.
    """
    attempts = int(state.get("attempts", 0)) + 1
    try:
        prompt = _build_prompt(state)
        llm = build_chat_openai()
        resp = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_MESSAGE),
                HumanMessage(content=prompt),
            ]
        )
        content = _stringify_message_content(resp.content)
        data = parse_json_response(content)
        matrix = data.get("pairwise_matrix")
        if not isinstance(matrix, list) or len(matrix) != len(MCDA_DIMENSIONS):
            raise ValueError("pairwise_matrix must be a 5x5 list")
        for row in matrix:
            if not isinstance(row, list) or len(row) != len(MCDA_DIMENSIONS):
                raise ValueError("each matrix row must have length 5")
        return {
            "attempts": attempts,
            "pairwise_matrix": matrix,
            "llm_reasoning": str(data.get("reasoning", "")).strip(),
            "last_error": "",
        }
    except Exception as e:
        logger.warning(
            "LLM-AHP LLM-call attempt %s/%s failed (kind=%s): %s",
            attempts,
            LLM_AHP_MAX_RETRIES,
            state.get("kind"),
            e,
        )
        return {
            "attempts": attempts,
            "pairwise_matrix": None,
            "llm_reasoning": "",
            "last_error": str(e),
        }


def _fallback_weights(state: LLMAHPState) -> Dict[str, float]:
    if state.get("kind") == "phase":
        return normalize_dimension_weights(state.get("scenario_weights") or {})
    return equal_dimension_weights()


def _solve_ahp_node(state: LLMAHPState) -> Dict[str, Any]:
    """Solve AHP weights from the pairwise matrix and validate consistency."""
    matrix = state.get("pairwise_matrix")
    if not matrix:
        return {
            "last_error": state.get("last_error") or "missing pairwise_matrix",
        }
    try:
        weights_list, cr = ahp_weights_from_matrix(cast(List[List[float]], matrix))
        if cr > 0.1:
            return {
                "consistency_ratio": cr,
                "last_error": (
                    f"AHP consistency ratio {cr:.4f} exceeds Saaty threshold 0.1"
                ),
            }
        weights = {k: float(v) for k, v in zip(MCDA_DIMENSIONS.keys(), weights_list)}
        llm_reason = state.get("llm_reasoning", "")
        reasoning = (
            f"LLM-AHP {state.get('kind')} (CR={cr:.4f}). {llm_reason}"
        ).strip()
        return {
            "weights": weights,
            "consistency_ratio": cr,
            "reasoning": reasoning,
            "last_error": "",
        }
    except Exception as e:
        logger.warning("LLM-AHP solve_ahp failed: %s", e)
        return {"last_error": str(e)}


def _retry_or_finish(state: LLMAHPState) -> Literal["retry", "finish"]:
    if state.get("weights"):
        return "finish"
    if int(state.get("attempts", 0)) >= LLM_AHP_MAX_RETRIES:
        return "finish"
    return "retry"


def _finalize_node(state: LLMAHPState) -> Dict[str, Any]:
    """
    Ensure a weights output exists, using fallback if retries are exhausted.
    """
    if state.get("weights"):
        return {}
    fallback = _fallback_weights(state)
    last_err = state.get("last_error", "")
    kind = state.get("kind")
    logger.warning(
        "LLM-AHP %s failed after %s attempts; applying fallback weights. Last error: %s",
        kind,
        LLM_AHP_MAX_RETRIES,
        last_err,
    )
    reasoning_prefix = (
        "Normalized scenario_mcda_weights fallback"
        if kind == "phase"
        else "Equal weights fallback"
    )
    return {
        "weights": fallback,
        "consistency_ratio": None,
        "reasoning": f"{reasoning_prefix} after LLM failure: {last_err}",
        "pairwise_matrix": None,
    }


def build_llm_ahp_subgraph():
    """
    Compile and return the LLM-AHP subgraph.
    """
    graph = StateGraph(LLMAHPState)
    graph.add_node("invoke_llm", _invoke_llm_node)
    graph.add_node("solve_ahp", _solve_ahp_node)
    graph.add_node("finalize", _finalize_node)

    graph.add_edge(START, "invoke_llm")
    graph.add_edge("invoke_llm", "solve_ahp")
    graph.add_conditional_edges(
        "solve_ahp",
        _retry_or_finish,
        {"retry": "invoke_llm", "finish": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


@register_tool
class LLMAHPTool(BaseTool):
    """
    Unified LLM-AHP weighting tool backed by a LangGraph subgraph.

    Dispatches between scenario-level and phase-specific weighting via the
    ``kind`` argument. Output schema:
    ``{"weights": {...}, "pairwise_matrix": ..., "consistency_ratio": ..., "reasoning": "..."}``.
    """

    name = "llm_ahp"
    category = ToolCategory.DECISION_WEIGHTS

    def __init__(self) -> None:
        self._subgraph = build_llm_ahp_subgraph()

    @property
    def subgraph(self):
        return self._subgraph

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        kind: LLMAHPKind,
        scenario: str,
        weight_revision: str = "",
        scene_semantic_context: str = "",
        scenario_weights: Optional[Dict[str, float]] = None,
        time_horizon: Any = None,
        phase: Union[PlanningPhase, str, None] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if kind not in ("scenario", "phase"):
            raise ValueError(f"kind must be 'scenario' or 'phase', got {kind!r}")
        if kind == "phase":
            if scenario_weights is None:
                raise ValueError("scenario_weights is required when kind='phase'")
            if phase is None:
                raise ValueError("phase is required when kind='phase'")

        initial: LLMAHPState = {
            "kind": kind,
            "scenario": scenario,
            "weight_revision": weight_revision or "",
            "attempts": 0,
        }
        if kind == "scenario" and (scene_semantic_context or "").strip():
            initial["scene_semantic_context"] = scene_semantic_context.strip()
        if scenario_weights is not None:
            initial["scenario_weights"] = scenario_weights
        if time_horizon is not None:
            initial["time_horizon"] = time_horizon
        if phase is not None:
            initial["phase"] = phase

        final_state = self._subgraph.invoke(initial)
        out = {
            "weights": final_state.get("weights"),
            "pairwise_matrix": final_state.get("pairwise_matrix"),
            "consistency_ratio": final_state.get("consistency_ratio"),
            "reasoning": final_state.get("reasoning", ""),
        }
        if ctx is not None:
            ctx.set(f"llm_ahp_{kind}_result", out)
        return out


__all__ = [
    "LLMAHPKind",
    "LLMAHPState",
    "LLMAHPTool",
    "build_llm_ahp_subgraph",
]
