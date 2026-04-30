# -*- coding: utf-8 -*-
"""Proprietary prompt constants and scenario-narrative helpers for RCPP-Agent."""
from __future__ import annotations

from typing import Dict, List
from configs.config import MCDA_DIMENSIONS, MULTI_SCENARIO


INSTRUCTION_PARSE_SYSTEM_PROMPT = """You extract structured planning parameters from the user's single-sentence instruction.
Rules:
- area: Short study-area slug. Known registry keys: {known_area_slugs}. Map common names (e.g. Nanjing Gulou, Gulou district) to the matching slug (e.g. gulou).
- time_horizon: Exactly two integers [start_year, end_year], start_year <= end_year. If years are missing, use [2025, 2030].
- scenario: One of efficiency_oriented, equity_oriented, balance_oriented. Map phrases like efficiency, equity, balanced, balance to the correct *_oriented value.

Return ONLY one JSON object (no markdown, no explanation) with keys exactly: "area", "time_horizon", "scenario".
Example: {{"area": "gulou", "time_horizon": [2025, 2030], "scenario": "equity_oriented"}}
"""

NATURAL_LANGUAGE_USER_PROMPT_EN = """
Hello, and welcome to RCPP-Agent, a multi-agent architecture for fine-grained roadside charging pile (RCP) planning.

Enter one sentence that specifies:
  - Study area (e.g. Gulou District, Nanjing City, Jiangsu Province, China)
  - Planning time horizon (start and end years)
  - Scenario: efficiency-oriented, equity-oriented, or balance-oriented

Example: Plan roadside charging piles in Gulou District, Nanjing City, Jiangsu Province, China from 2025 to 2030 under an equity-oriented scenario.

Please enter your instruction:
""".strip()


# Environment Perception Agent: Scene-semantic memory prompts

SCENE_SYSTEM_PROMPT = """You are an expert in roadside parking-stall scene perception for fine-grained roadside EV charging planning.
Focus ONLY on the RED-BOXED roadside parking stall and its lateral zone.
Follow the exclusionary priority: (1) existing RCP, (2) functional zone, (3) obstacle count ≥ 3.

Output ONLY a JSON with this exact schema (no extra text):
{
  "has_existing_RCP": boolean,
  "is_functional_zone": boolean,
  "functional_zone_type": "none" | "bicycle_parking_area" | "bus_stop" | "greenbelt",
  "has_ground_obstacle": boolean,
  "ground_obstacle_types": "pole" | "fire_hydrant" | "tree_trunk" | "cabinet" | "bicycle_rack" | "sign_post" | "shrubbery" | "none",
  "ground_obstacle_count": integer,
  "visual_distractors_noted": ["none" | "tree_canopy" | "greenbelt"],
  "clearance_visual_assessment": boolean,
  "scene_reasoning": "concise reasoning referencing the RED-BOXED STALL",
  "confidence_score": 5
}

Do not mention non-red-box zones. Do not output anything outside the JSON."""


SEMANTIC_SYSTEM_PROMPT = """You are an expert in roadside parking stall (RPS) semantic reasoning for fine-grained EV charging planning.

Goal
Transform quantitative Statistical Summaries into a structured semantic_reasoning JSON object by applying the Expert Rule Set.

Expert Rule Set (Mandatory Logic)
Functional Zone Type (Priority: Residential vs Commercial ratio)
Residential-Oriented: Residential count > 1.5 * Commercial count AND Residential count >= 5. (Focus: Overnight slow charging).
Commercial-Oriented: Commercial count > 1.5 * Residential count AND Commercial count >= 5. (Focus: Daytime fast charging).
Mixed Functional Type: Both Residential count >= 5 and Commercial count >= 5. (Focus: Stable all-day load).
Low Density: Both Residential and Commercial counts < 5.

Commuting Flow (Total Flow = Inflow + Outflow)
High Traffic: Total Flow > 100,000. (High turnover, roadside hotspots).
Medium Traffic: Total Flow between 50,000 and 100,000. (Steady flow, standard load).
Low Traffic: Total Flow < 50,000. (Low activity, destination-based).
Sensitive Constraints (Sum = Medical + Education + Public POIs)
High Risk: Sum >= 2. (Flag emergency lanes, school safety, strict controls).
Low Risk: 0 < Sum < 2. (Minor compliance risks).
No Risk: Sum = 0.

Grid Accessibility (Distance = Grid Distance)
Excellent: Distance < 500m. (Low cost, high power support).
General: Distance between 1000m and 2000m. (Moderate cost).
Very Poor: Distance >= 5000m. (Extreme engineering difficulty).

Input mapping (from the Statistical Summary JSON in the user message)
- Residential / commercial counts: poi_counts_1km.residential, poi_counts_1km.commercial.
- Sensitive sum: add medical + education + government_public (or equivalent public categories) from poi_counts_sensitive_0.1km when listed separately; if only a single aggregate is provided, use it as the sum.
- Total Flow: traffic_flow_stats.total_inflow + traffic_flow_stats.total_outflow.
- Grid distance in metres: use infrastructure_dist.grid_distance_km * 1000 when grid_distance_km is non-negative; apply Grid Accessibility rules to that distance in metres.

Output JSON Schema
Output ONLY valid JSON:
{
  "rps_id": "string",
  "functional_zone_type": "Qualitative definition + charging strategy + temporal load prediction.",
  "commuting_flow": "Traffic volume assessment + potential for turnover/topping-up demand.",
  "sensitive_constraints": "Risk level (High/Low/No) + specific impact of sensitive POIs on compliance.",
  "grid_accessibility": "Accessibility rating + engineering cost/feasibility implication.",
  "bsv_image": "copy filename from input Statistical Summary if present, else null"
}

Writing Style Requirements
Reasoning must be synthesized: "Based on [Data], the area is [Type], therefore [Inference]."
Use clear, logical descriptions to explain the relationship between data and the final inference.
Ensure the semantic_reasoning fields follow the logic of the scenario examples provided in the Expert Rule Set.
No extra text outside the JSON."""


# Multi-scenario & Phased Decision-Making Agents: LLM-AHP shared prompt intro

LLM_AHP_PROMPT_INTRO = (
    "The workflow uses multi-criteria decision analysis (MCDA) with the Analytic Hierarchy Process (AHP) "
    "to derive **relative importance weights** for five criteria used to evaluate and prioritize RCP sites. "
    "Keep all judgments specific to **urban RCP deployment** (curbside / roadside charging infrastructure), "
    "policy compliance, and SDG-aligned planning—not abstract generic MCDA.\n\n"
    "Criteria (pairwise matrix **row and column order** must be exactly this sequence):\n"
    f"- **technical** ({MCDA_DIMENSIONS['technical']}): site engineering feasibility, construction and "
    "operational constraints, grid/interface fit, and technical risk for roadside deployment.\n"
    f"- **economic** ({MCDA_DIMENSIONS['economic']}): lifecycle cost-effectiveness, utilization and revenue "
    "logic, ROI and resource-efficiency of RCP investment.\n"
    f"- **social** ({MCDA_DIMENSIONS['social']}): equitable access, underserved areas, user inclusion and "
    "social acceptance of charging service.\n"
    f"- **traffic** ({MCDA_DIMENSIONS['traffic']}): road capacity, safety, and traffic-flow impacts of "
    "curbside charging siting.\n"
    f"- **policy** ({MCDA_DIMENSIONS['policy']}): permits, regulations, standards, and alignment with urban "
    "planning and energy/charging policy frameworks.\n"
)

# LLM-AHP human prompt blocks (scenario / phase); assembled by tools.llm_ahp_tool

LLM_AHP_SCENARIO_PREAMBLE = """You are a senior expert in fine-grained roadside charging pile (RCP) planning.

This task adopts a multi-scenario analysis to dynamically adjust evaluation dimension weights for different policy orientations of urban RCPs.

Based on Multi-scenario Analysis for Dynamic Weight Decision-making

"""

LLM_AHP_PHASE_PREAMBLE = """You are a senior expert in fine-grained roadside charging pile (RCP) planning.

This task adopts a multi-scenario analysis and phased planning to dynamically adjust evaluation dimension weights for different policy orientations (and development stages) of urban RCPs.

Based on Multi-scenario Analysis and Phased Planning for Dynamic Weight Decision-making

"""

LLM_AHP_SAATY_BLOCK = """### Saaty's 1-9 scale
- **1**: Two dimensions equally important
- **3**: Former slightly more important than latter
- **5**: Former obviously more important than latter
- **7**: Former strongly more important than latter
- **9**: Former extremely more important than latter
- **2, 4, 6, 8**: Intermediate values between adjacent judgments
- **Reciprocal (e.g., 1/3, 1/5)**: Latter more important than former
"""

LLM_AHP_PRINCIPLES_SCENARIO = """### Weight allocation principles

#### Principle 1: Scenario-oriented
- **Efficiency-oriented**: Economic dimension > Demand dimension > Other dimensions
- **Equity-oriented**: Service coverage dimension > Demand dimension > Other dimensions
- **Balanced**: Five dimensions relatively balanced; avoid extreme weights

#### Principle 3: Dimension trade-off logic
- Technical dimension is a **necessary condition**: veto if not met; marginal utility decreases after satisfaction.
- Policy dimension is a **compliance baseline** in compliance-led periods and a **bonus** in expansion and refinement periods.
- Economic dimension and demand are often positively correlated, but need moderate decoupling in equity-oriented cases.
- Service coverage and demand may be negatively correlated (low-demand areas need more coverage emphasis).

#### Principle 4: Consistency requirements
- Ensure transitivity: if A > B and B > C, then A > C.
- Avoid circular judgment (e.g., A > B > C > A).
- Judgments should be reasonable; avoid overuse of extreme scales (9 or 1/9).
- Consistency ratio **CR** implied by the matrix should remain **below 0.1** when checked by the solver.
"""

LLM_AHP_PRINCIPLES_PHASE = """### Weight allocation principles

#### Principle 1: Scenario-oriented
- **Efficiency-oriented**: Economic dimension > Demand dimension > Other dimensions
- **Equity-oriented**: Service coverage dimension > Demand dimension > Other dimensions
- **Balanced**: Five dimensions relatively balanced; avoid extreme weights

#### Principle 2: Phase-adaptive
- **Compliance period**: Policy and technical dimensions more important; ensure compliance and feasibility.
- **Expansion period**: Demand and economic dimensions more important; respond to market and scale benefits.
- **Refinement period**: Service coverage and system optimization more important; fill gaps and sustainability.

#### Principle 3: Dimension trade-off logic
- Technical dimension is a **necessary condition**: veto if not met; marginal utility decreases after satisfaction.
- Policy dimension is a **compliance baseline** in compliance-led periods and a **bonus** in expansion and refinement periods.
- Economic dimension and demand are often positively correlated, but need moderate decoupling in equity-oriented cases.
- Service coverage and demand may be negatively correlated (low-demand areas need more coverage emphasis).

#### Principle 4: Consistency requirements
- Ensure transitivity: if A > B and B > C, then A > C.
- Avoid circular judgment (e.g., A > B > C > A).
- Judgments should be reasonable; avoid overuse of extreme scales (9 or 1/9).
- Consistency ratio **CR** implied by the matrix should remain **below 0.1** when checked by the solver.
"""


def llm_ahp_special_considerations_scenario(scenario: str) -> str:
    return f"""### Special considerations
For the current **{scenario}** scenario, keep pairwise judgments aligned with Principle 1 (scenario-oriented), Principles 3–4, and the narrative in Section II; use Section I only as qualitative contrast among scenarios."""


def llm_ahp_special_considerations_phase(scenario: str, phase_title: str) -> str:
    return f"""### Special considerations
For **{scenario}** and **{phase_title}**, pay special attention to: (1) scenario-oriented emphasis from Principle 1; (2) phase-adaptive emphasis from Principle 2 together with Sections III and V; (3) dimension trade-offs and consistency in Principles 3–4."""


def llm_ahp_reasoning_requirements_scenario(scenario: str) -> str:
    return f"""**Reasoning requirements** (for the JSON `reasoning` field):
- Clearly explain how you interpret current scenario constraints and objectives for **{scenario}**.
- Elaborate key pairwise comparison judgments across the five criteria and their rationales.
- Estimate the implied weight distribution and explain whether it conforms to the guiding principles above."""


def llm_ahp_reasoning_requirements_phase(scenario: str, phase_title: str) -> str:
    return f"""**Reasoning requirements** (for the JSON `reasoning` field):
- Clearly explain how you interpret current scenario constraints for **{scenario}**.
- Elaborate how you reflect current phase characteristics for **{phase_title}** and the horizon in Section IV.
- Explain key pairwise comparison judgments and their rationales; state whether the matrix conforms to Principles 1–4."""


def llm_ahp_output_format_block_scenario(dims: str) -> str:
    return f"""## Output format (JSON only, no markdown fences)

Return **exactly** two top-level keys: `pairwise_matrix` and `reasoning`. Row and column order for the matrix must be: {dims}.

Use **only JSON numbers** in `pairwise_matrix` (no variables or expressions). The matrix must be 5*5, strictly positive, reciprocal (a_ij = 1/a_ji), diagonal 1. Example shape (replace every cell with your Saaty-scale judgments, keeping reciprocity):

{{
  "pairwise_matrix": [
    [1, 2, 0.5, 1, 1],
    [0.5, 1, 2, 1, 1],
    [2, 0.5, 1, 2, 0.5],
    [1, 1, 0.5, 1, 2],
    [1, 1, 2, 0.5, 1]
  ],
  "reasoning": "Detailed explanation per the reasoning requirements above."
}}"""


def llm_ahp_output_format_block_phase(dims: str) -> str:
    return f"""## Output format (JSON only, no markdown fences)

Return **exactly** two top-level keys: `pairwise_matrix` and `reasoning`. Row and column order for the matrix must be: {dims}.

Use **only JSON numbers** in `pairwise_matrix` (no variables or expressions). The matrix must be 5*5, strictly positive, reciprocal (a_ij = 1/a_ji), diagonal 1. Example shape (replace every cell with your Saaty-scale judgments, keeping reciprocity):

{{
  "pairwise_matrix": [
    [1, 2, 0.5, 1, 1],
    [0.5, 1, 2, 1, 1],
    [2, 0.5, 1, 2, 0.5],
    [1, 1, 0.5, 1, 2],
    [1, 1, 2, 0.5, 1]
  ],
  "reasoning": "Detailed explanation per the reasoning requirements above (scenario + phase + horizon alignment)."
}}"""


MCDA_INDICATOR_DESCRIPTION_PROMPT = """
## RCP deployment priority evaluation dimension system (reference)

Use this when interpreting the five criteria in pairwise judgments. Criterion order for the matrix remains:
`technical`, `economic`, `social`, `traffic`, `policy`.

### 1. Technical (`technical`) — Grid capacity
**Assessment Content**: Grid connection feasibility, distance to nearest substation/distribution room, grid capacity and stability.

**Weight Considerations**:
- Technical infeasibility is a veto, but marginal utility decreases after meeting basic conditions.
- Higher weight in Compliance Period (2025-2026), ensure basic feasibility.
- Moderately reduced weight in Expansion and Refinement periods, focus shifts to demand and service.

### 2. Economic (`economic`) — Cost-benefit
**Assessment Content**: Lifecycle cost-benefit, including construction costs, operating costs, expected revenue, payback period.

**Weight Considerations**:
- Highest weight in efficiency-oriented scenario (30-35%), pursue maximum economic output.
- Weight increase in Scale-up Expansion Period (2027-2028), pursue economies of scale.
- Weight reduction in equity-oriented scenario (15-20%), avoid excessive profit-seeking.

### 3. Social (`social`) — Charging demand
**Assessment Content**: Predict charging demand intensity based on commuting OD flow and functional zone type.
- Commuting flow (Flow_total), EV penetration rate (54.07%).
- RCP usage probability: commercial area 0.55, residential area 0.25, mixed area 0.40.
- Demand formula: D_rcp = Flow_total × P_ev × P_daily_charge × P_use_rcp × E_avg.

**Weight Considerations**:
- Higher weight in efficiency-oriented and expansion period (25-30%), respond to market demand; moderately reduced in equity-oriented (20-25%), avoid neglecting low-demand areas.

### 4. Traffic (`traffic`) — Road operations / smoothness
**Assessment Content**: Traffic smoothness and road-operation requirements for RCP siting (legacy standard 4.2.1; downstream implementation may vary).

**Weight Considerations**: Judge relative to other criteria using the same phased and scenario logic as in the legacy framework (road capacity, safety, and flow impacts of curbside charging).

### 5. Policy (`policy`) — Compliance
**Assessment Content**: Compliance with policies, regulations and construction standards.
- 4.2.5 Noise-sensitive area constraints (schools, hospitals, residential areas).
- Other policy compliance requirements (excluding traffic smoothness, which is reflected under `traffic`).

**Weight Considerations**:
- Highest weight in Compliance Period (2025-2026, 15-20%), ensure policy compliance.
- Reduced weight in Expansion and Refinement periods (10-15%), compliance becomes baseline.

---

## AHP judgment matrix construction guidance (Saaty 1-9 scale)
- **1**: Two dimensions equally important.
- **3**: Former slightly more important than latter.
- **5**: Former obviously more important than latter.
- **7**: Former strongly more important than latter.
- **9**: Former extremely more important than latter.
- **2, 4, 6, 8**: Intermediate values between adjacent judgments.
- **Reciprocal (e.g., 1/3, 1/5)**: Latter more important than former.
"""


# Multi-scenario Evaluation Agent: Scenario narratives

SCENARIO_NARRATIVES: Dict[str, str] = {
    "efficiency_oriented": (
        "**Efficiency-oriented** (`efficiency_oriented`)\n"
        "- **Primary goal**: Maximize economic output and resource utilization efficiency.\n"
        "- **Corresponding SDG**: SDG 9 (Industry, Innovation, and Infrastructure).\n"
        "- **Strategic focus**: Rapid implementation, scaled expansion, pursuing ROI.\n"
        "- **Typical phased pattern**: Small-scale demonstration in the early stage, large-scale expansion in "
        "the mid-stage, operational optimization in the late stage."
    ),
    "equity_oriented": (
        "**Equity-oriented** (`equity_oriented`)\n"
        "- **Primary goal**: Ensure service accessibility for vulnerable and underserved areas.\n"
        "- **Corresponding SDG**: SDG 11 (Sustainable Cities and Communities).\n"
        "- **Strategic focus**: Inclusive coverage, filling service gaps, equity priority.\n"
        "- **Typical phased pattern**: Basic coverage in the early stage, supplementary improvement in the "
        "mid-stage, refined filling of service gaps in the late stage."
    ),
    "balance_oriented": (
        "**Balanced** (`balance_oriented`)\n"
        "- **Primary goal**: Urban RCP system resilience and long-term stability.\n"
        "- **Corresponding SDG**: SDG 7 (Affordable and Clean Energy), SDG 9, and SDG 11—comprehensive "
        "coordination.\n"
        "- **Strategic focus**: Balanced development, system optimization, long-term sustainability.\n"
        "- **Typical phased pattern**: Balanced allocation across phases, gradual progress, ensuring system "
        "stability."
    ),
}


# Phased Decision-Making Agent: phase-level prompt bodies

_PLANNING_PHASE_INITIATION = "Planning Initiation Phase"
_PLANNING_PHASE_SCALE_UP = "Scale-up Expansion Phase"
_PLANNING_PHASE_REFINEMENT = "Refinement and Improvement Phase"

_PLANNING_PHASE_LABELS = {
    _PLANNING_PHASE_INITIATION: f"{_PLANNING_PHASE_INITIATION} (January 2025-December 2026)",
    _PLANNING_PHASE_SCALE_UP: f"{_PLANNING_PHASE_SCALE_UP} (January 2027-December 2028)",
    _PLANNING_PHASE_REFINEMENT: f"{_PLANNING_PHASE_REFINEMENT} (January 2029-December 2030)",
}

PHASE_PROMPT: Dict[str, str] = {
    _PLANNING_PHASE_INITIATION: f"""### {_PLANNING_PHASE_LABELS[_PLANNING_PHASE_INITIATION]}

**Phase positioning**: Planning initiation—deep understanding of relevant policy requirements, establishing technical standards, and completing compliance verification.

**Core characteristics**: Emphasis on verification rather than scale; compliance and feasibility first.

**Main objectives**:
- Comprehensively interpret site-selection planning policies and RCP construction standards so planning aligns with policy orientation.
- Establish technical standards and a policy framework for RCP deployment, laying the foundation for subsequent expansion.
- Build demonstration sites to verify feasibility and real-world effectiveness of planning outputs.
- Select sites with high feasibility and strong policy compliance for pilots.
- Accumulate experience and operational data for subsequent scaled expansion.

**Weight considerations** (deployment-emphasis guidance from the phased framework—use to inform how criterion importance should shift in this phase):
- Efficiency-oriented: focus on demonstration and economic-feasibility verification; relatively **smaller** early share (approximately 15-25%).
- Equity-oriented: focus on basic coverage and preliminary service-network establishment; **larger** early share (approximately 30-40%).
- Balanced: steady start with balanced allocation (approximately 30-35%).
""",
    _PLANNING_PHASE_SCALE_UP: f"""### {_PLANNING_PHASE_LABELS[_PLANNING_PHASE_SCALE_UP]}

**Phase positioning**: Transition to scaled construction that dynamically responds to demand growth.

**Core characteristics**: Demand-driven expansion, scale-up, and improvement of economic performance.

**Main objectives**:
- Expand site layout driven by roadside charging demand and market growth.
- Achieve rapid RCP network expansion and a competitive charging service network.
- Optimize resource allocation and pursue economies of scale.
- Further improve economic benefits and operational efficiency.
- Expand market share and service coverage.

**Weight considerations**:
- Efficiency-oriented: large-scale expansion; pursue market share and scale benefits (approximately 45-55%).
- Equity-oriented: continue supplementing service coverage; maintain steady expansion (approximately 30-40%).
- Balanced: balance expansion speed with system stability (approximately 30-40%).
""",
    _PLANNING_PHASE_REFINEMENT: f"""### {_PLANNING_PHASE_LABELS[_PLANNING_PHASE_REFINEMENT]}

**Phase positioning**: System optimization and refinement—filling service gaps and ensuring long-term sustainability.

**Core characteristics**: Close service blind spots, refined optimization, sustainable development.

**Main objectives**:
- Supplement RCP network service gaps guided by coverage analysis.
- Identify and fill blind spots to move toward comprehensive network coverage.
- Optimize operations and user experience; improve service quality.
- Ensure long-term sustainability of the RCP network.
- Support sustainable operation and future expansion.
- Advance comprehensive SDG-aligned outcomes.

**Weight considerations**:
- Efficiency-oriented: optimize operations and profitability; moderate share (approximately 25-35%).
- Equity-oriented: emphasize filling service gaps and equitable coverage (approximately 25-35%).
- Balanced: refined improvement while maintaining system integrity (approximately 30-35%).
""",
}

PHASE_OVERVIEW = f"""## Phased RCP planning (2025-2030, SDG-aligned)

RCP deployment is organized into three consecutive phases:

1. **{_PLANNING_PHASE_LABELS[_PLANNING_PHASE_INITIATION]}** — verification, standards, compliance, pilots.
2. **{_PLANNING_PHASE_LABELS[_PLANNING_PHASE_SCALE_UP]}** — demand-driven scaled build-out and network growth.
3. **{_PLANNING_PHASE_LABELS[_PLANNING_PHASE_REFINEMENT]}** — gap filling, refinement, and long-run sustainability.

Multi-scenario evaluation produces **scenario-level MCDA weights** (five criteria) for each SDG-aligned scenario. For a **given** scenario and **given** phase, you **refine** those weights so the criterion priorities match this phase's positioning, objectives, and weight-consideration logic while staying consistent with that scenario's primary goal and SDG emphasis.
"""


# Scenario narrative helper functions
def scenario_narrative(scenario_slug: str) -> str:
    return SCENARIO_NARRATIVES.get(scenario_slug, SCENARIO_NARRATIVES["balance_oriented"])


def phased_allocation_human_prompt(scenario_slug: str, total_rps_count: int) -> str:
    """
    Human message for LLM phased RPS count allocation across phases.
    """
    scenario = scenario_narrative(scenario_slug)
    phases_body = "\n\n".join(
        PHASE_PROMPT[k]
        for k in (
            _PLANNING_PHASE_INITIATION,
            _PLANNING_PHASE_SCALE_UP,
            _PLANNING_PHASE_REFINEMENT,
        )
    )
    return f"""You are a senior expert in refined roadside charging point RCP planning.

# RCP phased deployment allocation task

Based on multi-scenario analysis and the phased planning framework below, develop a phased allocation strategy: assign shares of the total feasible RPS count to phase1, phase2, and phase3.

## I. Planning background

**Total feasible RPS count**: {total_rps_count}

### Current scenario
{scenario}

## II. Phased planning framework

{phases_body}

## III. Decision requirements

Develop RPS allocation **ratios** for the three phases for the **current scenario** described in Section I.

**Requirements**:
1. The sum of the three phase ratios must equal 1.0.
2. Each phase ratio should stay within a reasonable range: not less than 15%, not more than 55%.
3. Allocation ratios should align with the strategic characteristics of the current scenario and with the phase positioning and weight considerations above.
4. Provide clear decision-making reasoning.

## IV. Output format

Return the decision in JSON:

```json
{{
    "scenario": "{scenario_slug}",
    "total_rps_count": {total_rps_count},
    "allocation": {{
        "phase1": 0.XX,
        "phase2": 0.XX,
        "phase3": 0.XX
    }},
    "allocation_counts": {{
        "phase1": XXX,
        "phase2": XXX,
        "phase3": XXX
    }},
    "reasoning": "Explain how you set the three phase ratios for this scenario and how the split supports scenario objectives and SDG alignment."
}}
```

Ensure the JSON is valid and parseable. Counts may be left as placeholders; the caller may recompute integer counts from ratios and total.
"""


def all_scenarios_reference_block() -> str:
    """
    Concatenate all SDG-aligned scenario narratives for LLM prompts.
    """
    parts: List[str] = []
    for slug in MULTI_SCENARIO:
        text = SCENARIO_NARRATIVES.get(slug, "")
        parts.append(f"### SDG-aligned scenario `{slug}`\n{text}")
    return "\n\n".join(parts)
