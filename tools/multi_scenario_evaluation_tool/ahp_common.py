# -*- coding: utf-8 -*-
"""
Shared AHP math, JSON parsing, and OpenAI client for LLM-AHP tools.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from configs.config import (
    DEEPSEEK_API_BASE_URL,
    DEFAULT_LLM_AHP_TEMPERATURE,
    LLM_AHP_MAX_RETRIES,
    MCDA_DIMENSIONS,
    get_llm_ahp_model,
    get_provider_api_key,
    provider_for_model,
)
from rcpp_core import scenario_narrative, all_scenarios_reference_block

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> Dict[str, object]:
    """
    Parse LLM-AHP JSON.
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
    raise json.JSONDecodeError("No JSON object found in LLM response", raw, 0)


def ahp_weights_from_matrix(matrix: List[List[float]]) -> Tuple[List[float], float]:
    """
    Turn one five by five judgment matrix into priority weights and a consistency ratio for AHP checks.
    """
    n = len(matrix)
    if n == 0:
        return [], 0.0
    a = np.array(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(a)
    idx = int(np.argmax(np.real(eigenvalues)))
    max_eigenvalue = float(np.real(eigenvalues[idx]))
    vec = np.real(eigenvectors[:, idx])
    weights = (vec / vec.sum()).tolist()
    ci = (max_eigenvalue - n) / (n - 1) if n > 1 else 0.0
    ri_dict = {1: 0, 2: 0, 3: 0.52, 4: 0.89, 5: 1.12, 6: 1.26, 7: 1.36, 8: 1.41, 9: 1.46}
    ri = ri_dict.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return weights, cr


def equal_dimension_weights() -> Dict[str, float]:
    """
    Return the same small weight for each of the five MCDA keys when no custom split exists.
    """
    n = len(MCDA_DIMENSIONS)
    return {k: 1.0 / n for k in MCDA_DIMENSIONS}


def normalize_dimension_weights(w: Dict[str, float]) -> Dict[str, float]:
    """
    Scale five input weights so they sum to one across the fixed MCDA key order.
    """
    vals = {k: float(w.get(k, 0.0)) for k in MCDA_DIMENSIONS}
    s = sum(vals.values())
    if s <= 0:
        return equal_dimension_weights()
    return {k: vals[k] / s for k in MCDA_DIMENSIONS}


def build_chat_openai(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_LLM_AHP_TEMPERATURE,
    *,
    json_mode: bool = True,
) -> ChatOpenAI:
    """
    Build a chat client for the configured LLM-AHP model name and API key.

    When ``json_mode`` is True, request OpenAI-compatible ``response_format``
    ``{"type": "json_object"}`` (supported by DeepSeek and recent OpenAI models)
    so the assistant message body is a single JSON object.
    """
    name = model_name or get_llm_ahp_model()
    provider = provider_for_model(name)
    key = api_key or get_provider_api_key(provider)
    if not key:
        raise RuntimeError(f"API key is not set for {provider!r} LLM-AHP.")
    model_kwargs: Dict[str, object] = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
    common: Dict[str, object] = {
        "model": name,
        "api_key": SecretStr(key),
        "temperature": temperature,
    }
    if model_kwargs:
        common["model_kwargs"] = model_kwargs
    if provider == "deepseek":
        common["base_url"] = DEEPSEEK_API_BASE_URL
    return ChatOpenAI.model_validate(common)


__all__ = [
    "scenario_narrative",
    "all_scenarios_reference_block",
    "parse_json_response",
    "ahp_weights_from_matrix",
    "equal_dimension_weights",
    "normalize_dimension_weights",
    "build_chat_openai",
    "LLM_AHP_MAX_RETRIES",
]
