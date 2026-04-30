# -*- coding: utf-8 -*-
"""
Register and query built-in tools by category.
"""
from __future__ import annotations

from typing import Dict, List, Type
from tools.base import BaseTool, ToolCategory


_REGISTRY: Dict[ToolCategory, List[Type[BaseTool]]] = {
    ToolCategory.DATA_FLOW: [],
    ToolCategory.MEMORY_CONSTRUCTION: [],
    ToolCategory.MCDA_INDICATORS: [],
    ToolCategory.DECISION_WEIGHTS: [],
    ToolCategory.REVIEW: [],
}


def register_tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    cat = cls.category
    if cls not in _REGISTRY[cat]:
        _REGISTRY[cat].append(cls)
    return cls


def get_tool_classes(category: ToolCategory) -> List[Type[BaseTool]]:
    return list(_REGISTRY.get(category, []))


def get_all_registered() -> Dict[ToolCategory, List[Type[BaseTool]]]:
    return {k: list(v) for k, v in _REGISTRY.items()}
