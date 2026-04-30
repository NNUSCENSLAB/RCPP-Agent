# -*- coding: utf-8 -*-
"""
Base class and category enumeration for built-in tools.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Optional


class ToolCategory(str, Enum):
    DATA_FLOW = "data_flow"
    MEMORY_CONSTRUCTION = "memory_construction"
    MCDA_INDICATORS = "mcda_indicators"
    DECISION_WEIGHTS = "decision_weights"
    REVIEW = "review"


@dataclass
class ToolContext:
    """
    Shared context for tool chains.
    """

    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


class BaseTool(ABC):
    """
    Synchronized Tool: name / category / description + run.
    """

    name: ClassVar[str] = "base_tool"
    category: ClassVar[ToolCategory] = ToolCategory.DATA_FLOW
    description: ClassVar[str] = ""

    @abstractmethod
    def run(self, ctx: Optional[ToolContext] = None, **kwargs: Any) -> Any:
        pass

    def __call__(self, ctx: Optional[ToolContext] = None, **kwargs: Any) -> Any:
        return self.run(ctx=ctx, **kwargs)
