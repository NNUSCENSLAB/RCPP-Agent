# -*- coding: utf-8 -*-
"""
LangGraph BaseStore backed by Neo4j for global scene_semantic_memory per study area.

Namespace convention: ("scene_semantic_memory", <area_slug>), key "global", value is the full RPS map dict.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from rcpp_core.build_scene_semantic_memory import (
    list_all_global_scene_memory_area_slugs,
    load_global_scene_semantic_memory,
    save_global_scene_semantic_memory,
)

logger = logging.getLogger(__name__)

SCENE_SEMANTIC_NS_ROOT = "scene_semantic_memory"
SCENE_SEMANTIC_KEY_GLOBAL = "global"


def area_slug_from_spatialite_context(spatialite_context: Optional[Dict[str, Any]]) -> str:
    s = str((spatialite_context or {}).get("area") or "").strip().lower()
    if not s:
        raise ValueError(
            "spatialite_context['area'] is required for global scene_semantic_memory in Neo4j."
        )
    return s


def scene_semantic_namespace(area_slug: str) -> tuple[str, ...]:
    return (SCENE_SEMANTIC_NS_ROOT, str(area_slug).strip().lower())


def _is_scene_global_namespace(namespace: tuple[str, ...], key: str) -> bool:
    return (
        len(namespace) == 2
        and namespace[0] == SCENE_SEMANTIC_NS_ROOT
        and bool(namespace[1])
        and key == SCENE_SEMANTIC_KEY_GLOBAL
    )


def _area_from_namespace(namespace: tuple[str, ...]) -> str:
    if len(namespace) != 2 or namespace[0] != SCENE_SEMANTIC_NS_ROOT:
        raise ValueError(
            f"Invalid namespace for Neo4j scene store: {namespace}. "
            f"Expected ('{SCENE_SEMANTIC_NS_ROOT}', <area_slug>)."
        )
    return str(namespace[1])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _item_for_area(area_slug: str) -> Item | None:
    data = load_global_scene_semantic_memory(area_slug)
    if not data:
        return None
    now = _utc_now()
    return Item(
        value=dict(data),
        key=SCENE_SEMANTIC_KEY_GLOBAL,
        namespace=scene_semantic_namespace(area_slug),
        created_at=now,
        updated_at=now,
    )


def _does_match(condition: MatchCondition, key: tuple[str, ...]) -> bool:
    match_type = condition.match_type
    path = condition.path
    if len(key) < len(path):
        return False
    if match_type == "prefix":
        for k_elem, p_elem in zip(key, path, strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
    if match_type == "suffix":
        for k_elem, p_elem in zip(reversed(key), reversed(path), strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
    raise ValueError(f"Unsupported match type: {match_type}")


def _simple_value_filter(value: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    if not flt:
        return True
    for fk, fv in flt.items():
        if isinstance(fv, dict) and any(str(k).startswith("$") for k in fv):
            logger.warning("Neo4j scene store search ignores operator filters for key %s", fk)
            continue
        if value.get(fk) != fv:
            return False
    return True


class Neo4jGlobalSceneSemanticStore(BaseStore):
    """
    Long-term store for merged scene_semantic_memory JSON.
    """

    supports_ttl = False

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        out: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                out.append(self._get(op))
            elif isinstance(op, PutOp):
                out.append(self._put(op))
            elif isinstance(op, SearchOp):
                out.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                out.append(self._list_namespaces(op))
            else:
                raise ValueError(f"Neo4jGlobalSceneSemanticStore: unknown op {type(op)}")
        return out

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, ops)

    def _get(self, op: GetOp) -> Item | None:
        if not _is_scene_global_namespace(op.namespace, op.key):
            logger.warning("Get ignored for unsupported namespace or key: %s %s", op.namespace, op.key)
            return None
        area = _area_from_namespace(op.namespace)
        return _item_for_area(area)

    def _put(self, op: PutOp) -> None:
        if not _is_scene_global_namespace(op.namespace, op.key):
            raise ValueError(
                f"Put only supports namespace {SCENE_SEMANTIC_NS_ROOT!r} plus area and key "
                f"{SCENE_SEMANTIC_KEY_GLOBAL!r}, got {op.namespace!r} {op.key!r}"
            )
        area = _area_from_namespace(op.namespace)
        if op.value is None:
            save_global_scene_semantic_memory(area, {})
            return
        if not isinstance(op.value, dict):
            raise TypeError("scene_semantic_memory value must be a dict or None")
        save_global_scene_semantic_memory(area, op.value)
        return None

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = op.namespace_prefix
        if not prefix or prefix[0] != SCENE_SEMANTIC_NS_ROOT:
            return []

        if len(prefix) >= 2:
            area = str(prefix[1])
            item = _item_for_area(area)
            if item is None:
                return []
            if not _simple_value_filter(item.value, op.filter):
                return []
            si = SearchItem(
                item.namespace,
                item.key,
                item.value,
                item.created_at,
                item.updated_at,
                None,
            )
            one = [si]
            return one[op.offset : op.offset + op.limit]

        areas = list_all_global_scene_memory_area_slugs()
        items: list[SearchItem] = []
        for area in areas:
            item = _item_for_area(area)
            if item is None:
                continue
            if not _simple_value_filter(item.value, op.filter):
                continue
            items.append(
                SearchItem(
                    item.namespace,
                    item.key,
                    item.value,
                    item.created_at,
                    item.updated_at,
                    None,
                )
            )
        return items[op.offset : op.offset + op.limit]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        areas = list_all_global_scene_memory_area_slugs()
        namespaces = [scene_semantic_namespace(a) for a in areas]
        if op.match_conditions:
            namespaces = [
                ns
                for ns in namespaces
                if all(_does_match(c, ns) for c in op.match_conditions)
            ]
        namespaces = sorted(namespaces)
        if op.max_depth is not None:
            namespaces = sorted({ns[: op.max_depth] for ns in namespaces})
        return namespaces[op.offset : op.offset + op.limit]
