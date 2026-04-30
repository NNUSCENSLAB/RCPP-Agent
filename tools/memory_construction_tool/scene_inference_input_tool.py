"""
Generate scene memory inference input JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import sqlite3

from tools.base import BaseTool, ToolCategory, ToolContext
from tools.registry import register_tool


def generate_inference_input_json(image_dir: Path, output_json_path: Path, image_root_prefix: str) -> int:
    samples: list[dict] = []
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [
        f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
    ]
    image_files.sort(key=lambda x: x.name)
    print(f"Found {len(image_files)} image files")

    inference_prompt = (
        "<image>\nPlease assess the physical environment of the red-boxed roadside parking "
        "stall's lateral zone according to the system prompt."
    )

    for img_file in image_files:
        server_image_path = f"{image_root_prefix.rstrip('/')}/{img_file.name}"
        samples.append(
            {
                "image": server_image_path,
                "conversations": [{"from": "human", "value": inference_prompt}],
            }
        )

    output_dir = Path(output_json_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated inference input JSON file: {output_json_path}")
    print(f"Contains {len(samples)} samples")
    if samples:
        print(f"Image path prefix example: {samples[0]['image']}")

    return len(samples)


def generate_scene_inference_from_spatialite(
    db_path: Path,
    output_json_path: Path,
    area: str,
    tables: Optional[Dict[str, str]] = None,
) -> int:

    _ = area
    tables = tables or {}
    bsv_layer = tables.get("bsv_image", "bsv_image")

    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(f'SELECT image_name, image_path FROM "{bsv_layer}"')
    rows = cur.fetchall()
    conn.close()
    if not rows:
        raise ValueError(f"Table {bsv_layer} has no data")

    inference_prompt = (
        "<image>\nPlease assess the physical environment of the red-boxed roadside parking "
        "stall's lateral zone according to the system prompt."
    )

    samples: list[dict] = []
    for _name, path in sorted(rows, key=lambda r: str(r[0] or "")):
        if path is None or (isinstance(path, float) and pd.isna(path)):
            continue
        p = str(path).strip()
        if not p:
            continue
        samples.append(
            {
                "image": p.replace("\\", "/"),
                "conversations": [{"from": "human", "value": inference_prompt}],
            }
        )

    if not samples:
        raise ValueError("bsv_image has no valid image_path, no scene memory inference samples generated")

    output_dir = Path(output_json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(
        f"Generated scene inference input from SpatiaLite: {output_json_path}, "
        f"{len(samples)} sample(s).",
        flush=True,
    )
    return len(samples)


@register_tool
class SceneInferenceInputTool(BaseTool):
    name = "scene_inference_input"
    category = ToolCategory.MEMORY_CONSTRUCTION

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        db_path: Optional[Path] = None,
        area: str = "gulou",
        output_json: Optional[Path] = None,
        tables: Optional[Dict[str, str]] = None,
        image_dir: Optional[Path] = None,
        image_root_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        if output_json is None:
            raise ValueError("output_json is required")
        if db_path:
            n = generate_scene_inference_from_spatialite(db_path, output_json, area, tables)
        elif image_dir and image_root_prefix:
            n = generate_inference_input_json(image_dir, output_json, image_root_prefix)
        else:
            raise ValueError("Please specify db_path, or both image_dir and image_root_prefix")
        if ctx is not None:
            ctx.set("scene_inference_input_path", str(Path(output_json).resolve()))
        return n
