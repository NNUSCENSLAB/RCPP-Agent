# -*- coding: utf-8 -*-
"""Merge scene-memory and semantic memor into a multimodal scene-semantic memory."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rcpp_core.semantic_rules import extract_statistical_summary
from tools.base import BaseTool, ToolCategory, ToolContext
from tools.registry import register_tool

def parse_image_filename(image_path: str) -> Tuple[float, float, float, int, int]:
    """
    Parse lng, lat, angle, and suffix indices from the image filename (lng_lat_angle_s1_s2.jpg).
    """
    filename = Path(image_path).name
    name_without_ext = filename.rsplit(".", 1)[0]
    parts = name_without_ext.split("_")
    if len(parts) >= 5:
        lng = float(parts[0])
        lat = float(parts[1])
        angle = float(parts[2])
        suffix1 = int(parts[3])
        suffix2 = int(parts[4])
        return lng, lat, angle, suffix1, suffix2
    raise ValueError(f"Unrecognized image filename pattern: {filename}")

def parse_semantic_memory(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse semantic memory fields from chat messages.
    """
    user_content = str(messages[0]["content"])
    summary = extract_statistical_summary(user_content)
    coordinates = summary.get("coordinates") or {}
    assistant_content = messages[1]["content"]
    semantic_data = json.loads(str(assistant_content))
    rps_id = str(semantic_data.get("rps_id") or summary.get("rps_id") or "")
    if not rps_id:
        raise ValueError("RPS ID not found in semantic record")
    lng = float(coordinates.get("lng"))
    lat = float(coordinates.get("lat"))
    angle = float(coordinates.get("angle", summary.get("angle", 0.0)))
    bsv_image = semantic_data.get("bsv_image") or summary.get("bsv_image", "")

    memory_fields = {
        "functional_zone_type": semantic_data.get("functional_zone_type", ""),
        "commuting_flow": semantic_data.get("commuting_flow", ""),
        "sensitive_constraints": semantic_data.get("sensitive_constraints", ""),
        "grid_accessibility": semantic_data.get("grid_accessibility", ""),
    }
    for optional in ("rule_facts", "evidence_fields", "_meta"):
        if optional in semantic_data:
            memory_fields[optional] = semantic_data[optional]
    
    return {
        'rps_id': rps_id,
        'lng': lng,
        'lat': lat,
        'angle': angle,
        'bsv_image': bsv_image,
        'semantic_memory': memory_fields,
    }

def parse_scene_memory_from_conversations(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse scene-memory JSON from the model reply in conversations.
    """
    conversations = item["conversations"]
    if len(conversations) < 2:
        raise ValueError(" Incorrect conversations format")
    
    gpt_value = conversations[1]['value']
    scene_memory = json.loads(gpt_value)
    
    return scene_memory

def extract_scene_memory_node(scene_memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the scene-memory node: fields placed under memory_node.scene_memory in the merged memory.
    """
    fields = (
        "has_existing_RCP",
        "is_functional_zone",
        "functional_zone_type",
        "has_ground_obstacle",
        "ground_obstacle_types",
        "ground_obstacle_count",
        "visual_distractors_noted",
        "clearance_visual_assessment",
        "scene_reasoning",
        "confidence_score",
        "_meta",
    )
    return {name: scene_memory[name] for name in fields if name in scene_memory}

def _pic_key_from_row(row: Any) -> Optional[str]:
    """Return stripped Pic string for use as map key, or None if missing."""
    pic_raw = row["Pic"]
    if pic_raw is None:
        return None
    try:
        if pic_raw != pic_raw:  # NaN without importing pandas at module import time.
            return None
    except (ValueError, TypeError):
        pass
    pic_str = str(pic_raw).strip()
    return pic_str or None


def load_shp_mapping(shp_path: str) -> Dict[str, Dict[str, Any]]:
    """
    The 'Pic' field serves as an anchor to merge scene memory and semantic memory.
    """
    import geopandas as gpd

    mapping: Dict[str, Dict[str, Any]] = {}
    shp_path_obj = Path(shp_path)

    if shp_path_obj.is_dir():
        shp_files = list(shp_path_obj.glob("*.shp"))
        print(f"Found {len(shp_files)} shapefile(s) under {shp_path}")
        for shp_file in shp_files:
            print(f"  Loading: {shp_file.name}")
            gdf = gpd.read_file(shp_file)
            print(f"    Columns: {list(gdf.columns)}")

            for idx, row in gdf.iterrows():
                try:
                    pic_key = _pic_key_from_row(row)
                    if pic_key is None:
                        continue

                    info: Dict[str, Any] = {}
                    if "name" in gdf.columns:
                        info["street_name"] = row["name"]
                    if "lng" in gdf.columns:
                        info["lng"] = float(row["lng"])
                    if "lat" in gdf.columns:
                        info["lat"] = float(row["lat"])
                    if "angle" in gdf.columns:
                        info["angle"] = float(row["angle"])

                    mapping[pic_key] = info
                except Exception as e:
                    print(f"    Warning: row {idx}: {e}")
                    continue
    else:
        gdf = gpd.read_file(shp_path)
        print(f"Columns: {list(gdf.columns)}")

        for idx, row in gdf.iterrows():
            try:
                pic_key = _pic_key_from_row(row)
                if pic_key is None:
                    continue

                info = {}
                if "name" in gdf.columns:
                    info["street_name"] = row["name"]
                if "lng" in gdf.columns:
                    info["lng"] = float(row["lng"])
                if "lat" in gdf.columns:
                    info["lat"] = float(row["lat"])
                if "angle" in gdf.columns:
                    info["angle"] = float(row["angle"])

                mapping[pic_key] = info
            except Exception as e:
                print(f"Warning: row {idx}: {e}")
                continue

    return mapping


def run_merge(
    scene_memory_path: str,
    semantic_memory_path: str,
    shp_path: str,
    output_path: str,
    *,
    mismatch_report_path: Optional[str] = None,
    area_slug: Optional[str] = None,
) -> None:
    print("Loading inputs...")

    with open(scene_memory_path, "r", encoding="utf-8") as f:
        scene_results = json.load(f)
        print(f"Scene memory records: {len(scene_results)}")

    with open(semantic_memory_path, "r", encoding="utf-8") as f:
        semantic_data = json.load(f)

    print(f"Semantic memory records: {len(semantic_data)}")

    shp_mapping = load_shp_mapping(shp_path)
    print(f"Shapefile Pic keys: {len(shp_mapping)}")

    scene_dict: Dict[str, Any] = {}
    for result in scene_results:
        image_path = str(result.get("image", ""))
        try:
            image_filename = Path(image_path).name

            scene_memory = parse_scene_memory_from_conversations(result)

            lng, lat, angle, suffix1, suffix2 = parse_image_filename(image_path)

            scene_dict[image_filename] = {
                "lng": lng,
                "lat": lat,
                "angle": angle,
                "suffix1": suffix1,
                "suffix2": suffix2,
                "image_path": image_path,
                "scene_memory": scene_memory,
            }
        except Exception as e:
            print(f"Failed to parse scene memory ({image_path or 'unknown'}): {e}")
            continue

    semantic_dict: Dict[str, Any] = {}
    for item in semantic_data:
        try:
            semantic_info = parse_semantic_memory(item['messages'])
            bsv_image = semantic_info.get('bsv_image', '')
            if bsv_image:
                semantic_dict[bsv_image] = semantic_info
        except Exception as e:
            print(f"Failed to parse semantic memory: {e}")
            continue

    output_data: List[Dict[str, Any]] = []
    mismatch_report: List[Dict[str, Any]] = []
    rps_counter = 1
    rps_prefix = (area_slug or "rcp").strip().lower().replace(" ", "_")
    rps_prefix = "".join(c if c.isalnum() or c == "_" else "_" for c in rps_prefix).upper()[:24] or "RCP"

    matched_semantic_images: set[str] = set()

    for image_filename, scene_item in scene_dict.items():
        if image_filename in semantic_dict:
            semantic_item = semantic_dict[image_filename]
            matched_semantic_images.add(image_filename)

            scene_lng = scene_item["lng"]
            scene_lat = scene_item["lat"]

            shp_info = shp_mapping.get(image_filename, {})
            street_name = shp_info.get("street_name", "")
            if "lng" in shp_info:
                scene_lng = shp_info["lng"]
            if "lat" in shp_info:
                scene_lat = shp_info["lat"]

            rps_id = f"{rps_prefix}-{rps_counter:04d}"
            rps_counter += 1

            output_record = {
                'RPS_id': rps_id,
                'coordinates': [scene_lng, scene_lat],
                'street_name': street_name,
                'memory_node': {
                    "scene_memory": extract_scene_memory_node(scene_item["scene_memory"]),
                    'semantic_memory': semantic_item['semantic_memory']
                },
                'street_view_images': [scene_item['image_path']]
            }
            
            output_data.append(output_record)
        else:
            scene_lng = scene_item["lng"]
            scene_lat = scene_item["lat"]
            scene_angle = scene_item["angle"]
            mismatch_report.append(
                {
                    "image_filename": image_filename,
                    "coordinates": f"({scene_lng}, {scene_lat}, {scene_angle})",
                    "note": "Scene memory has no matching semantic memory",
                }
            )

    for bsv_image, semantic_item in semantic_dict.items():
        if bsv_image not in matched_semantic_images:
            sem_lng = semantic_item['lng']
            sem_lat = semantic_item['lat']
            sem_angle = semantic_item['angle']
            mismatch_report.append(
                {
                    "image_filename": bsv_image,
                    "coordinates": f"({sem_lng}, {sem_lat}, {sem_angle})",
                    "note": "Semantic memory has no matching scene memory",
                }
            )

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nWrote merged memory -> {output_path}")
    print(f"Merged records: {len(output_data)}")

    if mismatch_report_path is None:
        mismatch_report_path = str(Path(output_path).parent / "mismatch_report.txt")
    Path(mismatch_report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(mismatch_report_path, "w", encoding="utf-8") as f:
        f.write("Mismatch report\n")
        if area_slug:
            f.write(f"Area slug: {area_slug}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total scene records: {len(scene_results)}\n")
        f.write(f"Total semantic records: {len(semantic_data)}\n")
        f.write(f"Matched pairs: {len(output_data)}\n")
        f.write(f"Mismatches: {len(mismatch_report)}\n")
        f.write("=" * 60 + "\n\n")

        if mismatch_report:
            f.write("Details:\n\n")
            for i, report in enumerate(mismatch_report, 1):
                f.write(f"{i}. image_filename: {report.get('image_filename', 'unknown')}\n")
                f.write(f"   coordinates: {report['coordinates']}\n")
                if "note" in report:
                    f.write(f"   note: {report['note']}\n")
                f.write("\n")
        else:
            f.write("All records matched.\n")

    if mismatch_report:
        print(f"\nMismatches: {len(mismatch_report)}")
        print(f"Report saved -> {mismatch_report_path}")
        print("First 10 mismatches:")
        for report in mismatch_report[:10]:
            print(
                f"  image: {report.get('image_filename', 'unknown')}, "
                f"coords: {report['coordinates']}, {report.get('note', '')}"
            )
        if len(mismatch_report) > 10:
            print(f"  ... {len(mismatch_report) - 10} more (see report file)")
    else:
        print("\nNo mismatches.")


@register_tool
class MergeSceneSemanticMemoryTool(BaseTool):
    name = "merge_scene_semantic_memory"
    category = ToolCategory.MEMORY_CONSTRUCTION

    def run(
        self,
        ctx: Optional[ToolContext] = None,
        *,
        scene_file: str,
        semantic_file: str,
        output_file: str,
        shp_path: str,
        mismatch_report_path: Optional[str] = None,
        area_slug: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        run_merge(
            scene_file,
            semantic_file,
            shp_path,
            output_file,
            mismatch_report_path=mismatch_report_path,
            area_slug=area_slug,
        )
        if ctx is not None:
            ctx.set("merged_memory_path", str(Path(output_file).resolve()))
