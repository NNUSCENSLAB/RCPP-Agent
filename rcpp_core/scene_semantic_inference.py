# -*- coding: utf-8 -*-
"""
Scene and semantic memory inference package.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from rcpp_core.memory_schema import (
    build_memory_metadata,
    quality_score,
    validate_scene_memory,
    validate_semantic_memory,
)
from rcpp_core.model_registry import ModelDeployment
from rcpp_core.prompts import SCENE_SYSTEM_PROMPT, SEMANTIC_SYSTEM_PROMPT
from rcpp_core.semantic_rules import (
    deterministic_semantic_memory,
    derive_semantic_facts,
    extract_statistical_summary,
    validate_semantic_against_rules,
)

logger = logging.getLogger(__name__)


def _start_runtime_measurement() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    return time.perf_counter()


def _finish_runtime_measurement(started_at: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"latency_ms": round((time.perf_counter() - started_at) * 1000, 3)}
    try:
        import torch

        if torch.cuda.is_available():
            result["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
    except Exception:
        pass
    return result


def _load_records(path: str) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Inference dataset must be a JSON array or JSONL: {path}")
    return value


def _first_user_content(item: Dict[str, Any]) -> str:
    messages = item.get("messages") or item.get("conversations") or []
    for message in messages:
        if (message.get("role") or message.get("from")) in {"user", "human"}:
            return str(message.get("content", message.get("value", "")))
    raise ValueError("Inference record has no user message")


def _scene_predict(messages: List[Dict], model, processor, tokenizer) -> str:
    """
    Run one vision-language decode and return the generated text string only.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = cast(
        Tuple[Any, Any],
        process_vision_info(messages),
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Locate the first balanced brace block in text and load it as JSON or return None.
    """
    start_idx = text.find("{")
    if start_idx == -1:
        return None
    brace_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    if brace_count != 0:
        return None
    json_str = text[start_idx: end_idx + 1]
    try:
        value = json.loads(json_str)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _repair_semantic_with_general_provider(
    facts: Dict[str, Any], fallback: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Optionally ask the configured text model to rewrite prose, never facts."""
    enabled = os.getenv(
        "RCPP_SEMANTIC_REPAIR_WITH_GENERAL",
        os.getenv("RCPP_SEMANTIC_REPAIR_WITH_DEEPSEEK", "false"),
    )
    if enabled.lower() not in {"1", "true", "yes"}:
        return None
    from configs.config import get_provider_api_key, provider_for_model

    model = os.getenv("RCPP_GENERAL_MODEL", "deepseek-chat")
    provider = provider_for_model(model)
    api_key = get_provider_api_key(provider)
    if not api_key:
        logger.warning("Semantic repair requested but the %s API key is not set", provider)
        return None
    try:
        import httpx

        response = httpx.post(
            os.getenv(
                "RCPP_GENERAL_BASE_URL",
                "https://api.deepseek.com/chat/completions"
                if provider == "deepseek"
                else "https://api.openai.com/v1/chat/completions",
            ),
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the four semantic-memory explanation strings. Return JSON only. "
                            "Copy rule_facts, rps_id, bsv_image and all numeric evidence exactly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"rule_facts": facts, "safe_template": fallback}, ensure_ascii=False
                        ),
                    },
                ],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        repaired = _extract_json_from_text(content)
        if repaired is None:
            return None
        repaired.setdefault("rps_id", fallback["rps_id"])
        repaired.setdefault("bsv_image", fallback.get("bsv_image"))
        return repaired
    except Exception as exc:
        logger.warning("General-provider semantic repair failed; using deterministic fallback: %s", exc)
        return None


def _parse_and_validate_scene_json(
    text: str, sample_id: int
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Try strict JSON on the full string, else extract the first object for logging by sample_id.
    """
    try:
        value = json.loads(text)
        return (value, True) if isinstance(value, dict) else (None, False)
    except json.JSONDecodeError:
        obj = _extract_json_from_text(text)
        if obj is not None:
            logger.warning("Sample %s: JSON extracted from mixed content", sample_id)
            return obj, True
        logger.error("Sample %s: Failed to parse JSON; raw head: %r", sample_id, text[:200])
        return None, False


def _semantic_generate_one(
    model, tokenizer, user_content: str, max_new_tokens: int, max_length: int
) -> str:
    """
    Generate one assistant reply from user_content using the semantic system prompt.
    """
    import torch

    messages = [
        {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _pretrained_location_kwargs(path: str) -> Dict[str, Any]:
    return {"local_files_only": True} if Path(path).expanduser().exists() else {}


def _load_scene_bundle(base_path: str, adapter_path: str, family: str = "auto"):
    """
    Load the fine-tuned MLLM along with image processor and tokenizer for multimodal scene understanding tasks.
    """
    import torch
    from peft.peft_model import PeftModel
    import transformers
    from transformers import AutoProcessor, AutoTokenizer, GenerationConfig

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logger.info("Loading scene VL base: %s", base_path)
    location_kwargs = _pretrained_location_kwargs(base_path)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, use_fast=False, trust_remote_code=True, **location_kwargs
    )
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True, **location_kwargs)
    if family == "qwen2_5_vl":
        model_class = transformers.Qwen2_5_VLForConditionalGeneration
    else:
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if model_class is None:
            raise RuntimeError("Qwen3-VL requires transformers>=4.57")
    base_model = model_class.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        **location_kwargs,
    )
    model = base_model
    if adapter_path:
        logger.info("Loading scene adapter: %s", adapter_path)
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16,
            **_pretrained_location_kwargs(adapter_path),
        )
    model.eval()
    gen_cfg = cast(GenerationConfig, model.generation_config)
    gen_cfg.do_sample = False
    gen_cfg.temperature = 0.2
    gen_cfg.top_p = 0.8
    gen_cfg.num_beams = 2
    return model, processor, tokenizer


def _load_semantic_bundle(base_path: str, adapter_path: str):
    """
    Load semantic base weights and LoRA adapter as one text model and tokenizer in reasoning model.
    """
    import torch
    from peft.peft_model import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logger.info("Loading semantic LM base: %s", base_path)
    location_kwargs = _pretrained_location_kwargs(base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, **location_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        **location_kwargs,
    )
    model = base_model
    if adapter_path:
        logger.info("Loading semantic adapter: %s", adapter_path)
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.bfloat16,
            **_pretrained_location_kwargs(adapter_path),
        )
    model.eval()
    return model, tokenizer


def run_scene_inference_file(
    input_file: str,
    output_file: str,
    base_path: str,
    adapter_path: str,
    bundle_cache: Dict[Tuple[str, str], Tuple[Any, Any, Any]],
    deployment: Optional[ModelDeployment] = None,
) -> None:
    """
    Loop a JSON array of image and conversation items through the MLLM for scene memory inference.
    Append parsed scene JSON or an error string per item, reuse bundle_cache for model load, write output_file.
    """
    key = (base_path, adapter_path)
    if key not in bundle_cache:
        bundle_cache[key] = _load_scene_bundle(
            base_path, adapter_path, deployment.family if deployment else "auto"
        )
    model, processor, tokenizer = bundle_cache[key]

    test_dataset = _load_records(input_file)

    inference_results: List[Dict] = []
    for i, item in enumerate(test_dataset):
        try:
            runtime_started = _start_runtime_measurement()
            images = item.get("images") or []
            image_path = item.get("image") or (images[0] if images else "")
            if not image_path:
                raise ValueError("Scene inference record has no image")
            input_conversations = item.get("conversations") or [
                {"from": "human", "value": _first_user_content(item)}
            ]
            user_prompt_raw = _first_user_content(item)
            user_text = user_prompt_raw.replace("<image>", "").strip()
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SCENE_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_text},
                    ],
                },
            ]
            response = ""
            pred_json: Optional[Dict[str, Any]] = None
            validation_errors: list[str] = []
            for attempt in range(2):
                response = _scene_predict(messages, model, processor, tokenizer)
                pred_json, is_valid = _parse_and_validate_scene_json(response, i + 1)
                validation_errors = (
                    validate_scene_memory(pred_json) if is_valid and pred_json is not None else ["invalid JSON"]
                )
                if not validation_errors:
                    break
                if attempt == 0:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": response}],
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": "Return corrected JSON only. Validation errors: "
                                + "; ".join(validation_errors),
                            }],
                        }
                    )
            # Do not copy a gold assistant label from ms-swift test JSONL.
            result_item = {
                "image": image_path,
                "conversations": [dict(input_conversations[0])],
                "metadata": dict(item.get("metadata") or {}),
            }
            if pred_json is not None and not validation_errors:
                model_info = deployment or ModelDeployment(
                    key="custom_scene", kind="scene", base_model=base_path, adapter_path=adapter_path
                )
                pred_json["_meta"] = build_memory_metadata(
                    base_model=model_info.base_model,
                    adapter_version=model_info.adapter_version,
                    quality=quality_score("scene", [], pred_json),
                    evidence_source=model_info.evidence_source,
                )
                pred_json["_meta"]["runtime"] = _finish_runtime_measurement(runtime_started)
                result_item["conversations"].append(
                    {"from": "gpt", "value": json.dumps(pred_json, ensure_ascii=False)}
                )
            else:
                err = json.dumps(
                    {
                        "error": "Scene output failed validation after retry",
                        "validation_errors": validation_errors,
                        "requires_human_review": True,
                        "raw_output": response[:500] if response else "",
                    },
                    ensure_ascii=False,
                )
                result_item["conversations"].append({"from": "gpt", "value": err})
            inference_results.append(result_item)
        except Exception as e:
            logger.exception("Scene sample %s failed: %s", i + 1, e)
            inference_results.append(
                {
                    "image": item.get("image", ""),
                    "metadata": dict(item.get("metadata") or {}),
                    "conversations": item.get("conversations", [])[:1]
                    + [{"from": "gpt", "value": json.dumps({"error": str(e)}, ensure_ascii=False)}],
                }
            )

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, indent=2, ensure_ascii=False)
    logger.info("Scene inference saved: %s (%s items)", output_file, len(inference_results))


def run_semantic_inference_file(
    input_file: str,
    output_file: str,
    base_path: str,
    adapter_path: str,
    bundle_cache: Dict[Tuple[str, str], Tuple[Any, Any]],
    max_new_tokens: int = 256,
    max_length: int = 1024,
    deployment: Optional[ModelDeployment] = None,
    evaluation_mode: bool = False,
) -> None:
    """
    Loop a JSON array of records with a first user message through the LLM for semantic memory inference.
    """
    key = (base_path, adapter_path)
    if key not in bundle_cache:
        bundle_cache[key] = _load_semantic_bundle(base_path, adapter_path)
    model, tokenizer = bundle_cache[key]

    data = _load_records(input_file)

    out_items: List[Dict] = []
    for idx, item in enumerate(data):
        runtime_started = _start_runtime_measurement()
        user_msg = _first_user_content(item)
        summary = extract_statistical_summary(user_msg)
        facts = derive_semantic_facts(summary)
        base_user_msg = user_msg.split("【Rule Facts", 1)[0].rstrip()
        grounded_user_msg = (
            f"{base_user_msg}\n\n【Rule Facts - copy exactly】\n"
            f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n/no_think"
        )
        response = ""
        parsed: Optional[Dict[str, Any]] = None
        validation_errors: list[str] = []
        for attempt in range(2):
            prompt = grounded_user_msg
            if attempt:
                prompt += (
                    "\n\nPrevious response was invalid. Return corrected JSON only. Errors: "
                    + "; ".join(validation_errors)
                )
            response = _semantic_generate_one(model, tokenizer, prompt, max_new_tokens, max_length)
            parsed = _extract_json_from_text(response)
            if parsed is not None:
                parsed.setdefault("rps_id", str(summary.get("rps_id") or ""))
                parsed.setdefault("bsv_image", summary.get("bsv_image"))
            validation_errors = (
                validate_semantic_memory(parsed) + validate_semantic_against_rules(parsed, summary)
                if parsed is not None
                else ["invalid JSON"]
            )
            if not validation_errors:
                break

        used_fallback = bool(validation_errors or parsed is None)
        used_general_repair = False
        if used_fallback and evaluation_mode:
            parsed = parsed or {
                "error": "Semantic output failed validation after retry",
                "raw_output": response[:500],
            }
            parsed["validation_errors"] = validation_errors
        elif used_fallback:
            logger.warning("Semantic sample %s used deterministic fallback: %s", idx + 1, validation_errors)
            safe_fallback = deterministic_semantic_memory(summary)
            repaired = _repair_semantic_with_general_provider(facts, safe_fallback)
            if repaired is not None:
                repair_errors = validate_semantic_memory(repaired) + validate_semantic_against_rules(
                    repaired, summary
                )
                parsed = repaired if not repair_errors else safe_fallback
                used_general_repair = not repair_errors
            else:
                parsed = safe_fallback
        model_info = deployment or ModelDeployment(
            key="custom_semantic", kind="semantic", base_model=base_path, adapter_path=adapter_path
        )
        parsed["_meta"] = build_memory_metadata(
            base_model=model_info.base_model,
            adapter_version=model_info.adapter_version,
            quality=quality_score("semantic", validation_errors if used_fallback else [], parsed),
            evidence_source=(
                "raw_model_invalid"
                if evaluation_mode and used_fallback
                else "deterministic_rules+general_provider_repair"
                if used_general_repair
                else "deterministic_rules"
                if used_fallback
                else model_info.evidence_source
            ),
        )
        parsed["_meta"]["runtime"] = _finish_runtime_measurement(runtime_started)
        pred = json.dumps(parsed, ensure_ascii=False)
        out_items.append(
            {
                "metadata": dict(item.get("metadata") or {}),
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": pred},
                ]
            }
        )
        if (idx + 1) % 50 == 0:
            logger.info("Semantic inference %s / %s", idx + 1, len(data))

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, indent=2, ensure_ascii=False)
    logger.info("Semantic inference saved: %s (%s items)", output_file, len(out_items))
