# -*- coding: utf-8 -*-
"""
Scene and semantic memory inference package.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from rcpp_core.prompts import SCENE_SYSTEM_PROMPT, SEMANTIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _parse_and_validate_scene_json(
    text: str, sample_id: int
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Try strict JSON on the full string, else extract the first object for logging by sample_id.
    """
    try:
        return json.loads(text), True
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
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def _load_scene_bundle(base_path: str, adapter_path: str):
    """
    Load the fine-tuned MLLM along with image processor and tokenizer for multimodal scene understanding tasks.
    """
    import torch
    from peft.peft_model import PeftModel
    from transformers import AutoProcessor, AutoTokenizer, GenerationConfig, Qwen2_5_VLForConditionalGeneration

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logger.info("Loading scene VL base: %s", base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Loading scene LoRA: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.bfloat16)
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
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    logger.info("Loading semantic LoRA: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer


def run_scene_inference_file(
    input_file: str,
    output_file: str,
    base_path: str,
    adapter_path: str,
    bundle_cache: Dict[Tuple[str, str], Tuple[Any, Any, Any]],
) -> None:
    """
    Loop a JSON array of image and conversation items through the MLLM for scene memory inference.
    Append parsed scene JSON or an error string per item, reuse bundle_cache for model load, write output_file.
    """
    key = (base_path, adapter_path)
    if key not in bundle_cache:
        bundle_cache[key] = _load_scene_bundle(base_path, adapter_path)
    model, processor, tokenizer = bundle_cache[key]

    with open(input_file, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)

    inference_results: List[Dict] = []
    for i, item in enumerate(test_dataset):
        try:
            image_path = item["image"]
            input_conversations = item["conversations"]
            user_prompt_raw = input_conversations[0]["value"]
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
            response = _scene_predict(messages, model, processor, tokenizer)
            pred_json, is_valid = _parse_and_validate_scene_json(response, i + 1)
            result_item = {"image": image_path, "conversations": list(input_conversations)}
            if is_valid:
                result_item["conversations"].append(
                    {"from": "gpt", "value": json.dumps(pred_json, ensure_ascii=False)}
                )
            else:
                err = json.dumps(
                    {"error": "Failed to parse JSON", "raw_output": response[:200] if response else ""},
                    ensure_ascii=False,
                )
                result_item["conversations"].append({"from": "gpt", "value": err})
            inference_results.append(result_item)
        except Exception as e:
            logger.exception("Scene sample %s failed: %s", i + 1, e)
            inference_results.append(
                {
                    "image": item.get("image", ""),
                    "conversations": item.get("conversations", [])
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
) -> None:
    """
    Loop a JSON array of records with a first user message through the LLM for semantic memory inference.
    """
    key = (base_path, adapter_path)
    if key not in bundle_cache:
        bundle_cache[key] = _load_semantic_bundle(base_path, adapter_path)
    model, tokenizer = bundle_cache[key]

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_items: List[Dict] = []
    for idx, item in enumerate(data):
        user_msg = item["messages"][0]["content"]
        pred = _semantic_generate_one(model, tokenizer, user_msg, max_new_tokens, max_length)
        out_items.append(
            {
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
