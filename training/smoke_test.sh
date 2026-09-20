#!/usr/bin/env bash
set -euo pipefail

KIND="${1:?usage: smoke_test.sh scene|semantic AUDITED_DATASET_DIR OUTPUT_DIR}"
DATASET_DIR="${2:?usage: smoke_test.sh scene|semantic AUDITED_DATASET_DIR OUTPUT_DIR}"
OUTPUT_DIR="${3:?usage: smoke_test.sh scene|semantic AUDITED_DATASET_DIR OUTPUT_DIR}"
SMOKE_JSONL="$DATASET_DIR/smoke_20.jsonl"
python training/make_smoke_subset.py "$DATASET_DIR/train.jsonl" "$SMOKE_JSONL" --count 20

if [[ "$KIND" == "scene" ]]; then
  TRAIN_JSONL="$SMOKE_JSONL" VAL_JSONL="$SMOKE_JSONL" OUTPUT_DIR="$OUTPUT_DIR" MAX_STEPS=2 NUM_EPOCHS=1 SAVE_STEPS=1 EVAL_STEPS=1 bash training/train_scene_qlora.sh
  MODEL_KEY=scene_qwen3_vl_8b_qlora
else
  TRAIN_JSONL="$SMOKE_JSONL" VAL_JSONL="$SMOKE_JSONL" OUTPUT_DIR="$OUTPUT_DIR" MAX_STEPS=2 NUM_EPOCHS=1 SAVE_STEPS=1 EVAL_STEPS=1 bash training/train_semantic_qlora.sh
  MODEL_KEY=semantic_qwen3_8b_qlora
fi

CHECKPOINT="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail -n 1)"
if [[ -z "$CHECKPOINT" ]]; then
  echo "No checkpoint produced by smoke training" >&2
  exit 2
fi
python evaluation/run_memory_inference.py "$KIND" "$SMOKE_JSONL" "$OUTPUT_DIR/reload_predictions.json" --model-key "$MODEL_KEY" --adapter-path "$CHECKPOINT"
