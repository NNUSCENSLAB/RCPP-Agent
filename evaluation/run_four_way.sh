#!/usr/bin/env bash
set -euo pipefail

KIND="${1:?usage: run_four_way.sh scene|semantic TEST_JSONL OUTPUT_DIR}"
TEST_JSONL="${2:?usage: run_four_way.sh scene|semantic TEST_JSONL OUTPUT_DIR}"
OUTPUT_DIR="${3:?usage: run_four_way.sh scene|semantic TEST_JSONL OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

if [[ "$KIND" == "scene" ]]; then
  OLD_KEY=scene_qwen25_current
  NEW_KEY=scene_qwen3_vl_8b_qlora
else
  OLD_KEY=semantic_qwen25_current
  NEW_KEY=semantic_qwen3_8b_qlora
fi

python evaluation/run_memory_inference.py "$KIND" "$TEST_JSONL" "$OUTPUT_DIR/old_base.json" --model-key "$OLD_KEY" --base-only
python evaluation/run_memory_inference.py "$KIND" "$TEST_JSONL" "$OUTPUT_DIR/old_lora.json" --model-key "$OLD_KEY"
python evaluation/run_memory_inference.py "$KIND" "$TEST_JSONL" "$OUTPUT_DIR/new_base.json" --model-key "$NEW_KEY" --base-only
python evaluation/run_memory_inference.py "$KIND" "$TEST_JSONL" "$OUTPUT_DIR/new_qlora.json" --model-key "$NEW_KEY"

python evaluation/evaluate_memory.py "$KIND" "$TEST_JSONL" "$OUTPUT_DIR/report.json" \
  "old_base=$OUTPUT_DIR/old_base.json" \
  "old_lora=$OUTPUT_DIR/old_lora.json" \
  "new_base=$OUTPUT_DIR/new_base.json" \
  "new_qlora=$OUTPUT_DIR/new_qlora.json"

if ! python evaluation/promotion_gate.py "$OUTPUT_DIR/report.json" --kind "$KIND" --output "$OUTPUT_DIR/promotion.json"; then
  echo "Candidate did not pass promotion gate; production registry remains unchanged."
fi
