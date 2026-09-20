#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_JSONL:?set TRAIN_JSONL to the audited train.jsonl}"
: "${VAL_JSONL:?set VAL_JSONL to the audited validation.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/scene_qwen3_vl_8b_qlora_seed_${SEED:-42}}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-true}"

python training/check_training_gate.py "$TRAIN_JSONL"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
swift sft \
  --model "$MODEL" \
  --dataset "$TRAIN_JSONL" \
  --val_dataset "$VAL_JSONL" \
  --train_type lora \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant true \
  --torch_dtype bfloat16 \
  --num_train_epochs "$NUM_EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing true \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner "$FREEZE_ALIGNER" \
  --max_length 4096 \
  --attn_impl flash_attn \
  --eval_strategy steps \
  --save_strategy steps \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 3 \
  --logging_steps 5 \
  --seed "$SEED" \
  --data_seed 42 \
  --output_dir "$OUTPUT_DIR" \
  --report_to tensorboard

python training/select_best_checkpoint.py "$OUTPUT_DIR"
python training/write_run_manifest.py scene "$OUTPUT_DIR" "$MODEL" "$TRAIN_JSONL" "$VAL_JSONL" --seed "$SEED"
