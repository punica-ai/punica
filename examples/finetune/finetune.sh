#!/bin/bash

cd "$(dirname "$0")"

if [ $# -ne 1 ]
  then
    echo "usage: $0 <dataset>"
    echo "Example datasets: gsm8k, sqlctx, viggo"
    exit 1
fi
DATASET=$1

LORA_RANK=16
OUTPUT_DIR="../../model/$DATASET-r$LORA_RANK"

python3 run-llmtuner.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --flash_attn \
    --do_train \
    --template empty \
    --dataset $DATASET \
    --dataset_dir data/ \
    --finetuning_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_RANK \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_cache \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate 5e-5 \
    --num_train_epochs 4 \
    --plot_loss \
    --fp16
