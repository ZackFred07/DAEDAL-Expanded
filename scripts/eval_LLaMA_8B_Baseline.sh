#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ACCELERATE_NUM_PROCESSES=1
unset WORLD_SIZE RANK LOCAL_RANK NODE_RANK
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

BASE_OUTPUT_PATH="./results/llama_8b_baseline"
# if you have the local checkpoint folder, point to it. Otherwise leave as HF id and require login
MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"

TASKS=("gsm8k" "math500")
LENGTHS=(32 64 128 256 512)

for task in "${TASKS[@]}"; do
  for length in "${LENGTHS[@]}"; do
    echo "======================================================"
    echo "<<LLaMA Baseline>> -> Task: ${task}, L_init: ${length}"
    echo "======================================================"
    OUT="${BASE_OUTPUT_PATH}/${task}_${length}"

    python evaluation_script.py \
      -m dllm_eval \
      --model LLaMA \
      --tasks "${task}" \
      --batch_size 1 \
      --model_args "pretrained=${MODEL_PATH}" \
      --gen_kwargs "max_new_tokens=${length},temperature=0.2,top_p=0.95" \
      --num_fewshot 0 \
      --output_path "${OUT}" \
      --log_samples \
      --apply_chat_template \
      --fewshot_as_multiturn

    python metrics/${task}.py --model_path "${MODEL_PATH}" --res_path "${OUT}"
  done
done
