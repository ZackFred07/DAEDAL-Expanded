#!/bin/bash
set -e

# Output folder for LLaMA baseline runs
BASE_OUTPUT_PATH="./results/llama_baseline"

# Choose which LLaMA model to load from HuggingFace
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

# Tasks
TASKS_GSM_MATH=("gsm8k" "math500")
TASKS_CODE=("humaneval" "mbpp")

# Generation lengths
LENGTHS=(64 128 256 512 1024)

##############################
#   GSM8K / MATH500
##############################
for task in "${TASKS_GSM_MATH[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "======================================================"
        echo "<<LLaMA Baseline>> Task: ${task}, max_gen_toks: ${length}"
        echo "======================================================"

        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"

        accelerate launch --config_file accelerate_config.yaml evaluation_script.py \
            -m dllm_eval \
            --model huggingface \
            --tasks "${task}" \
            --batch_size 2 \
            --model_args "pretrained=${MODEL_NAME},dtype=bfloat16,device=cuda" \
            --gen_kwargs "max_gen_toks=${length},temperature=0.0,do_sample=False" \
            --num_fewshot 0 \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn
    done
done


##############################
#   HUMANEVAL / MBPP
##############################
for task in "${TASKS_CODE[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "======================================================"
        echo "<<LLaMA Baseline>> Task: ${task}, max_gen_toks: ${length}"
        echo "======================================================"

        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"

        accelerate launch --config_file accelerate_config.yaml evaluation_script.py \
            -m dllm_eval \
            --model huggingface \
            --tasks "${task}" \
            --batch_size 2 \
            --model_args "pretrained=${MODEL_NAME},dtype=bfloat16,device=cuda" \
            --gen_kwargs "max_gen_toks=${length},temperature=0.0,do_sample=False" \
            --num_fewshot 0 \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --confirm_run_unsafe_code
    done
done