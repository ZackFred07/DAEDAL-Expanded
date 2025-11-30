#!/bin/bash
set -e


BASE_OUTPUT_PATH="./results/ldaedal_1p5"
MODEL_PATH="./ckpts/LLaDA-1.5"


TASKS=("gsm8k" "math500" "svamp")
LENGTHS=(64)
for task in "${TASKS[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "======================================================"
        echo "<<LDAEDAL>> -> Task: ${task}, L_init: ${length}"
        echo "======================================================"
        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"

        accelerate launch --config_file accelerate_config.yaml evaluation_script.py \
            -m dllm_eval \
            --model LLaDA_LDAEDAL \
            --tasks "${task}" \
            --batch_size 2 \
            --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
            --gen_kwargs "block_length=32,initial_gen_length=${length},max_gen_length=2048,cfg_scale=0.0,high_conf_threshold=0.9,low_conf_threshold=0.1,eos_confidence_threshold=0.5,expand_eos_confidence_threshold=0.9,expansion_factor=8,eos_check_tokens=32 "  \
            --num_fewshot 0  \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn

        python metrics/${task}.py \
            --model_path "${MODEL_PATH}" \
            --res_path "${OUTPUT_PATH}"
    done
done


TASKS=("humaneval" "mbpp" "mmlu")
LENGTHS=(64)
for task in "${TASKS[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "======================================================"
        echo "<<LDAEDAL>> -> Task: ${task}, L_init: ${length}"
        echo "======================================================"
        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"

        accelerate launch --config_file accelerate_config.yaml evaluation_script.py \
            -m dllm_eval \
            --model LLaDA_LDAEDAL \
            --tasks "${task}" \
            --batch_size 2 \
            --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
            --gen_kwargs "block_length=32,initial_gen_length=${length},max_gen_length=2048,cfg_scale=0.0,high_conf_threshold=0.9,low_conf_threshold=0.1,eos_confidence_threshold=0.5,expand_eos_confidence_threshold=0.9,expansion_factor=8,eos_check_tokens=32 "  \
            --num_fewshot 0  \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --confirm_run_unsafe_code

        python metrics/${task}.py \
            --model_path "${MODEL_PATH}" \
            --res_path "${OUTPUT_PATH}"
    done
done
