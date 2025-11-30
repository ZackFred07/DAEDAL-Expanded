#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --account=course_cap6614
#SBATCH --job-name="LDAEDAL_LLADA"

# Original Repo Setup
module load anaconda
conda create -n daedal python=3.10
conda activate daedal
pip install -r requirements.txt

# Fixed Requirements for Newton
pip install sacrebleu sqlitedict torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# Get Checkpoints (Choose Which One You Need and Comment Out What You Don't)
mkdir ckpts
cd ckpts
git lfs install
git clone https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
git clone https://huggingface.co/GSAI-ML/LLaDA-1.5
git clone https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
cd ..

# Run Evaluation (Choose Which One You Need and Comment Out What You Don't)
# sh scripts/eval_LLaDA_1p5_DAEDAL.sh
# sh scripts/eval_LLaDA_1p5_Baseline.sh
# sh scripts/eval_LLaDA_DAEDAL.sh
# sh scripts/eval_LLaDA_Baseline.sh
# sh scripts/eval_LLaMA_8B_Baseline.sh
# sh scripts/eval_Dream_Baseline.sh
# sh scripts/eval_Dream_DAEDAL.sh
sh scripts/eval_LLaDA_LDAEDAL.sh
# sh scripts/eval_LLaDA_1p5_LDAEDAL.sh
# sh scripts/eval_Dream_LDAEDAL.sh
