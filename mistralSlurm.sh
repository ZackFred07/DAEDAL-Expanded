#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --account=course_cap6614
#SBATCH --job-name="MISTRAL"

# Original Repo Setup
module load anaconda
conda create -n mistral python=3.10
conda activate mistral
pip install -r requirements.txt

# Fixed Requirements for Newton
pip install sacrebleu sqlitedict torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install mistral-common mistral-inference --upgrade

# Get Checkpoints (Choose Which One You Need and Comment Out What You Don't)
mkdir ckpts
cd ckpts
git lfs install
git clone https://huggingface.co/mistralai/Ministral-8B-Instruct
cd ..

# Run Evaluation (Choose Which One You Need and Comment Out What You Don't)
sh scripts/eval_Mistral.sh