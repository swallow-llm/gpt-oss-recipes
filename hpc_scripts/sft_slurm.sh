#!/bin/bash
#SBATCH --job-name=sft_gpt_oss_20b
#SBATCH --partition=your_partition
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --mem=1200G
#SBATCH --output=%x_%j_%n.out
#SBATCH --error=%x_%j_%n.err


ulimit -v unlimited
ulimit -m unlimited
ulimit -l unlimited

module purge
module load cuda/12.8

# git clone https://github.com/susumuota/gpt-oss-recipes.git
# cd gpt-oss-recipes
#
# uv venv gpt-oss --python 3.11 && source gpt-oss/bin/activate && uv pip install --upgrade pip
# uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
# skip this for H100:
#     uv pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
# uv pip install -r requirements.txt
# uv pip install deepspeed

# shellcheck source=/dev/null
source gpt-oss/bin/activate

accelerate launch \
    --config_file configs/zero3.yaml \
    sft.py \
    --config configs/sft_full.yaml
