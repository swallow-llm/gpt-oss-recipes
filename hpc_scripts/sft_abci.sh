#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -N sft_gpt_oss_20b
#PBS -k oe


cd "${PBS_O_WORKDIR}" || exit

# shellcheck source=/dev/null
source /etc/profile.d/modules.sh

module purge
module load cuda/12.8/12.8.1

# git clone https://github.com/susumuota/gpt-oss-recipes.git
# cd gpt-oss-recipes
#
# uv venv -p 3.12
# source .venv/bin/activate

# uv pip install "torch>=2.8.0" --index-url https://download.pytorch.org/whl/cu128
# uv pip install "trl>=0.21.0" "peft>=0.17.0" "transformers>=4.55.2" trackio
# uv pip install deepspeed

# shellcheck source=/dev/null
source .venv/bin/activate

accelerate launch \
    --config_file configs/zero3.yaml \
    sft.py \
    --config configs/sft_full.yaml
