#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=8:00:00
#$ -N sft_gpt_oss_20b
#$ -e $JOB_NAME_$JOB_ID.err
#$ -o $JOB_NAME_$JOB_ID.out


# set group disk directory
export EXP_BS_HOME="/gs/bs/your_group/your_home"

export HF_HOME="${EXP_BS_HOME}/.cache/huggingface"
export UV_CACHE_DIR="${EXP_BS_HOME}/.cache/uv"
export UV_LINK_MODE=copy
export APPTAINER_CACHEDIR="${EXP_BS_HOME}/.apptainer"
export APPTAINER_MKSQUASHFS_ARGS="-processors 1"
export VLLM_CACHE_ROOT="${EXP_BS_HOME}/.cache/vllm"
export FLASHINFER_WORKSPACE_BASE="${EXP_BS_HOME}"


ulimit -v unlimited
ulimit -m unlimited
ulimit -l unlimited

module purge
module load cuda/12.8.0


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
