#!/bin/bash
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=8:00:00
#$ -N infer_gpt_oss_120b


# set group disk directory
export EXP_BS_HOME="/gs/bs/your_group/your_home"

export HF_HOME="${EXP_BS_HOME}/.cache/huggingface"
export UV_CACHE_DIR="${EXP_BS_HOME}/.cache/uv"
export UV_LINK_MODE=copy
export APPTAINER_CACHEDIR="${EXP_BS_HOME}/.apptainer"
export APPTAINER_MKSQUASHFS_ARGS="-processors 1"
export VLLM_CACHE_ROOT="${EXP_BS_HOME}/.cache/vllm"
export FLASHINFER_WORKSPACE_BASE="${EXP_BS_HOME}"

module purge
module load gcc/14.2.0
module load cuda/12.8.0

# create a uv virtual environment and install the required dependencies
#
# uv venv -p 3.12
# source .venv/bin/activate
#
# uv pip install --pre vllm==0.10.1+gptoss \
#     --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
#     --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
#     --index-strategy unsafe-best-match \
#     --no-cache

# for H100, H200 GPUs
export TORCH_CUDA_ARCH_LIST="9.0"

# use integer GPU index instead of UUID to avoid errors
export CUDA_VISIBLE_DEVICES="0"

# shellcheck source=/dev/null
source .venv/bin/activate

# for 20b with 1 GPU with gpu_1=1
# vllm serve openai/gpt-oss-20b

# for 120b with 1 GPU with gpu_1=1
# vllm serve openai/gpt-oss-120b --max_model_len 116288  # default is 131072 (128k)

# for 120b with 2 GPUs with node_h=1
export CUDA_VISIBLE_DEVICES="0,1"
vllm serve openai/gpt-oss-120b --tensor-parallel-size 2
