#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -N infer_gpt_oss_120b
#PBS -k oe


cd "${PBS_O_WORKDIR}" || exit

# shellcheck source=/dev/null
source /etc/profile.d/modules.sh

module purge
module load cuda/12.8/12.8.1

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

# for 20b with 1 GPU
# vllm serve openai/gpt-oss-20b

# for 120b with 1 GPU
vllm serve openai/gpt-oss-120b

# for 120b with 8 GPUs with a rt_HF job (not rt_HG)
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# vllm serve openai/gpt-oss-120b --tensor-parallel-size 8
