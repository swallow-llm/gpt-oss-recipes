#!/bin/bash
#SBATCH --job-name=infer_gpt_oss_120b
#SBATCH --partition=your_partition
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=60
#SBATCH --mem=300G
#SBATCH --output=%x_%j_%n.out
#SBATCH --error=%x_%j_%n.err


ulimit -v unlimited
ulimit -m unlimited
ulimit -l unlimited

module purge
module load cuda/12.8

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
