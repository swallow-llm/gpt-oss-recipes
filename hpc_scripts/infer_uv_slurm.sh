#!/bin/bash
#SBATCH --job-name=infer_gpt_oss
#SBATCH --partition=your_partition
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=150G
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
# uv pip install vllm==0.10.1.1 --torch-backend=cu128  # not auto

model_name="openai/gpt-oss-20b"
# model_name="openai/gpt-oss-120b"

# for H100, H200 GPUs
export TORCH_CUDA_ARCH_LIST="9.0"
# use integer GPU index instead of UUID to avoid errors
# for 1 GPU
export CUDA_VISIBLE_DEVICES="0"
# for 2 GPUs
# export CUDA_VISIBLE_DEVICES="0,1"

# if you use gpu_1, you need to change the port number to avoid conflicts with other users
VLLM_PORT=$(( 50000 + SLURM_JOB_ID % 10000 ))
# if you use node_f, you don't need to change the port number. just use the default port 8000.
# VLLM_PORT=8000
VLLM_IP=$(hostname --ip-address | awk '{print $1}')

echo "vLLM server will start at ${VLLM_IP}:${VLLM_PORT}"
echo "You can port-forward it to your local machine with the following command:"
echo "    ssh host -L 8000:${VLLM_IP}:${VLLM_PORT} -N"
echo "Then you can access the vLLM server at http://localhost:8000/v1/models from your local machine"

# shellcheck source=/dev/null
source .venv/bin/activate

# for 20b and 120b with 1 GPU
vllm serve "$model_name" --port "$VLLM_PORT"

# for 120b with 2 GPUs
# vllm serve "$model_name" --port "$VLLM_PORT" --tensor-parallel-size 2
