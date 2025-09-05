#!/bin/bash


module purge
module load cuda/latest

# create a uv virtual environment and install the required dependencies
#
# uv venv -p 3.12
# source .venv/bin/activate
#
# for Blackwell
# uv pip install vllm[flashinfer]==0.10.1.1 --torch-backend=cu128  # not auto
#
# for others
# uv pip install vllm==0.10.1.1 --torch-backend=cu128  # not auto

model_name="openai/gpt-oss-20b"
# model_name="openai/gpt-oss-120b"

# for Blackwell
# pick only one out of the two for MoE implementation
# bf16 activation for MoE. matching reference precision (default).
export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE="1"
# mxfp8 activation for MoE. faster, but higher risk for accuracy.
# export VLLM_USE_FLASHINFER_MXFP4_MOE="1"

# for Blackwell (H200 or RTX PRO 6000 Blackwell)
export TORCH_CUDA_ARCH_LIST="12.0"
# for H100, H200 GPUs
# export TORCH_CUDA_ARCH_LIST="9.0"
# for A6000
# export TORCH_CUDA_ARCH_LIST="8.6"

# specify which GPU to use if necessary
# export CUDA_VISIBLE_DEVICES="0"

# if you use a shared machine, you need to change the port number to avoid conflicts with other users
VLLM_PORT=8000

echo "vLLM server will start at localhost:${VLLM_PORT}"
echo "You can port-forward it to your local machine with the following command:"
echo "    ssh host -L 8000:localhost:${VLLM_PORT} -N"
echo "Then you can access the vLLM server at http://localhost:8000/v1/models from your local machine"

# shellcheck source=/dev/null
source .venv/bin/activate

# for 20b and 120b with 1 GPU
vllm serve "$model_name" --port "$VLLM_PORT"

# for 120b with 2 GPUs
# vllm serve "$model_name" --port "$VLLM_PORT" --tensor-parallel-size 2
