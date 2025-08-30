#!/bin/bash
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=1:00:00
#$ -N infer_gpt_oss
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

module purge
# module load gcc/14.2.0
module load cuda/12.8.0

# create a uv virtual environment and install the required dependencies
#
# uv venv -p 3.12
# source .venv/bin/activate
#
# uv pip install vllm==0.10.1 --torch-backend=cu128  # not auto

model_name="openai/gpt-oss-20b"
# model_name="openai/gpt-oss-120b"

# for H100, H200 GPUs
export TORCH_CUDA_ARCH_LIST="9.0"
# use integer GPU index instead of UUID to avoid errors
# for gpu_1
export CUDA_VISIBLE_DEVICES="0"
# for node_h
# export CUDA_VISIBLE_DEVICES="0,1"

# shellcheck disable=SC2153
JOBID="$JOB_ID"
# if you use a rt_HG job, you need to change the port number to avoid conflicts with other users
VLLM_PORT=$(( 50000 + JOBID % 10000 ))
# if you use a rt_HF job, you don't need to change the port number. just use the default port 8000.
# VLLM_PORT=8000
VLLM_IP=$(hostname --ip-address | awk '{print $1}')

echo "vLLM server will start at ${VLLM_IP}:${VLLM_PORT}"
echo "You can port-forward it to your local machine with the following command:"
echo "    ssh tsubame -L 8000:${VLLM_IP}:${VLLM_PORT} -N"
echo "Then you can access the vLLM server at http://localhost:8000/v1/models from your local machine"

# shellcheck source=/dev/null
source .venv/bin/activate

# for 20b and 120b with 1 GPU with gpu_1=1
vllm serve "$model_name" --port "$VLLM_PORT"

# for 120b with 2 GPUs with node_h=1
# vllm serve "$model_name" --port "$VLLM_PORT" --tensor-parallel-size 2
