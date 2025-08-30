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
module load cuda/12.8.0

# pull the Singularity image if not already done
# apptainer pull vllm-openai-v0.10.1.1.sif docker://vllm/vllm-openai:v0.10.1.1
sif_path="${EXP_BS_HOME}/sif/vllm-openai-v0.10.1.1.sif"
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

# for 20b and 120b with 1 GPU with gpu_1=1
apptainer run --nv -B /gs -B /apps "$sif_path" --model "$model_name" --port "$VLLM_PORT"

# for 120b with 2 GPUs with node_h=1
# apptainer run --nv -B /gs -B /apps "$sif_path" --model "$model_name" --port "$VLLM_PORT" \
#     --tensor-parallel-size 2
