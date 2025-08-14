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


ulimit -v unlimited
ulimit -m unlimited
ulimit -l unlimited

module purge
module load cuda/12.8.0

# build the Singularity image if not already done
# apptainer build vllm-openai-gptoss.sif docker://vllm/vllm-openai:gptoss

# for H100 GPUs
export TORCH_CUDA_ARCH_LIST="9.0"

# for 20b with 1 GPU
# apptainer run --nv -B /gs -B /apps vllm-openai-gptoss.sif \
#     --model openai/gpt-oss-20b

# for 120b with 1 GPU
# apptainer run --nv -B /gs -B /apps vllm-openai-gptoss.sif \
#     --model openai/gpt-oss-120b \
#     --max_model_len 116288  # default is 131072 (128k)

# for 120b with 2 GPUs
apptainer run --nv -B /gs -B /apps vllm-openai-gptoss.sif \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 2
