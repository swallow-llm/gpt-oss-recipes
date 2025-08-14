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
module load singularitypro/4.1.7

# build the Singularity image if not already done
# singularity build vllm-openai-gptoss.sif docker://vllm/vllm-openai:gptoss

# for H100, H200 GPUs
export TORCH_CUDA_ARCH_LIST="9.0"

# use integer GPU index instead of UUID to avoid errors
export CUDA_VISIBLE_DEVICES="0"

# for 20b with 1 GPU
# singularity run --nv -B /apps vllm-openai-gptoss.sif \
#     --model openai/gpt-oss-20b

# for 120b with 1 GPU
singularity run --nv -B /apps vllm-openai-gptoss.sif \
    --model openai/gpt-oss-120b

# for 120b with 8 GPUs with a rt_HF job (not rt_HG)
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# singularity run --nv -B /apps vllm-openai-gptoss.sif \
#     --model openai/gpt-oss-120b \
#     --tensor-parallel-size 8
