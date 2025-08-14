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

# build the Singularity image if not already done
# singularity build vllm-openai-gptoss.sif docker://vllm/vllm-openai:gptoss

# for H100 GPUs
export TORCH_CUDA_ARCH_LIST="9.0"

# for 20b model with 1 GPU
# singularity run --nv -B /home/appli vllm-openai-gptoss.sif  \
#     --model openai/gpt-oss-20b

# for 120b model with 1 GPU
# singularity run --nv -B /home/appli vllm-openai-gptoss.sif  \
#     --model openai/gpt-oss-120b \
#     --max_model_len 116288  # default is 131072 (128k)

# for 120b model with 2 GPUs
singularity run --nv -B /home/appli vllm-openai-gptoss.sif  \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 2
