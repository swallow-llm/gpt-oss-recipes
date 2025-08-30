#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -N infer_gpt_oss
#PBS -k oe
#PBS -j oe
#PBS -v USE_SSH=1


cd "${PBS_O_WORKDIR}" || exit

# shellcheck source=/dev/null
source /etc/profile.d/modules.sh

module purge
module load singularitypro/4.1.7
module load cuda/12.8/12.8.1

# pull the Singularity image if not already done
# singularity pull vllm-openai-v0.10.1.1.sif docker://vllm/vllm-openai:v0.10.1.1
sif_path="${HOME}/sif/vllm-openai-v0.10.1.1.sif"
model_name="openai/gpt-oss-20b"
# model_name="openai/gpt-oss-120b"

# for H100, H200 GPUs
export TORCH_CUDA_ARCH_LIST="9.0"
# use integer GPU index instead of UUID to avoid errors
# for rt_HG
export CUDA_VISIBLE_DEVICES="0"
# for rt_HF
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

JOBID=$(echo "${PBS_JOBID}" | cut -d '.' -f 1)
# if you use a rt_HG job, you need to change the port number to avoid conflicts with other users
VLLM_PORT=$(( 50000 + JOBID % 10000 ))
# if you use a rt_HF job, you don't need to change the port number. just use the default port 8000.
# VLLM_PORT=8000
VLLM_IP=$(hostname --ip-address | awk '{print $1}')

echo "vLLM server will start at ${VLLM_IP}:${VLLM_PORT}"
echo "You can port-forward it to your local machine with the following command:"
echo "    ssh abci -L 8000:${VLLM_IP}:${VLLM_PORT} -N"
echo "Then you can access the vLLM server at http://localhost:8000/v1/models from your local machine"

# for 20b and 120b with 1 GPU with a rt_HG job
singularity run --nv -B /apps "$sif_path" --model "$model_name" --port "$VLLM_PORT"

# for 120b with 8 GPUs with a rt_HF job (not rt_HG)
# singularity run --nv -B /apps "$sif_path" --model "$model_name" --port "$VLLM_PORT" --tensor-parallel-size 8
