#!/bin/bash
#PBS -q rt_HF
#PBS -l select=2
#PBS -l walltime=1:00:00
#PBS -N sft_multi_gpt_oss_20b
#PBS -k oe
#PBS -j oe
#PBS -v USE_SSH=1


cd "${PBS_O_WORKDIR}" || exit

# shellcheck source=/dev/null
source /etc/profile.d/modules.sh

module purge
module load cuda/12.8/12.8.1

# git clone https://github.com/susumuota/gpt-oss-recipes.git
# cd gpt-oss-recipes
#
# uv venv -p 3.12
# source .venv/bin/activate

# uv pip install "torch>=2.8.0" --index-url https://download.pytorch.org/whl/cu128
# uv pip install "trl>=0.21.0" "peft>=0.17.0" "transformers>=4.55.2" "triton>=3.4" "kernels>=0.9.0"
# uv pip install accelerate deepspeed trackio wandb
# hf auth login
# wandb login

# shellcheck source=/dev/null
source .venv/bin/activate

export NUM_MACHINES=$(cat "${PBS_NODEFILE}" | uniq | wc -l)
export NUM_GPUS=$(nvidia-smi -L | grep -c "^GPU")
export NUM_PROCESSES=$(( NUM_MACHINES * NUM_GPUS ))
export JOB_ID=$(echo "${PBS_JOBID}" | cut -d '.' -f 1)
export MASTER_ADDR=$(cat "${PBS_NODEFILE}" | uniq | head -n 1)
export MASTER_PORT=$(( 50000 + JOB_ID % 10000 ))
export HOSTFILE="hostfile.${JOB_ID}"

echo "NUM_MACHINES: ${NUM_MACHINES}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "JOB_ID: ${JOB_ID}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "HOSTFILE: ${HOSTFILE}"

awk '{print $0 " slots=8"}' "${PBS_NODEFILE}" > "${HOSTFILE}"
cat "${HOSTFILE}"

accelerate launch \
    --config_file configs/zero3.yaml \
    --deepspeed_multinode_launcher pdsh \
    --deepspeed_hostfile "${HOSTFILE}" \
    --num_machines "${NUM_MACHINES}" \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --rdzv_backend c10d \
    --max_restarts 0 \
    sft.py \
    --config configs/sft_full.yaml \
    --attn_implementation kernels-community/vllm-flash-attn3 \
    --report_to wandb \
    --max_length 16384
