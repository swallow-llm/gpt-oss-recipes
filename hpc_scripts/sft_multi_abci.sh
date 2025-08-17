#!/bin/bash
#PBS -q rt_HF
#PBS -l select=2
#PBS -l walltime=8:00:00
#PBS -N sft_multi_gpt_oss_20b
#PBS -k oe
#PBS -j oe
#PBS -v USE_SSH=1


cd "${PBS_O_WORKDIR}" || exit

# shellcheck source=/dev/null
source /etc/profile.d/modules.sh

module purge
module load cuda/12.8/12.8.1
module load hpcx/2.20

# git clone https://github.com/susumuota/gpt-oss-recipes.git
# cd gpt-oss-recipes
#
# uv venv -p 3.12
# source .venv/bin/activate

# uv pip install "torch>=2.8.0" --index-url https://download.pytorch.org/whl/cu128
# uv pip install "trl>=0.21.0" "peft>=0.17.0" "transformers>=4.55.2" trackio
# uv pip install deepspeed

# shellcheck source=/dev/null
source .venv/bin/activate

export NUM_MACHINES=$(cat "${PBS_NODEFILE}" | uniq | wc -l)
export NUM_PROCESSES=$(( NUM_MACHINES * 8 ))
export JOBID=$(echo "${PBS_JOBID}" | cut -d '.' -f 1)
export MASTER_ADDR=$(cat "${PBS_NODEFILE}" | uniq | head -n 1)
export MASTER_PORT=$(( 50000 + JOBID % 10000 ))

echo "NUM_MACHINES: ${NUM_MACHINES}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "JOBID: ${JOBID}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

export HOSTFILE="hostfile.${JOBID}"
echo "HOSTFILE: $HOSTFILE"

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
    --max_restarts 0 \
    sft.py \
    --config configs/sft_full.yaml
