# Batch scripts for various HPC clusters

- [ABCI 3.0](https://docs.abci.ai/v3/)
- [TSUBAME 4.0](https://www.t4.cii.isct.ac.jp/manuals)
- [Slurm](https://slurm.schedmd.com/)


## Pre-requisites

- [Install uv](https://docs.astral.sh/uv/getting-started/installation/) if not already installed.

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Git clone the repository.

```shell
git clone https://github.com/swallow-llm/gpt-oss-recipes.git
cd gpt-oss-recipes
```

## Inference

### How to prepare uv virtual environments

- Create a uv virtual environment and install the required dependencies. See [this instructions](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) for more details.
- For ABCI 3.0,

```shell
source /etc/profile.d/modules.sh

module purge
module load cuda/12.8/12.8.1

export UV_LINK_MODE=copy

uv venv -p 3.12
source .venv/bin/activate

uv pip install vllm==0.10.1 --torch-backend=cu128  # not auto
```

- For TSUBAME 4.0,

```shell
module purge
module load gcc/14.2.0
module load cuda/12.8.0

export UV_LINK_MODE=copy

uv venv -p 3.12
source .venv/bin/activate

uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match \
    --no-cache
```

### How to prepare vLLM containers

- For ABCI 3.0, convert the [vLLM Docker image](https://hub.docker.com/r/vllm/vllm-openai/tags) to a Singularity image.

```shell
source /etc/profile.d/modules.sh
module load singularitypro/4.1.7
singularity pull vllm-openai-v0.10.1.1.sif docker://vllm/vllm-openai:v0.10.1.1
```

- For TSUBAME 4.0, convert the [vLLM Docker image](https://hub.docker.com/r/vllm/vllm-openai/tags) to a Apptainer image.

```shell
apptainer pull vllm-openai-v0.10.1.1.sif docker://vllm/vllm-openai:v0.10.1.1
```

## Supervised Fine-tuning

### How to prepare uv virtual environments

- Create a uv virtual environment and install the required dependencies. See [this instructions](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers) for more details.
- For ABCI 3.0,

```shell
source /etc/profile.d/modules.sh

module purge
module load cuda/12.8/12.8.1

export UV_LINK_MODE=copy

uv venv -p 3.12
source .venv/bin/activate

uv pip install "torch>=2.8.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install "trl>=0.21.0" "peft>=0.17.0" "transformers>=4.55.2" "triton>=3.4" "kernels>=0.9.0"
uv pip install accelerate deepspeed trackio wandb
hf auth login
wandb login
```
