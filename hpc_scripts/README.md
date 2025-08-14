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

- For ABCI 3.0 and TSUBAME 4.0, create a uv virtual environment and install the required dependencies. See [this instructions](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) for more details.

```shell
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
module load singularitypro/4.1.7
singularity build vllm-openai-gptoss.sif docker://vllm/vllm-openai:gptoss
```

- For TSUBAME 4.0, convert the [vLLM Docker image](https://hub.docker.com/r/vllm/vllm-openai/tags) to a Apptainer image.

```shell
apptainer build vllm-openai-gptoss.sif docker://vllm/vllm-openai:gptoss
```

