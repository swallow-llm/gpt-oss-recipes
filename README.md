# OpenAI GPT-OSS Recipes

Collection of scripts demonstrating different optimization techniques for OpenAI's GPT-OSS models (20B and 120B parameters).

## Scripts

- `generate_flash_attention_20b.py` - 20B model with Flash Attention
- `generate_tp_20b.py` - 20B model with Tensor Parallelism  
- `generate_tp_120b.py` - 120B model with Tensor Parallelism
- `generate_flash_attention_tp_120b.py` - 120B model with Flash Attention + Tensor Parallelism
- `generate_all_120b.py` - 120B model with all optimizations (Expert Parallelism, Tensor Parallelism, Flash Attention)

## Installation

First create a virtual environment using e.g. `uv`:

```sh
uv venv gpt-oss --python 3.11 && source gpt-oss/bin/activate && uv pip install --upgrade pip
```

Next install PyTorch:

```sh
uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```

Finall install the remaining dependencies:

```sh
# TODO: Update the requirements to use latest transformers
uv pip install -r requirements.txt
```

## Usage

### Inference

```bash
python generate_<script_name>.py
```

or for distributed:

```bash
torchrun --nproc_per_node=x generate_<script_name>.py
```

### Training

For full-parameter training on one node of 8 GPUs, run:

```bash
accelerate launch --config_file zero3.yaml sft.py --config sft_full.yaml
```

For LoRA training on one GPU, run:

```bash
python sft.py --config sft_lora.yaml
```

**Note:** Scripts expect model files to be available at `/fsx/vb/new-oai/gpt-oss-{20b,120b}-trfs`. Update the `model_path` variable in each script to point to your model directory.
