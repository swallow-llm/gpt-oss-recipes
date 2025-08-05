# OpenAI GPT-OSS Recipes

Collection of scripts demonstrating different optimization and fine-tuning techniques for OpenAI's GPT-OSS models (20B and 120B parameters).

## Scripts

- `generate_tp.py` - Model with Tensor Parallelism.
- `generate_flash_attention.py` - Model with Flash Attention + Tensor Parallelism.
- `generate_tp_continuous_batching.py` - Model with Flash Attnetion + Tensor Parallelism and continuous batching.
- `generate_all.py` - Model with all optimizations: Expert Parallelism, Tensor Parallelism, Flash Attention.
- `sft.py` - Script for fine-tuning the model using supervised fine-tuning (SFT). Supports both full-parameter training and LoRA training.

### Model Configuration

All generation scripts support both 20B and 120B models. To switch between model sizes, simply edit the `model_path` variable at the top of each script:

```python
# Model configuration - uncomment the model size you want to use
model_path = "openai/gpt-oss-120b"  # 120B model (default)
# model_path = "openai/gpt-oss-20b"  # 20B model - uncomment this line and comment the line above
```

The scripts automatically configure the appropriate device mapping and settings based on the selected model size.

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

Before running any script, edit the `model_path` variable to select your desired model size (20B or 120B).

Run a generation script:

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
accelerate launch --config_file zero3.yaml sft.py --config configs/sft_full.yaml
```

For LoRA training on one GPU, run:

```bash
python sft.py --config configs/sft_lora.yaml
```

To change the dataset or training hyperparameters, either modify the `sft_lora.yaml` or `sft_full.yaml` files or pass them as command line arguments e.g.:

```bash
accelerate launch --config_file zero3.yaml \
    sft.py --config sft_full.yaml \
    --dataset_name DATASET_NAME
```