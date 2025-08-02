# GPT OSS Recipes

Collection of scripts demonstrating different optimization techniques for OpenAI's GPT OSS models (20B and 120B parameters).

## Scripts

- `generate_flash_attention_20b.py` - 20B model with Flash Attention
- `generate_tp_20b.py` - 20B model with Tensor Parallelism  
- `generate_tp_120b.py` - 120B model with Tensor Parallelism
- `generate_flash_attention_tp_120b.py` - 120B model with Flash Attention + Tensor Parallelism
- `generate_all_120b.py` - 120B model with all optimizations (Expert Parallelism, Tensor Parallelism, Flash Attention)

## Usage

```bash
python generate_<script_name>.py
```

or for distributed:

```bash
torchrun --nproc_per_node=x generate_<script_name>.py
```

**Note:** Scripts expect model files to be available at `/fsx/vb/new-oai/gpt-oss-{20b,120b}-trfs`. Update the `model_path` variable in each script to point to your model directory.
