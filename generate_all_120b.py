from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from transformers.distributed import DistributedConfig

model_path = "openai/gpt-oss-120b"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

# Set up chat template
messages = [
    {
        "role": "user",
        "content": "Explain how expert parallelism works in large language models.",
    }
]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
)

device_map = {
    "distributed_config": DistributedConfig(
        enable_expert_parallel=1
    ),  # Enable Expert Parallelism
    "tp_plan": "auto",  # Enables Tensor Parallelism
}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    attn_implementation="vllm-flash-attn3:flash_attn_varlen_func",
    **device_map,
)

model.eval()

# Tokenize and generate
inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, generation_config=generation_config)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model response:", response.split("assistant\n")[-1].strip())
