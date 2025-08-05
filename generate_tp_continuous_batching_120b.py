import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_path = "/fsx/vb/new-oai/gpt-oss-120b-trfs-latest"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

messages = [
    {
        "role": "user",
        "content": "Explain how expert parallelism works in large language models.",
    }
]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

generation_config = GenerationConfig(
    max_new_tokens=1000,
    do_sample=True,
)

device_map = {
    "tp_plan": "auto",  # Enable Tensor Parallelism
}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    use_kernels=True,
    attn_implementation="paged_attention|kernels-community/vllm-flash-attn3",
    **device_map,
)

model.eval()

# Tokenize and generate
inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, generation_config=generation_config)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model response:", response.split("assistant\n")[-1].strip())