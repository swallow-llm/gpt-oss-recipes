import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_path = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages = [
    {"role": "user", "content": "Explain tensor parallelism in simple terms."}
]

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    # Flash Attention with Sinks
    attn_implementation="kernels-community/vllm-flash-attn3:flash_attn_varlen_func",
    device_map="auto",
)

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)

# Decode and skip everything before the `final` channel with the answer
response = tokenizer.decode(outputs[0])
print("Model response:", response.split("<|channel|>final<|message|>")[-1].strip())
