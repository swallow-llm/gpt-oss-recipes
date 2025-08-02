import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_path = "/fsx/vb/new-oai/gpt-oss-120b-trfs"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

messages = [
    {"role": "user", "content": "Explain tensor parallelism in simple terms."}
]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
)

device_map = {
    "tp_plan": "auto",  # Tensor Parallelism only
}

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    attn_implementation="ft-hf-o-c/vllm-flash-attn3:flash_attn_varlen_func", # Flash Attention with Sinks
    **device_map,
)

model.eval()

# Tokenize and generate
inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, generation_config=generation_config)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model response:", response.split("assistant\n")[-1].strip())