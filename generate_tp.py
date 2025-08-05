import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Model configuration - uncomment the model size you want to use
model_path = "openai/gpt-oss-120b"  # 120B model (default)
# model_path = "openai/gpt-oss-20b"  # 20B model - uncomment this line and comment the line above

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

messages = [{"role": "user", "content": "Explain tensor parallelism in simple terms."}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
)

device_map = {
    "tp_plan": "auto",  # Tensor Parallelism only
} if "120b" in model_path else "auto"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    **device_map,
)

model.eval()

# Tokenize and generate
inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, generation_config=generation_config)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model response:", response.split("assistant\n")[-1].strip())