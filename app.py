import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup - Define the specific model version
model_name = "Qwen/Qwen2.5-0.5B"

# 2. Load Model & Tokenizer
# device_map="auto" will automatically use GPU if available, otherwise CPU
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Define your prompt
prompt = "Explain why the sky is blue"

# 4. Format prompt for the Instruct model
# Qwen expects specific "chat" formatting (System/User roles)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 5. Tokenize and Generate
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating response...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128
)

# 6. Decode output
# The model returns the input + the new text, so we clean it up
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("-" * 30)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
print("-" * 30)
