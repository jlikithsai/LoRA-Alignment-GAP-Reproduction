import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig

# 1. Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
OUTPUT_DIR = "./qwen_dolly_full_ft_cpu"

# 2. Load Tokenizer & Model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in float32 for CPU stability
# NO quantization, NO device_map="auto" (force CPU)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="cpu", 
    dtype=torch.float32 
)

# Enable Gradient Checkpointing to save RAM (trades speed for memory)
model.gradient_checkpointing_enable()

# 3. Load & Format Dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# Create a Train/Test split
dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

def format_dolly_to_chat(example):
    user_content = example['instruction']
    if example['context']:
        user_content += f"\n\nContext:\n{example['context']}"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example['response']}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    return {"text": text}

train_dataset = train_dataset.map(format_dolly_to_chat)
eval_dataset = eval_dataset.map(format_dolly_to_chat)

# 4. Training Arguments
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=512,
    
    # Batch size must be tiny to fit gradients in RAM
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16, # Higher accumulation to simulate larger batch
    
    # Training Duration
    num_train_epochs=1,
    
    # Learning Rate (LOWER for full FT than LoRA)
    learning_rate=2e-5,  # 2e-5 is standard for full FT; LoRA uses 2e-4
    
    # Logging
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    
    # CPU Specifics
    use_cpu=True,
    fp16=False,
    bf16=False,
)

# 5. Initialize Trainer (NO PEFT CONFIG)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    processing_class=tokenizer,
)

print(f"Total Trainable Parameters: {model.num_parameters()}")

# 6. Train
print("Starting FULL FINE-TUNING on CPU... (This will be very slow)")
trainer.train()

# 7. Save
print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)