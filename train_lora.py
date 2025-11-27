import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# 1. Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
OUTPUT_DIR = "./qwen_dolly_cpu_final"

# 2. Load Tokenizer & Model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in float32 for CPU stability
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="cpu", 
    dtype=torch.float32 
)

# 3. Setup LoRA Config
# We DO NOT manually wrap the model here. We pass this config to the Trainer.
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 4. Load & Format Dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
dataset =  dataset.select(range(2000))
# Create a Train/Test split (90% train, 10% validation)
# This allows us to log evaluation metrics to see if the model is actually learning
dataset_split = dataset.train_test_split(test_size=0.1)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

print(f"Training on {len(train_dataset)} examples")
print(f"Validating on {len(eval_dataset)} examples")

def format_dolly_to_chat(example):
    """
    Converts Dolly dataset format to Qwen's chat template.
    """
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

# Format both splits
train_dataset = train_dataset.map(format_dolly_to_chat)
eval_dataset = eval_dataset.map(format_dolly_to_chat)

# 5. Training Arguments
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=512,
    
    # Batch size: Small for CPU to prevent RAM overload
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    
    # Training Duration: 1 full pass over the data
    num_train_epochs=1,
    
    # Learning Rate
    learning_rate=2e-4,
    
    # Logging & Evaluation
    logging_strategy="steps",
    logging_steps=50,           # Log every 50 steps
    eval_strategy="steps",      # Evaluate every X steps
    eval_steps=500,             # Check validation loss every 500 steps
    
    # Saving
    save_strategy="steps",
    save_steps=500,             # Save a checkpoint every 500 steps
    save_total_limit=2,         # Only keep the last 2 checkpoints to save disk space
    
    # CPU Specifics
    use_cpu=True,
    fp16=False,
    bf16=False,
)

# 6. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass eval set here
    args=training_args,
    peft_config=peft_config,    # Pass PEFT config here (Trainer handles wrapping)
    processing_class=tokenizer,
)

# Optional: Print trainable parameters correctly before starting
# We access the model inside the trainer to see the wrapped version
print("Model Adapter Status:")
trainer.model.print_trainable_parameters()

# 7. Train
print("Starting full training on CPU...")
trainer.train()

# 8. Save Final Model
print(f"Saving final model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)