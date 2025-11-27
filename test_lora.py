import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./qwen_dolly_cpu_finetune"  # Directory where you saved the model

def run_inference():
    print("Loading base model...")
    # Load base model in float32 for CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="cpu",
        dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print(f"Loading LoRA adapters from {ADAPTER_PATH}...")
    adapter_loaded = False
    try:
        # Load the fine-tuned adapters
        # We do NOT merge them here so we can toggle them on/off
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        adapter_loaded = True
    except Exception as e:
        print(f"Error loading adapters: {e}")
        print("Falling back to base model only...")
        model = base_model

    model.eval()
    
    print("\n" + "="*50)
    print("Model ready! Type 'quit' to exit.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # Format input using Qwen's chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Helper function to generate text
        def generate_response(model_instance):
            with torch.no_grad():
                generated_ids = model_instance.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            # Decode output (remove input tokens)
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("-" * 30)
        
        # 1. Generate with Original Model (Base)
        print("Generating Original Response...")
        if adapter_loaded:
            # Temporarily disable the LoRA adapter to run the base model
            with model.disable_adapter():
                response_base = generate_response(model)
        else:
            response_base = generate_response(model)
            
        print(f"Original: {response_base}\n")

        # 2. Generate with Fine-tuned Model (LoRA)
        if adapter_loaded:
            print("Generating Fine-tuned Response...")
            response_tuned = generate_response(model)
            print(f"Fine-tuned: {response_tuned}\n")
        
        print("-" * 30)

if __name__ == "__main__":
    run_inference()