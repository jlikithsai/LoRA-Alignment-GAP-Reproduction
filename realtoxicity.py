import os
import time
import json
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===========================================================
# CONFIG
# ===========================================================
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./qwen_dolly_cpu_finetune"

OUTPUT_CSV = "toxicity_rtp_benchmark.csv"
JSON_DIR = "toxicity_rtp_json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES_PER_PROMPT = 25
GEN_TEMPERATURE = 0.7
GEN_MAX_NEW_TOKENS = 50

os.makedirs(JSON_DIR, exist_ok=True)

# ===========================================================
# LOAD MODELS
# ===========================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

print("Loading baseline model...")
baseline = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map=DEVICE,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
baseline.eval()

print("Loading LoRA model...")
lora_model = PeftModel.from_pretrained(baseline, ADAPTER_PATH)
lora_model.eval()


# ===========================================================
# TOXICITY MODEL (LOCAL)
# ===========================================================
print("Loading local toxicity model (unitary/toxic-bert)...")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

tox_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
tox_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
tox_model.to(DEVICE)
tox_model.eval()

TOXIC_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

def score_toxicity(text):
    """Return toxicity scores using a local model."""
    inputs = tox_tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        logits = tox_model(**inputs).logits
        probs = torch.sigmoid(logits)[0].tolist()

    return {label: float(score) for label, score in zip(TOXIC_LABELS, probs)}


# ===========================================================
# GENERATION FUNCTION
# ===========================================================
def generate(model, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=0.9
        )

    gen_ids = out[:, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# ===========================================================
# LOAD REAL TOXICITY PROMPTS
# ===========================================================
dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
prompts = dataset["prompt"]["text"]
print(f"Loaded {len(prompts)} RealToxicityPrompts prompts")


# ===========================================================
# SETUP OUTPUT CSV
# ===========================================================
if not os.path.exists(OUTPUT_CSV):
    df = pd.DataFrame(columns=[
        "prompt",
        "model_type",
        "avg_toxicity",
        "max_toxicity"
    ])
    df.to_csv(OUTPUT_CSV, index=False)


# ===========================================================
# MAIN LOOP
# ===========================================================
for idx, prompt in enumerate(tqdm(prompts)):

    for model_name, model in [("baseline", baseline), ("lora", lora_model)]:

        sample_scores = []
        sample_json = []

        for s in range(NUM_SAMPLES_PER_PROMPT):

            answer = generate(model, prompt)
            scores = score_toxicity(answer)

            # use "toxic" as main toxicity score
            sample_scores.append(scores["toxic"])
            sample_json.append({
                "sample_id": s,
                "answer": answer,
                "scores": scores
            })

            time.sleep(0.05)  # small sleep for safety

        avg_tox = sum(sample_scores) / len(sample_scores)
        max_tox = max(sample_scores)

        # --------------- SAVE JSON ---------------
        out_json_path = os.path.join(JSON_DIR, f"rtp_{idx:05d}_{model_name}.json")
        with open(out_json_path, "w") as fp:
            json.dump({
                "prompt": prompt,
                "samples": sample_json,
                "avg_toxicity": avg_tox,
                "max_toxicity": max_tox,
            }, fp, indent=4)

        # --------------- SAVE CSV ---------------
        row = pd.DataFrame([{
            "prompt": prompt,
            "model_type": model_name,
            "avg_toxicity": avg_tox,
            "max_toxicity": max_tox,
        }])
        row.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)


print("\nDone!")
print("CSV saved to:", OUTPUT_CSV)
print("JSON samples saved to:", JSON_DIR)
