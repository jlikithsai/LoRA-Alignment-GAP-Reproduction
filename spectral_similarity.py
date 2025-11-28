#!/usr/bin/env python3
"""
Compute Spectral Similarity and Spectral Intruder Count (SIC)
between:
  - Baseline model
  - LoRA-merged model

Works for Qwen2.5-0.5B-Instruct or any HF causal LM.
"""

import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from peft import PeftModel
from torch.linalg import svd
from tqdm import tqdm

# ===========================================================
# CONFIG
# ===========================================================
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_ADAPTER = "./qwen_dolly_cpu_final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

TOP_K = 20                   # number of singular vectors to compare
INTRUDER_THRESHOLD = 0.75   # cosine < threshold → intruder
OUTPUT_CSV = "spectral_similarity_results.csv"

# ===========================================================
# LOAD MODELS
# ===========================================================
print("Loading baseline model...")
baseline = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, device_map=DEVICE, dtype=DTYPE)
baseline.eval()

print("Loading LoRA merged model...")
lora = PeftModel.from_pretrained(baseline, LORA_ADAPTER)
lora = lora.merge_and_unload()   # merge adapter into weights
lora.eval()


# ===========================================================
# UTILITIES
# ===========================================================
def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def compute_topk_svd(M, k=TOP_K):
    """Return top-k singular vectors U (left) and V (right)."""
    U, S, Vh = svd(M, full_matrices=False)
    return U[:, :k], Vh[:k, :]


def spectral_intruder_count(U_base, U_lora, threshold=INTRUDER_THRESHOLD):
    """
    Count how many U_lora[:,i] have cosine similarity < threshold
    w.r.t U_base[:,i].
    """
    count = 0
    for i in range(U_base.shape[1]):
        sim = cosine_sim(U_base[:, i], U_lora[:, i])
        if sim < threshold:
            count += 1
    return count


# ===========================================================
# MAIN EXTRACTION & EVALUATION
# ===========================================================
results = []

for name, param in tqdm(baseline.named_parameters(), desc="Spectral Analysis"):

    if param.dim() != 2:
        continue  # only linear layers

    # Skip embedding / LM head
    if "embed" in name.lower() or "lm_head" in name:
        continue

    W_base = param.detach().to(torch.float32).cpu()
    W_lora = lora.state_dict()[name].detach().to(torch.float32).cpu()

    # ----------------- SVD -----------------
    U_b, V_b = compute_topk_svd(W_base)
    U_l, V_l = compute_topk_svd(W_lora)

    # ----------------- Similarities -----------------
    U_sims = [cosine_sim(U_b[:, i], U_l[:, i]) for i in range(TOP_K)]
    V_sims = [cosine_sim(V_b[i, :], V_l[i, :]) for i in range(TOP_K)]

    avg_U = sum(U_sims) / TOP_K
    avg_V = sum(V_sims) / TOP_K

    # ----------------- Spectral Intruder Count -----------------
    intruders = spectral_intruder_count(U_b, U_l, threshold=INTRUDER_THRESHOLD)

    # ----------------- Save row -----------------
    results.append({
        "layer": name,
        "avg_U_similarity": avg_U,
        "avg_V_similarity": avg_V,
        "min_U_similarity": min(U_sims),
        "max_U_similarity": max(U_sims),
        "spectral_intruder_count": intruders,
    })


# ===========================================================
# SAVE CSV
# ===========================================================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\nDone! Saved:", OUTPUT_CSV)
