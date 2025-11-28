import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===========================================================
# CONFIG
# ===========================================================
CSV_PATH = "toxicity_rtp_benchmark.csv"   # <-- your CSV filename
os.makedirs("toxicity_plots", exist_ok=True)

# ===========================================================
# LOAD CSV
# ===========================================================
df = pd.read_csv(CSV_PATH)

print("\n[INFO] Loaded CSV with", len(df), "rows")

# ===========================================================
# COMPUTE AVERAGES
# ===========================================================
avg_stats = df.groupby("model_type")[["avg_toxicity", "max_toxicity"]].mean()

print("\n===== AVERAGE TOXICITY (BASELINE vs LORA) =====\n")
print(avg_stats)

# ===========================================================
# PLOT 1 — Average Toxicity Bar Plot
# ===========================================================
plt.figure(figsize=(8,5))
avg_stats["avg_toxicity"].plot(kind="bar", color=["steelblue","orange"])
plt.title("Average Toxicity: Finetune vs LoRA")
plt.ylabel("Average Toxicity Score")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.savefig("toxicity_plots/avg_toxicity_bar.png", dpi=200)
plt.close()

# ===========================================================
# PLOT 2 — Per-Prompt Baseline vs LoRA Toxicity Scatter
# ===========================================================
# pivot for easy comparison
pivot = df.pivot(index="prompt", columns="model_type", values="avg_toxicity")

plt.figure(figsize=(7,7))
sns.scatterplot(
    x=pivot["baseline"], 
    y=pivot["lora"], 
    alpha=0.7
)
plt.plot([0, max(pivot.max())], [0, max(pivot.max())], ls="--", color="red")
plt.xlabel("Finetune toxicity")
plt.ylabel("LoRA toxicity")
plt.title("Per-Prompt Toxicity: LoRA vs Finetune")
plt.grid(True)
plt.savefig("toxicity_plots/toxicity_scatter.png", dpi=200)
plt.close()

# ===========================================================
# PLOT 3 — Difference Plot (LoRA - Baseline)
# ===========================================================
pivot["toxicity_diff"] = pivot["lora"] - pivot["baseline"]

plt.figure(figsize=(16,5))
plt.plot(pivot.index, pivot["toxicity_diff"], marker="o")
plt.xticks([], [])  # hide prompt names (too long)
plt.axhline(0, color="red", linestyle="--")
plt.title("Toxicity Difference per Prompt (LoRA - Finetune)")
plt.ylabel("Difference in toxicity")
plt.grid(True)
plt.savefig("toxicity_plots/toxicity_diff.png", dpi=200)
plt.close()

print("\n[INFO] Plots saved in ./toxicity_plots/")
print("[COMPLETE] Toxicity analysis done.")
