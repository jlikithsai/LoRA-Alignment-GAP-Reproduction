import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ===========================================================
# CONFIG
# ===========================================================
FINETUNED_CSV = "spectral_similarity_ft.csv"
LORA_CSV = "spectral_similarity_results.csv"

OUTPUT_CSV = "comparative_ft_lora.csv"
os.makedirs("plots_compare", exist_ok=True)

# ===========================================================
# LOAD DATA
# ===========================================================
print("[INFO] Loading finetuned and LoRA CSVs...")

df_ft = pd.read_csv(FINETUNED_CSV)
df_lora = pd.read_csv(LORA_CSV)

# Extract layer ID and module
for df in [df_ft, df_lora]:
    df["layer_id"] = df["layer"].str.extract(r"layers\.(\d+)\.").astype(int)
    df["module"] = df["layer"].str.extract(
        r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    )[0]
    df["drift_U"] = 1 - df["avg_U_similarity"]
    df["drift_V"] = 1 - df["avg_V_similarity"]
    df["spectral_drift"] = (
        df["drift_U"] + df["drift_V"] + df["spectral_intruder_count"]
    )

df_ft["model"] = "finetuned"
df_lora["model"] = "lora"

# ===========================================================
# MERGE FT AND LORA ON LAYER + MODULE
# ===========================================================
df = df_ft.merge(df_lora, on=["layer", "layer_id", "module"], suffixes=("_ft", "_lora"))

print("[INFO] Merge complete!")

# ===========================================================
# COMPUTE DIFFERENCES
# ===========================================================
df["delta_ft_lora"] = df["spectral_drift_ft"] - df["spectral_drift_lora"]

df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Comparative CSV saved to {OUTPUT_CSV}")

# ===========================================================
# PLOT 1 — Drift Comparison Per Layer
# ===========================================================
plt.figure(figsize=(16,6))
plt.plot(df["layer_id"], df["spectral_drift_ft"], label="Finetuned")
plt.plot(df["layer_id"], df["spectral_drift_lora"], label="Lora")
plt.title("Finetuned vs LoRA — Spectral Drift")
plt.xlabel("Layer")
plt.ylabel("Total Drift")
plt.legend()
plt.grid(True)
plt.savefig("plots_compare/1_drift_ft_vs_lora.png", dpi=200)
plt.close()

# ===========================================================
# PLOT 2 — Difference Plot (FT - LORA)
# ===========================================================
plt.figure(figsize=(16,6))
plt.plot(df["layer_id"], df["delta_ft_lora"], marker="o")
plt.title("Difference in Spectral Drift Between Finetuned and LoRA")
plt.xlabel("Layer")
plt.ylabel("Drift Difference (FT - LoRA)")
plt.grid(True)
plt.savefig("plots_compare/2_delta_ft_lora.png", dpi=200)
plt.close()

# ===========================================================
# PLOT 3 — Module-Level Heatmap
# ===========================================================
pivot = df.pivot(index="layer_id", columns="module", values="delta_ft_lora")

plt.figure(figsize=(14,10))
sns.heatmap(pivot, cmap="coolwarm", center=0)
plt.title("Heatmap: Finetuned vs LoRA Drift Difference")
plt.xlabel("Module")
plt.ylabel("Layer ID")
plt.savefig("plots_compare/3_heatmap_module_differences.png", dpi=200)
plt.close()

# ===========================================================
# PRINT TOP DIFFERENCES
# ===========================================================
print("\n========== TOP 20 DIFFERENCES BETWEEN FINETUNED AND LORA ==========")
print(df.sort_values("delta_ft_lora", ascending=False).head(20))

print("\n[COMPLETE] Comparative FT vs LoRA spectral analysis is done!")
print("Plots saved under: ./plots_compare/")
