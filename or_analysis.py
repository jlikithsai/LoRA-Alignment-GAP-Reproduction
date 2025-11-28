import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv('or-bench-benchmark.csv')

# Pivot the table so each prompt has two columns: baseline & lora refusal_rate
pivot = df.pivot_table(
    index="prompt",
    columns="model_type",
    values="refusal_rate"
).dropna()   # remove prompts that don't have BOTH baseline & lora rows

# Extract matched X and Y values
x = pivot["baseline"]
y = pivot["lora"]

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=50)

# Reference line
max_val = max(x.max(), y.max())
plt.plot([0, max_val], [0, max_val], 'r--')

plt.xlabel("Baseline Refusal Rate")
plt.ylabel("LoRA Refusal Rate")
plt.title("Per-Prompt Refusal Spread: Baseline vs LoRA")
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
