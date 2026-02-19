import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("outputs/metrics/baseline_results.csv")
aihrm = pd.read_csv("outputs/metrics/aihrm_results.csv")

combined = pd.concat([baseline, aihrm], ignore_index=True)

plt.figure(figsize=(8,5))
bars = plt.bar(combined["Model"], combined["F1"])

plt.title("F1 Score Comparison of Models")
plt.ylabel("F1 Score")
plt.xticks(rotation=30)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01,
             round(yval,3), ha='center')

plt.tight_layout()
plt.savefig("outputs/plots/f1_comparison.png", dpi=300)
plt.show()
