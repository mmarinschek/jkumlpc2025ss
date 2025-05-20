import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Adjust path if needed
csv_path = "jkumlpc2025ss/model_evaluation_summary/model_evaluation_summary.csv"
df = pd.read_csv(csv_path)

# Preprocess
df["Best Epoch"] = df["Best Epoch"].fillna(0)
df["Model Type"] = df["Model Name"].apply(lambda x: x.split('_')[0])
df["Feature Type"] = df["Model Name"].str.extract(r'(mfcc|melspectrogram|embeddings)', expand=False)
df["Params"] = df["Model Name"].str.replace(r'^.*?_', '', regex=True)

# Marker styles for feature types
marker_styles = {
    "mfcc": "o",
    "melspectrogram": "s",
    "embeddings": "D"
}

# Unique colors per model type
colors = sns.color_palette("tab10", n_colors=df["Model Type"].nunique())
model_color_map = {model: color for model, color in zip(df["Model Type"].unique(), colors)}

# Plot setup
plt.figure(figsize=(14, 10))

# Plot each point
for _, row in df.iterrows():
    plt.scatter(
        row["ROC-AUC Micro"],
        row["Balanced Accuracy"],
        marker=marker_styles.get(row["Feature Type"], "o"),
        color=model_color_map.get(row["Model Type"], "gray"),
        s=120,
        edgecolor="black",
        linewidth=0.7
    )
    plt.text(
        row["ROC-AUC Micro"] + 0.001,
        row["Balanced Accuracy"] + 0.001,
        row["Params"],
        fontsize=8,
        alpha=0.7
    )

# Legend setup
legend_elements = [
    plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', label=ftype,
               markersize=10, markeredgecolor='black')
    for ftype, m in marker_styles.items()
] + [
    plt.Line2D([0], [0], marker='o', color='w', label=model,
               markerfacecolor=color, markersize=10)
    for model, color in model_color_map.items()
]
plt.legend(handles=legend_elements, title="Feature Type / Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')

# Labels and layout
plt.title("Model Performance: ROC-AUC Micro vs. Balanced Accuracy", fontsize=14)
plt.xlabel("ROC-AUC Micro", fontsize=12)
plt.ylabel("Balanced Accuracy", fontsize=12)
plt.grid(True)
plt.tight_layout()

# Show or save
plt.show()
# plt.savefig("model_evaluation_scatter_plot.png", dpi=300)