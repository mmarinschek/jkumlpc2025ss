import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from configuration import ProjectConfig as CFG
import os

csv_path = CFG.SUMMARY_DIR+"/model_evaluation_summary.csv"
df = pd.read_csv(csv_path)

# Preprocess
df["Best Epoch"] = df["Best Epoch"].fillna(0)
df["Model Type"] = df["Model Name"].apply(lambda x: x.split('_')[0])
df["Feature Type"] = df["Model Name"].str.extract(r'(mfcc|melspectrogram|embeddings)', expand=False)
df["Params"] = df["Model Name"].str.replace(r'^.*?_', '', regex=True)

#Model Name,Best Epoch,Train ROC-AUC Micro,Train Balanced Accuracy,Val ROC-AUC Micro,Val Balanced Accuracy,Gap ROC-AUC Micro,Gap Balanced Accuracy


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
        row["Gap Balanced Accuracy"],
        row["Val Balanced Accuracy"],
        marker=marker_styles.get(row["Feature Type"], "o"),
        color=model_color_map.get(row["Model Type"], "gray"),
        s=120,
        edgecolor="black",
        linewidth=0.7
    )
    plt.text(
        row["Gap Balanced Accuracy"] + 0.001,
        row["Val Balanced Accuracy"] + 0.001,
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
plt.title("Model Performance: Gap Balanced Accuracy vs. Val Balanced Accuracy", fontsize=14)
plt.xlabel("Gap Balanced Accuracy", fontsize=12)
plt.ylabel("Val Balanced Accuracy", fontsize=12)
plt.grid(True)
plt.tight_layout()

# Show or save
plt.show()
# plt.savefig("model_evaluation_scatter_plot.png", dpi=300)