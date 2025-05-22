import os
import pandas as pd
from models import ModelConfig
from configuration import ProjectConfig as CFG

def extract_model_type(model_name: str) -> str:
    for model_type in ModelConfig.__members__:
        if model_name.startswith(model_type):
            return model_type
    return None  # or raise ValueError if preferred

def select_best_models(input_csv, output_csv, metric="Val Balanced Accuracy"):
    df = pd.read_csv(input_csv)

    df["Model Type"] = df["Model Name"].apply(extract_model_type)
    df = df[df["Model Type"].notna()]

    # Select the best entry per model type (highest metric value)
    best_df = df.loc[df.groupby("Model Type")[metric].idxmax()]

    best_df = best_df.sort_values(by="Model Type")
    best_df.to_csv(output_csv, index=False)
    print(f"Saved best models per type to: {output_csv}")

# --- Run it ---
input_csv = os.path.join(CFG.SUMMARY_DIR, "model_evaluation_summary.csv")
output_csv = os.path.join(CFG.SUMMARY_DIR, "best_model_per_model_type_evaluation_summary.csv")
select_best_models(input_csv, output_csv)