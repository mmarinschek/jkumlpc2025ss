import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

from configuration import ProjectConfig as CFG
from data_loader import (
    load_class_names,
    get_common_file_ids,
    load_all_features_and_labels,
    ScalerProvider,
)
from models import load_model, extract_feature_key
from jkumlpc2025ss.classify import create_splits, DataSplitConfig
from sklearn.metrics import roc_auc_score

def evaluate_file(X, Y, model):
    try:
        y_prob = model.predict_proba(X)
        if y_prob.shape[1] == 1:
            y_prob = np.hstack([1 - y_prob, y_prob])
        roc_auc = roc_auc_score(Y, y_prob, average="micro")
        return roc_auc
    except Exception:
        return np.nan

def main():
    CFG.make_dirs()
    print("Loading class names...")
    class_names = load_class_names(CFG.LABELS_DIR)

    print("Loading common file IDs...")
    common_ids = get_common_file_ids(CFG.FEATURES_DIR, CFG.LABELS_DIR)
    if CFG.FAST_MODE:
        common_ids = common_ids[:CFG.MAX_FILES]

    print("Creating data splits...")
    splits: List[DataSplitConfig] = create_splits(common_ids)
    test_split = next(s for s in splits if s.name == "test")
    train_split = next(s for s in splits if s.name == "train")

    # Load best model summary
    summary_path = os.path.join(CFG.SUMMARY_DIR, "best_model_evaluation_summary.csv")
    summary_df = pd.read_csv(summary_path)
    print(f"Loaded {len(summary_df)} models from summary.")

    per_file_results = []

    for _, row in summary_df.iterrows():
        model_name = row["Model Name"]
        feature_key = extract_feature_key(model_name, CFG.FEATURE_KEY_SETS.keys())
        resolved_keys = CFG.resolve_feature_keys(feature_key)

        print(f"\nEvaluating model: {model_name}")
        print(f"Using resolved features: {resolved_keys}")

        # Prepare scaler on training split
        scaler = ScalerProvider(learn=True)
        _ = load_all_features_and_labels(
            CFG.FEATURES_DIR, CFG.LABELS_DIR, train_split.file_ids, class_names, resolved_keys, scaler
        )

        # Load model
        model = load_model(CFG.OUTPUT_DIR, model_name)

        # Evaluate on test split per file
        for file_id in tqdm(test_split.file_ids, desc=f"Per-file eval for {model_name}"):
            data = load_all_features_and_labels(
                CFG.FEATURES_DIR, CFG.LABELS_DIR, [file_id], class_names, resolved_keys, scaler
            )
            X = np.concatenate([data["features"][k] for k in resolved_keys], axis=1)
            Y = data["labels"]

            roc_auc = evaluate_file(X, Y, model)
            per_file_results.append({
                "file_id": file_id,
                "model_name": model_name,
                "roc_auc_micro": roc_auc
            })

    # Create result frame
    results_df = pd.DataFrame(per_file_results)
    output_path = os.path.join(CFG.SUMMARY_DIR, "per_file_test_ranking.csv")
    print(f"Saving full results to {output_path}")
    results_df.to_csv(output_path, index=False)

    # Save top/bottom per model
    best_and_worst = []

    for model in results_df["model_name"].unique():
        model_df = results_df[results_df["model_name"] == model].sort_values(by="roc_auc_micro", ascending=False)
        best_and_worst.extend(model_df.head(5).to_dict(orient="records"))
        best_and_worst.extend(model_df.tail(5).to_dict(orient="records"))

    bw_df = pd.DataFrame(best_and_worst)
    bw_path = os.path.join(CFG.SUMMARY_DIR, "per_file_test_best_and_worst.csv")
    print(f"Saving best/worst files per model to {bw_path}")
    bw_df.to_csv(bw_path, index=False)

if __name__ == "__main__":
    main()