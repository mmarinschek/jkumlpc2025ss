import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from configuration import ProjectConfig as CFG
from data_loader import load_class_names, get_common_file_ids, load_all_features_and_labels

def compute_stats(values, eps=1e-2):
    mean = float(np.mean(values))
    std = float(np.std(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))

    is_zscore = abs(mean) < eps and abs(std - 1.0) < eps
    is_minmax = 0.0 - eps <= min_val <= 0.0 + eps and 1.0 - eps <= max_val <= 1.0 + eps

    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "zscore_normalized": is_zscore,
        "minmax_normalized": is_minmax
    }

def analyze_feature_normalization(result, output_csv_path):
    records = []
    for feature_key, feature_data in result["features"].items():
        for dim in tqdm(range(feature_data.shape[1]), desc=f"Analyzing {feature_key}"):
            values = feature_data[:, dim]
            stats = compute_stats(values)
            records.append({
                "feature_dim": f"{feature_key}_{dim}",
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "zscore_normalized": stats["zscore_normalized"],
                "minmax_normalized": stats["minmax_normalized"]
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Normalization analysis saved to: {output_csv_path}")

def main():
    class_names = load_class_names(CFG.LABELS_DIR)
    common_ids = get_common_file_ids(CFG.FEATURES_DIR, CFG.LABELS_DIR)
    simple_keys = CFG.resolve_all_simple_feature_keys()

    result = load_all_features_and_labels(
        CFG.FEATURES_DIR, CFG.LABELS_DIR, common_ids, class_names, required_keys=simple_keys
    )

    output_csv = os.path.join(CFG.SUMMARY_DIR, "feature_normalization_analysis.csv")
    os.makedirs(CFG.SUMMARY_DIR, exist_ok=True)
    analyze_feature_normalization(result, output_csv)

if __name__ == "__main__":
    main()