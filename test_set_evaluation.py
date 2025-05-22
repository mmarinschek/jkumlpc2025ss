import os
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from configuration import ProjectConfig as CFG
from data_loader import load_class_names, get_common_file_ids
from classify import load_data_in_splits, create_splits, DataSplitConfig
from jkumlpc2025ss.models import extract_feature_key
from models import load_model
from evaluation import evaluate_model


def main():
    CFG.make_dirs()

    print("Loading class names...")
    class_names = load_class_names(CFG.LABELS_DIR)
    print(f"{len(class_names)} classes found.")

    print("Finding file IDs...")
    common_ids = get_common_file_ids(CFG.FEATURES_DIR, CFG.LABELS_DIR)

    if CFG.FAST_MODE:
        common_ids = common_ids[:CFG.MAX_FILES]
        print(f"Fast mode enabled: evaluating {len(common_ids)} files only.")

    splits : List[DataSplitConfig] = create_splits(common_ids)

    test_split = next((s for s in splits if s.name == "test"), None)

    feature_datasets = load_data_in_splits(class_names, splits)

    # Load model summaries
    summary_path = os.path.join(CFG.SUMMARY_DIR, "best_model_per_model_type_evaluation_summary.csv")
    if not os.path.exists(summary_path):
        print(f"No evaluation summary found at {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)
    print(f"Loaded {len(summary_df)} evaluated models.")

    results = []

    for idx, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Evaluating on Test Set"):
        model_name = row["Model Name"]
        feature_key = extract_feature_key(model_name, CFG.FEATURE_KEY_SETS.keys())

        resolved_keys = CFG.resolve_feature_keys(feature_key)
        print(f"Resolved simple feature keys for model training: {resolved_keys}")

        X_train = np.concatenate([feature_datasets[k]["train"][0] for k in resolved_keys], axis=1)
        Y_train = feature_datasets[resolved_keys[0]]["train"][1]

        X_val = np.concatenate([feature_datasets[k]["val"][0] for k in resolved_keys], axis=1)
        Y_val = feature_datasets[resolved_keys[0]]["val"][1]

        X_test = np.concatenate([feature_datasets[k]["test"][0] for k in resolved_keys], axis=1)
        Y_test = feature_datasets[resolved_keys[0]]["test"][1]

        print (f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}, X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}, X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

        print(f"\nEvaluating {model_name} on test set ({feature_key})")
        try:
            model_result = load_model(CFG.OUTPUT_DIR, model_name)
            eval_result = evaluate_model(model_result, X_test, Y_test, class_names, CFG.OUTPUT_DIR)
            eval_dict = eval_result.to_dict()
            eval_dict["Model Name"] = model_name
            results.append(eval_dict)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

        # Save to CSV
        out_path = os.path.join(CFG.SUMMARY_DIR, "test_set_evaluation_summary.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"\nTest set results written to: {out_path}")

if __name__ == "__main__":
    main()