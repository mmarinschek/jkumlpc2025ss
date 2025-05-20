# test_set_evaluation.py
import os
import pandas as pd
from tqdm import tqdm

from configuration import ProjectConfig as CFG
from data_loader import load_class_names, get_common_file_ids, load_data
from models import ModelResult, load_model
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split

def extract_feature_key(model_name: str, valid_keys) -> str:
    """
    Extracts the feature key from the model name by matching against known valid feature keys.
    Assumes feature key is the suffix after the last underscore.
    """
    for key in valid_keys:
        if model_name.endswith(f"_{key}"):
            return key
    raise ValueError(f"Could not extract valid feature key from model name '{model_name}'")

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

    train_ids, temp_ids = train_test_split(common_ids, test_size=0.3, random_state=CFG.RANDOM_SEED)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=CFG.RANDOM_SEED)

    feature_datasets = {}
    for feature_key in CFG.FEATURE_KEYS:
        print(f"Loading test data for feature: {feature_key}")
        X_test, Y_test = load_data(CFG.FEATURES_DIR, CFG.LABELS_DIR, test_ids, class_names, feature_key)
        feature_datasets[feature_key] = (X_test, Y_test)

    # Load model summaries
    summary_path = os.path.join(CFG.SUMMARY_DIR, "model_evaluation_summary.csv")
    if not os.path.exists(summary_path):
        print(f"No evaluation summary found at {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)
    print(f"Loaded {len(summary_df)} evaluated models.")

    for idx, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Evaluating on Test Set"):
        model_name = row["Model Name"]
        feature_key = extract_feature_key(model_name, CFG.FEATURE_KEYS)

        print(f"\nEvaluating {model_name} on test set ({feature_key})")
        try:
            model_result = load_model(CFG.OUTPUT_DIR, model_name)
            X_test, Y_test = feature_datasets[feature_key]
            eval_result = evaluate_model(model_result, X_test, Y_test, class_names, CFG.OUTPUT_DIR)
            print(eval_result.to_dict())
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

if __name__ == "__main__":
    main()