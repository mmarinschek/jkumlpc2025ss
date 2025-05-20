import os
from sklearn.model_selection import train_test_split
from data_loader import load_class_names, get_common_file_ids, load_data
from models import ModelConfig, ModelResult, save_model, train_multi_label_model
from evaluation import evaluate_model, ModelEvaluationTracker
from visualization import visualize_audio_features_labels
from utils import format_clickable_path
import pandas as pd
from tqdm import tqdm
from configuration import ProjectConfig as CFG

CFG.make_dirs()

class HyperparameterSearchSpace:
    def __init__(self):
        self.search_space = []
        for config in ModelConfig:
            model_class = config.value
            for params in model_class.hyperparameter_grid():
                for feature_key in CFG.FEATURE_KEYS:
                    self.search_space.append((config, params, feature_key))

    def __len__(self):
        return len(self.search_space)

    def __iter__(self):
        return iter(self.search_space)

def main():
    class_names = load_class_names(CFG.LABELS_DIR)
    print(f"Detected {len(class_names)} classes.")

    common_ids = get_common_file_ids(CFG.FEATURES_DIR, CFG.LABELS_DIR)
    
    if CFG.FAST_MODE:
        common_ids = common_ids[:CFG.MAX_FILES]
        print(f"Fast mode enabled: Using only {len(common_ids)} files.")
        
    train_ids, temp_ids = train_test_split(common_ids, test_size=0.3, random_state=CFG.RANDOM_SEED)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=CFG.RANDOM_SEED)

    feature_datasets = {}
    for feature_key in CFG.FEATURE_KEYS:
        print(f"Loading feature set: {feature_key}")

        X_train, Y_train = load_data(CFG.FEATURES_DIR, CFG.LABELS_DIR, train_ids, class_names, feature_key)
        X_val, Y_val = load_data(CFG.FEATURES_DIR, CFG.LABELS_DIR, val_ids, class_names, feature_key)
        X_test, Y_test = load_data(CFG.FEATURES_DIR, CFG.LABELS_DIR, test_ids, class_names, feature_key)

        feature_datasets[feature_key] = {
            "train": (X_train, Y_train),
            "val": (X_val, Y_val),
            "test": (X_test, Y_test)
        }

        print(f"Train Samples shape: {X_train.shape[0]}, Val samples shape: {X_val.shape[0]}, Test samples shape: {X_test.shape[0]}")
    
    #visualize_audio_features_labels(AUDIO_DIR, FEATURES_DIR, LABELS_DIR, common_ids, class_names, feature_key=DESIRED_FEATURE_KEY, n=1, top_k=10)

    run_hyper_param_search(class_names, feature_datasets, CFG.REVERSE)


def run_hyper_param_search(class_names, feature_datasets, reverse):
    tracker = ModelEvaluationTracker(CFG.OUTPUT_DIR, CFG.SUMMARY_DIR)
    full_search_space = HyperparameterSearchSpace()

    filtered_search_space = [
        (config, params, feature_key) 
        for config, params, feature_key in full_search_space 
        if not tracker.already_evaluated(ModelResult.derive_model_name(config, params, feature_key))
    ]
    
    if reverse:
        filtered_search_space = list(reversed(filtered_search_space))

    for config, params, feature_key in tqdm(filtered_search_space, desc="Hyperparameter Search Progress", total=len(filtered_search_space)):
        model_name = ModelResult.derive_model_name(config, params, feature_key)
        print(f"\nTraining {model_name}")
        
        X_train, Y_train = feature_datasets[feature_key]["train"]
        X_val, Y_val     = feature_datasets[feature_key]["val"]
        X_test, Y_test   = feature_datasets[feature_key]["test"]

        trained_model_result : ModelResult = train_multi_label_model(X_train, Y_train, X_val, Y_val, config, params, feature_key)
        save_model(trained_model_result, CFG.OUTPUT_DIR)
        eval_result = evaluate_model(trained_model_result, X_val, Y_val, class_names, CFG.OUTPUT_DIR)
        eval_result.best_epoch = trained_model_result.best_epoch #if the model is epoch enabled, provide the best epoch here.
        tracker.add_result(eval_result)
if __name__ == "__main__":
    main()