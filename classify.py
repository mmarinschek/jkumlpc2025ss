import os
from sklearn.model_selection import train_test_split
from data_loader import load_class_names, get_common_file_ids, load_data
from models import ModelConfig, ModelResult, save_model, train_multi_label_model
from evaluation import evaluate_model, ModelEvaluationTracker
from visualization import visualize_audio_features_labels
from utils import format_clickable_path
import pandas as pd
from tqdm import tqdm


FAST_MODE = False
REVERSE = False
MAX_FILES = 10

# Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))         # /path/to/jkumlpc2025ss
BASE_DIR = os.path.dirname(CURRENT_DIR)                          # /path/to/

DATA_DIR = os.path.join(BASE_DIR, "MLPC2025_classification")
FEATURES_DIR = os.path.join(DATA_DIR, "audio_features")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models" if not FAST_MODE else "trained_models_fast")
SUMMARY_DIR = os.path.join(CURRENT_DIR, "model_evaluation_summary" if not FAST_MODE else "model_evaluation_summary_fast")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

FEATURE_KEYS = ["mfcc", "melspectrogram", "embeddings"]

class HyperparameterSearchSpace:
    def __init__(self):
        self.search_space = []
        for config in ModelConfig:
            model_class = config.value
            for params in model_class.hyperparameter_grid():
                for feature_key in FEATURE_KEYS:
                    self.search_space.append((config, params, feature_key))

    def __len__(self):
        return len(self.search_space)

    def __iter__(self):
        return iter(self.search_space)

def main():
    class_names = load_class_names(LABELS_DIR)
    print(f"Detected {len(class_names)} classes.")

    common_ids = get_common_file_ids(FEATURES_DIR, LABELS_DIR)
    if FAST_MODE:
        common_ids = common_ids[:MAX_FILES]
        print(f"Fast mode enabled: Using only {len(common_ids)} files.")

    feature_datasets = {}
    for feature_key in FEATURE_KEYS:
        X, Y = load_data(FEATURES_DIR, LABELS_DIR, common_ids, class_names, feature_key)
        feature_datasets[feature_key] = (X, Y)
        print(f"Feature key : {feature_key}, ")
        print(f"Total Samples: {X.shape[0]}, Feature Dim: {X.shape[1]}, Classes: {Y.shape[1]}")
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    
    #visualize_audio_features_labels(AUDIO_DIR, FEATURES_DIR, LABELS_DIR, common_ids, class_names, feature_key=DESIRED_FEATURE_KEY, n=1, top_k=10)

    run_hyper_param_search(class_names, feature_datasets, REVERSE)


def run_hyper_param_search(class_names, feature_datasets, reverse):
    tracker = ModelEvaluationTracker(OUTPUT_DIR, SUMMARY_DIR)
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
        
        X = feature_datasets[feature_key][0]
        Y = feature_datasets[feature_key][1]
        
        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
        X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

        trained_model_result : ModelResult = train_multi_label_model(X_train, Y_train, X_val, Y_val, config, params, feature_key)
        save_model(trained_model_result, OUTPUT_DIR)
        eval_result = evaluate_model(trained_model_result, X_val, Y_val, class_names, OUTPUT_DIR)
        eval_result.best_epoch = trained_model_result.best_epoch #if the model is epoch enabled, provide the best epoch here.
        tracker.add_result(eval_result)
if __name__ == "__main__":
    main()