import os
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import ScalerProvider, load_all_features_and_labels, load_class_names, get_common_file_ids
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
                for feature_key_set_name in CFG.FEATURE_KEY_SETS.keys():
                    self.search_space.append((config, params, feature_key_set_name))

    def __len__(self):
        return len(self.search_space)

    def __iter__(self):
        return iter(self.search_space)
    
    def filter_by_model_type(self, model_config: ModelConfig):
        filtered = HyperparameterSearchSpace()
        filtered.search_space = [
            (cfg, params, fk) for cfg, params, fk in self.search_space
            if cfg == model_config
        ]
        return filtered

    def filter_by_params(self, **criteria):
        def match(params):
            return all(getattr(params, k, None) == v for k, v in criteria.items())

        filtered = HyperparameterSearchSpace()
        filtered.search_space = [
            (cfg, params, fk) for cfg, params, fk in self.search_space
            if match(params)
        ]
        return filtered    
from enum import Enum
class DataSplitConfig:
    def __init__(self, name: str, file_ids: List[str], fit_scaler: bool):
        self.name = name
        self.file_ids = file_ids
        self.fit_scaler = fit_scaler
        
def main():
    class_names = load_class_names(CFG.LABELS_DIR)
    print(f"Detected {len(class_names)} classes.")

    common_ids = get_common_file_ids(CFG.FEATURES_DIR, CFG.LABELS_DIR)
    
    if CFG.FAST_MODE:
        common_ids = common_ids[:CFG.MAX_FILES]
        print(f"Fast mode enabled: Using only {len(common_ids)} files.")
        
    train_ids, temp_ids = train_test_split(common_ids, test_size=0.3, random_state=CFG.RANDOM_SEED)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=CFG.RANDOM_SEED)

    splits = [
        DataSplitConfig("train", train_ids, fit_scaler=True),
        DataSplitConfig("val", val_ids, fit_scaler=False),
        DataSplitConfig("test", test_ids, fit_scaler=False),
    ]

    simple_feature_keys = CFG.resolve_all_simple_feature_keys()
    
    print(f"Detected {len(simple_feature_keys)} simple feature keys: {simple_feature_keys}")

    feature_datasets = {}

    for feature_key in simple_feature_keys:
        feature_datasets[feature_key] = {}
        
    scaler = ScalerProvider(learn=True)

    for split in splits:
        print(f"\nLoading split: {split.name}, #files: {len(split.file_ids)}")
        
        scaler.learn = split.fit_scaler
        
        result = load_all_features_and_labels(
            CFG.FEATURES_DIR, 
            CFG.LABELS_DIR, 
            split.file_ids, 
            class_names, 
            required_keys=simple_feature_keys, 
            scaler_provider=scaler
        )

        for feature_key in simple_feature_keys:
            X = result["features"][feature_key]
            Y = result["labels"]
            feature_datasets[feature_key][split.name] = (X, Y)
            print(f"Feature: {feature_key} | Split: {split.name} | Samples: {X.shape[0]} | Dim: {X.shape[1]}")
    
    #visualize_audio_features_labels(AUDIO_DIR, FEATURES_DIR, LABELS_DIR, common_ids, class_names, feature_key=DESIRED_FEATURE_KEY, n=1, top_k=10)

    run_hyper_param_search(class_names, feature_datasets, CFG.REVERSE)


def run_hyper_param_search(class_names, feature_datasets, reverse):
    tracker = ModelEvaluationTracker(CFG.OUTPUT_DIR, CFG.SUMMARY_DIR)
    full_search_space = HyperparameterSearchSpace().filter_by_model_type(ModelConfig.RANDOM_FOREST).filter_by_params(n_estimators=50, max_depth=5)

    filtered_search_space = [
        (config, params, feature_key) 
        for config, params, feature_key in full_search_space 
        if not tracker.already_evaluated(ModelResult.derive_model_name(config, params, feature_key))
    ]
    
    if reverse:
        filtered_search_space = list(reversed(filtered_search_space))

    for config, params, feature_key in tqdm(filtered_search_space, desc="Hyperparameter Search Progress", total=len(filtered_search_space)):
        model_name = ModelResult.derive_model_name(config, params, feature_key)
        print(f"\nTraining {model_name} for features {feature_key} with {config.name} and params: {params}")
        
        resolved_keys = CFG.resolve_feature_keys(feature_key)
        
        print(f"Resolved simple feature keys for model training: {resolved_keys}")

        X_train = np.concatenate([feature_datasets[k]["train"][0] for k in resolved_keys], axis=1)
        Y_train = feature_datasets[resolved_keys[0]]["train"][1]

        X_val = np.concatenate([feature_datasets[k]["val"][0] for k in resolved_keys], axis=1)
        Y_val = feature_datasets[resolved_keys[0]]["val"][1]

        X_test = np.concatenate([feature_datasets[k]["test"][0] for k in resolved_keys], axis=1)
        Y_test = feature_datasets[resolved_keys[0]]["test"][1]
        
        print (f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}, X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}, X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

        trained_model_result : ModelResult = train_multi_label_model(X_train, Y_train, X_val, Y_val, config, params, feature_key)
        save_model(trained_model_result, CFG.OUTPUT_DIR)
        eval_result = evaluate_model(trained_model_result, X_val, Y_val, class_names, CFG.OUTPUT_DIR)
        eval_result.best_epoch = trained_model_result.best_epoch #if the model is epoch enabled, provide the best epoch here.
        tracker.add_result(eval_result)
if __name__ == "__main__":
    main()