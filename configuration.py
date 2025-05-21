import os
from typing import List

class ProjectConfig:
    FAST_MODE = False
    MAX_FILES = 10
    REVERSE = False
    RANDOM_SEED = 42
    
    FEATURE_KEY_SETS = {
        "embeddings": ["embeddings"]
        # "mfcc": ["mfcc"],
        # "melspectrogram": ["melspectrogram"],
        # "mfcc_with_deltas": ["mfcc", "mfcc_delta", "mfcc_delta2"],
        # "baseaudio": ["flatness", "centroid", "flux", "energy", "power", "bandwidth", "contrast", "zerocrossingrate"],
        # "embeddings_plus_baseaudio": ["embeddings", "baseaudio"],
        # "embeddings_plus_mfcc_with_deltas": ["embeddings", "mfcc_with_deltas"],
        # "embeddings_plus_melspectrogram": ["embeddings", "melspectrogram"],
        # "embeddings_plus_mfcc_with_deltas_plus_melspectrogram": ["embeddings", "mfcc_with_deltas", "melspectrogram"],
        # "embeddings_plus_mfcc_with_deltas_plus_melspectrogram_plus_baseaudio": ["embeddings", "mfcc_with_deltas", "melspectrogram", "baseaudio"],
    }
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR)

    DATA_DIR = os.path.join(BASE_DIR, "MLPC2025_classification")
    FEATURES_DIR = os.path.join(DATA_DIR, "audio_features")
    LABELS_DIR = os.path.join(DATA_DIR, "labels")
    AUDIO_DIR = os.path.join(DATA_DIR, "audio")

    OUTPUT_DIR = os.path.join(BASE_DIR, "trained_models" if not FAST_MODE else "trained_models_fast")
    SUMMARY_DIR = os.path.join(CURRENT_DIR, "model_evaluation_summary" if not FAST_MODE else "model_evaluation_summary_fast")

    @classmethod
    def make_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUMMARY_DIR, exist_ok=True)
        
    @classmethod
    def resolve_feature_keys(cls, key: str) -> List[str]:
        resolved = set()

        def _resolve(k):
            # Either not defined (atomic), or defined as identity → atomic
            if k not in cls.FEATURE_KEY_SETS or cls.FEATURE_KEY_SETS[k] == [k]:
                resolved.add(k)
            else:
                for sub_k in cls.FEATURE_KEY_SETS[k]:
                    _resolve(sub_k)

        _resolve(key)
        return sorted(resolved)
    
    @classmethod
    def resolve_all_simple_feature_keys(cls):
        resolved = set()
        for key in cls.FEATURE_KEY_SETS:
            resolved.update(cls.resolve_feature_keys(key))
        return sorted(resolved)