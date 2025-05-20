import os

class ProjectConfig:
    FAST_MODE = False
    MAX_FILES = 10
    REVERSE = False
    RANDOM_SEED = 42
    FEATURE_KEYS = ["mfcc", "melspectrogram", "embeddings"]

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