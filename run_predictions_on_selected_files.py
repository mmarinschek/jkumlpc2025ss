# run_prediction_on_selected_files.py

import os
import pandas as pd
import numpy as np
from jkumlpc2025ss.configuration import ProjectConfig as CFG
from jkumlpc2025ss.visualization import AudioLabelSegment
from jkumlpc2025ss.models import load_model
from jkumlpc2025ss.data_loader import load_class_names, load_all_features_and_labels, ScalerProvider
from jkumlpc2025ss.classify import create_splits, DataSplitConfig

PREDICTION_INPUT_PATH = os.path.join(CFG.SUMMARY_DIR, "per_file_test_best_and_worst.csv")
PREDICTION_OUTPUT_PATH = os.path.join(CFG.SUMMARY_DIR, "file_level_predictions.csv")

def extract_segments_from_predictions(predictions: np.ndarray, class_names: list[str], frame_duration_sec: float = 0.120) -> list[AudioLabelSegment]:
    """
    Converts binary frame-wise predictions into temporal segments.

    Args:
        predictions: numpy array of shape (time_steps, num_classes)
        class_names: list of class names corresponding to prediction columns
        frame_duration_sec: duration per frame in seconds

    Returns:
        List of AudioLabelSegment instances.
    """
    segments = []
    num_classes = predictions.shape[1]

    for class_idx in range(num_classes):
        label = class_names[class_idx]
        sequence = predictions[:, class_idx]
        in_segment = False
        start_idx = None

        for i, val in enumerate(sequence):
            if val > 0 and not in_segment:
                start_idx = i
                in_segment = True
            elif val == 0 and in_segment:
                end_idx = i
                segments.append(AudioLabelSegment(
                    class_name=label,
                    annotator_index=0,  # single predictor
                    onset_sec=start_idx * frame_duration_sec,
                    offset_sec=end_idx * frame_duration_sec
                ))
                in_segment = False

        if in_segment:
            segments.append(AudioLabelSegment(
                class_name=label,
                annotator_index=0,
                onset_sec=start_idx * frame_duration_sec,
                offset_sec=len(sequence) * frame_duration_sec
            ))

    return segments

def run_predictions():
    df = pd.read_csv(PREDICTION_INPUT_PATH)
    class_names = load_class_names(CFG.LABELS_DIR)

    predictions = []

    for model_name in df["model_name"].unique():
        model = load_model(CFG.OUTPUT_DIR, model_name)
        feature_key = model_name.split("_")[-1]
        resolved_keys = CFG.resolve_feature_keys(feature_key)

        # Prepare scaler based on training split
        all_ids = [f for f in os.listdir(CFG.LABELS_DIR) if f.endswith(".npz")]
        all_ids = sorted(set(f.split("_")[0] for f in all_ids))
        splits: list[DataSplitConfig] = create_splits(all_ids)
        train_split = next(s for s in splits if s.name == "train")
        scaler = ScalerProvider(learn=True)
        _ = load_all_features_and_labels(CFG.FEATURES_DIR, CFG.LABELS_DIR, train_split.file_ids, class_names, resolved_keys, scaler)

        # Evaluate per file
        for _, row in df[df["model_name"] == model_name].iterrows():
            file_id = str(row["file_id"])
            result = load_all_features_and_labels(CFG.FEATURES_DIR, CFG.LABELS_DIR, [file_id], class_names, resolved_keys, scaler)

            X = np.concatenate([result["features"][k] for k in resolved_keys], axis=1)
            Y = result["labels"]

            Y_prob = model.predict_proba(X)
            Y_pred = (Y_prob > 0.5).astype(int)

            segments = extract_segments_from_predictions(Y_pred, class_names, frame_duration_sec=0.120)

            for seg in segments:
                predictions.append({
                    "file_id": file_id,
                    "model_name": model_name,
                    "class": seg.class_name,
                    "onset_sec": seg.onset_sec,
                    "offset_sec": seg.offset_sec
                })

    pd.DataFrame(predictions).to_csv(PREDICTION_OUTPUT_PATH, index=False)
    print(f"Saved predictions to {PREDICTION_OUTPUT_PATH}")


if __name__ == "__main__":
    run_predictions()
