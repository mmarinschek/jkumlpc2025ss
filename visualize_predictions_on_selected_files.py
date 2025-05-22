import os
import pandas as pd
from typing import Dict, List

from jkumlpc2025ss.configuration import ProjectConfig as CFG
from jkumlpc2025ss.visualization import AudioLabelSegment, visualize_audio_features_labels
from jkumlpc2025ss.data_loader import load_class_names

PREDICTION_CSV = os.path.join(CFG.SUMMARY_DIR, "file_level_predictions.csv")


def load_prediction_segments_by_file(csv_path: str) -> Dict[str, List[AudioLabelSegment]]:
    df = pd.read_csv(csv_path)
    grouped = {}

    for _, row in df.iterrows():
        file_id = str(row["file_id"])
        segment = AudioLabelSegment(
            class_name=row["class"],
            annotator_index=-1,
            onset_sec=row["onset_sec"],
            offset_sec=row["offset_sec"]
        )
        grouped.setdefault(file_id, []).append(segment)

    return grouped


def main():
    print("Loading predictions from:", PREDICTION_CSV)
    prediction_segments = load_prediction_segments_by_file(PREDICTION_CSV)
    class_names = load_class_names(CFG.LABELS_DIR)

    for file_id, segments in prediction_segments.items():
        print(f"Visualizing predictions for file {file_id}...")
        visualize_audio_features_labels(
            common_ids=[file_id],
            feature_key="mfcc",  # Adjust to match the model's feature set
            prediction_segments=segments
        )


if __name__ == "__main__":
    main()