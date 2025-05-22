import os
import pandas as pd
from configuration import ProjectConfig as CFG
from data_loader import load_class_names, get_common_file_ids
from visualization import visualize_audio_features_labels  # adjust the import as needed

# Paths
annotations_path = os.path.join(CFG.SUMMARY_DIR, "fidelity_annotation_check.csv")
annotations_df = pd.read_csv(annotations_path)

# Load class names
class_names = load_class_names(CFG.LABELS_DIR)

# Load annotations file with selected file_ids
file_ids = annotations_df["file_id"].astype(str).tolist()

# Filter only available files (intersection with common_ids)
common_ids = set(get_common_file_ids(CFG.FEATURES_DIR, CFG.LABELS_DIR))
valid_ids = [fid for fid in file_ids if fid in common_ids]

# Visualize one by one
visualize_audio_features_labels(
    audio_dir=CFG.AUDIO_DIR,
    features_dir=CFG.FEATURES_DIR,
    labels_dir=CFG.LABELS_DIR,
    common_ids=annotations_df["file_id"].astype(str).tolist(),
    class_names=load_class_names(CFG.LABELS_DIR),
    feature_key="mfcc",
    annotations_df=annotations_df
)