import os
import pandas as pd
from configuration import ProjectConfig as CFG
from data_loader import get_common_file_ids

# Paths
ANNOTATIONS_FILE = os.path.join(CFG.DATA_DIR, "annotations.csv")
OUTPUT_FILE = os.path.join(CFG.SUMMARY_DIR, "fidelity_annotation_check.csv")
os.makedirs(CFG.SUMMARY_DIR, exist_ok=True)

# Load data
annotations_df = pd.read_csv(ANNOTATIONS_FILE)
annotations_df["file_id"] = annotations_df["filename"].str.replace(".mp3", "", regex=False)

# Filter to only common file ids
common_ids = set(get_common_file_ids(CFG.FEATURES_DIR, CFG.LABELS_DIR))
annotations_df = annotations_df[annotations_df["file_id"].isin(common_ids)]

# Select one row per file, randomly
sampled_ids = annotations_df["file_id"].drop_duplicates().sample(n=30, random_state=CFG.RANDOM_SEED)
sampled_df = annotations_df[annotations_df["file_id"].isin(sampled_ids)]

# Group annotations per file
grouped = (
    sampled_df.groupby("file_id")
    .agg({
        "text": list,
        "categories": list,
        "onset": list,
        "offset": list
    })
    .reset_index()
)

def extract_flat_labels(cat_list):
    flat = []
    for entry in cat_list:
        try:
            flat.extend(eval(entry))  # Categories are stored as stringified lists
        except Exception:
            continue
    return list(set(flat))

grouped["mapped_labels"] = grouped["categories"].apply(extract_flat_labels)
grouped["fidelity_rating"] = ""
grouped["fidelity_comment"] = ""

grouped.to_csv(OUTPUT_FILE, index=False)
print(f"Fidelity inspection template saved to: {OUTPUT_FILE}")