import os
import pandas as pd
from configuration import ProjectConfig as CFG
from data_loader import get_common_file_ids

CFG.make_dirs()

ANNOTATIONS_FILE = os.path.join(CFG.DATA_DIR, "annotations.csv")
OUTPUT_FILE = os.path.join(CFG.SUMMARY_DIR, "fidelity_annotation_check.csv")

# Load the annotations CSV
annotations_df = pd.read_csv(ANNOTATIONS_FILE)

# Load common ids (i.e. the ids for which we have features and labels)
common_ids = set(get_common_file_ids(CFG.FEATURES_DIR, CFG.LABELS_DIR))

# Filter annotations to those that match our usable dataset
filtered_df = annotations_df[annotations_df['filename'].str.replace(".mp3", "").isin(common_ids)]

# Group by file and sort by longest annotations
grouped = (filtered_df.groupby("filename")
            .agg({"text": lambda x: list(x),
                  "categories": lambda x: list(x),
                  "onset": lambda x: list(x),
                  "offset": lambda x: list(x)})
            .reset_index())

# Unpack the list of categories into individual top-level labels

def extract_flat_labels(cat_list):
    # Use eval to handle the stringified list (like "['Alarm']")
    all_labels = []
    for entry in cat_list:
        try:
            all_labels.extend(eval(entry))
        except:
            continue
    return list(set(all_labels))

grouped["mapped_labels"] = grouped["categories"].apply(extract_flat_labels)

# Add column for manual rating (initially empty)
grouped["fidelity_rating"] = ""  # e.g. Good / Acceptable / Poor
grouped["fidelity_comment"] = ""  # optional free-form comment

# Save to CSV
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
grouped.to_csv(OUTPUT_FILE, index=False)

print(f"Saved fidelity inspection template to: {OUTPUT_FILE}")