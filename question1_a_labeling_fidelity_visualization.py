import os

import pandas as pd

from configuration import ProjectConfig as CFG
from visualization import visualize_audio_features_labels

ANNOTATION_CSV = os.path.join(CFG.SUMMARY_DIR, "fidelity_annotation_check.csv")
FEATURE_KEY = "mfcc"

# Load annotation file
annotations_df = pd.read_csv(ANNOTATION_CSV)
annotations_df["file_id"] = annotations_df["file_id"].astype(str)
annotations_df["fidelity_rating"] = annotations_df["fidelity_rating"].astype(str)
annotations_df["fidelity_comment"] = annotations_df["fidelity_comment"].astype(str)
annotations_df["file_id"] = annotations_df["file_id"].astype(str)

def is_unrated(val):
    return pd.isna(val) or str(val).strip().lower() in {"", "nan"}

remaining_df = annotations_df[annotations_df["fidelity_rating"].apply(is_unrated)]

print(f"{len(remaining_df)} entries remaining to rate.")

for i, row in remaining_df.iterrows():
    file_id = row["file_id"]

    print(f"\n--- ({i+1}/{len(annotations_df)}) Visualizing: {file_id} ---")

    visualize_audio_features_labels(common_ids=[file_id],feature_key=FEATURE_KEY)

    # Collect user feedback
    rating = input("Enter fidelity rating (Good/Acceptable/Poor or 'q' to quit): ").strip()

    if rating.lower() in {"q", "quit"}:
        print("Exiting fidelity review.")
        break

    comment = input("Optional comment: ").strip()

    # Update the DataFrame
    annotations_df.at[i, "fidelity_rating"] = rating
    annotations_df.at[i, "fidelity_comment"] = comment

    # Save intermediate result
    annotations_df.to_csv(ANNOTATION_CSV, index=False)
    print(f"Saved rating for {file_id}.\n")