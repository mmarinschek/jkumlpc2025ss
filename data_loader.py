import os
from typing import List
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm

def load_class_names(labels_dir):
    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith('.npz'))
    if not label_files:
        raise FileNotFoundError("No label files found in 'labels' directory.")
    sample_labels = np.load(os.path.join(labels_dir, label_files[0]))
    return list(sample_labels.keys())


def get_common_file_ids(features_dir, labels_dir):
    feature_ids = {f.split('.')[0] for f in os.listdir(features_dir) if f.endswith('.npz')}
    label_ids = {f.split('_')[0] for f in os.listdir(labels_dir) if f.endswith('.npz')}
    common_ids = sorted(feature_ids & label_ids)
    if not common_ids:
        raise FileNotFoundError("No matching feature and label files found.")
    return common_ids

class ScalerProvider:
    def __init__(self, learn=True):
        self.learn = learn
        self.scalers = {}

    def transform(self, key, data):
        if key not in self.scalers:
            if not self.learn:
                raise ValueError(f"No scaler available for feature '{key}' in inference mode.")
            scaler = StandardScaler()
            scaler.fit(data)
            self.scalers[key] = scaler
        return self.scalers[key].transform(data)

    def get_scalers(self):
        return self.scalers


def load_all_features_and_labels(features_dir, labels_dir, common_ids, class_names, required_keys: List[str], scaler_provider: ScalerProvider):
    all_labels = []
    all_feature_data = {k: [] for k in required_keys}

    for file_id in tqdm(common_ids, desc="Loading all requested features and labels once"):
        features_npz = np.load(os.path.join(features_dir, f"{file_id}.npz"))
        labels_npz = np.load(os.path.join(labels_dir, f"{file_id}_labels.npz"))

        # Validate keys
        for key in required_keys:
            if key not in features_npz:
                raise KeyError(f"Feature key '{key}' missing in file '{file_id}.npz'")

        num_steps = features_npz[required_keys[0]].shape[0]  # Assume all features aligned

        max_annotators = max(labels_npz[cls].shape[1] for cls in class_names if cls in labels_npz)

        for annotator_idx in range(max_annotators):
            label_matrix = []
            for cls in class_names:
                if cls not in labels_npz:
                    raise KeyError(f"Label class '{cls}' missing in file '{file_id}'")
                cls_labels = labels_npz[cls]
                if cls_labels.shape[0] != num_steps:
                    raise ValueError(f"Time mismatch for '{cls}' in '{file_id}'")

                if annotator_idx < cls_labels.shape[1]:
                    binary = (cls_labels[:, annotator_idx] > 0).astype(int)
                else:
                    binary = np.zeros(num_steps, dtype=int)

                label_matrix.append(binary)
            label_matrix = np.stack(label_matrix, axis=-1)
            all_labels.append(label_matrix)

            # Also collect all requested features for this annotator
            for key in required_keys:
                all_feature_data[key].append(features_npz[key])

    Y = np.vstack(all_labels)

    feature_dict = {}
    for key in required_keys:
        raw = np.vstack(all_feature_data[key])
        normalized = scaler_provider.transform(key, raw)
        feature_dict[key] = normalized

    return {"labels": Y, "features": feature_dict}