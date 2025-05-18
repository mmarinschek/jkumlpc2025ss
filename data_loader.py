import os
import numpy as np
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


def load_data(features_dir, labels_dir, common_ids, class_names, feature_key):
    all_features = []
    all_labels = []

    for file_id in tqdm(common_ids, desc="Loading file data for feature key : " + feature_key):
        features_npz = np.load(os.path.join(features_dir, f"{file_id}.npz"))
        if feature_key not in features_npz:
            raise KeyError(f"Feature key '{feature_key}' not found in '{file_id}.npz'.")
        features = features_npz[feature_key]  # Shape: (time_steps, feature_dim)
        num_steps = features.shape[0]

        labels_npz = np.load(os.path.join(labels_dir, f"{file_id}_labels.npz"))

        # Determine max number of annotators across classes
        max_annotators = max(labels_npz[cls].shape[1] for cls in class_names if cls in labels_npz)

        for annotator_idx in range(max_annotators):
            label_matrix = []
            for cls in class_names:
                if cls in labels_npz:
                    cls_labels = labels_npz[cls]
                    if cls_labels.shape[0] != num_steps:
                        raise ValueError(f"Mismatch in time steps for class '{cls}' in file '{file_id}'.")

                    # If this annotator exists for the class, use their labels
                    if annotator_idx < cls_labels.shape[1]:
                        annotator_labels = cls_labels[:, annotator_idx]
                        binary_labels = (annotator_labels > 0).astype(int)
                    else:
                        # Annotator didn't provide data for this class; assume negative
                        binary_labels = np.zeros(num_steps, dtype=int)
                else:
                    raise KeyError(f"Class '{cls}' not found in labels for file '{file_id}'.")
                
                label_matrix.append(binary_labels)

            label_matrix = np.stack(label_matrix, axis=-1)

            # Add this annotator's perspective to the dataset
            all_features.append(features)
            all_labels.append(label_matrix)

    X = np.vstack(all_features)  # Shape: (total_frames * annotators, feature_dim)
    Y = np.vstack(all_labels)    # Shape: (total_frames * annotators, num_classes)

    return X, Y