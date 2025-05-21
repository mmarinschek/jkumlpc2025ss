import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from data_loader import load_all_features_and_labels, load_class_names, get_common_file_ids
from configuration import ProjectConfig as CFG
from utils import format_clickable_path
import umap
from sklearn.feature_selection import mutual_info_classif

CFG.make_dirs()

def subsample_data(X, Y, max_samples=3000, seed=42):
    np.random.seed(seed)
    idx = np.random.choice(X.shape[0], size=min(max_samples, X.shape[0]), replace=False)
    return X[idx], Y[idx]

def load_class_group_colors(csv_path, class_names):
    grouping_df = pd.read_csv(csv_path)
    grouping_df = grouping_df.set_index('class_name').loc[class_names]

    super_groups = grouping_df['super_group'].unique()
    super_palette = sns.color_palette("Set2", len(super_groups)) 
    super_group_to_color = {sg: super_palette[i] for i, sg in enumerate(super_groups)}

    group_offsets = {
        group: 0.7 + 0.6 * (i / (len(grouping_df['group'].unique()) - 1))
        for i, group in enumerate(sorted(grouping_df['group'].unique()))
    }

    class_colors = []
    for _, row in grouping_df.iterrows():
        base_color = np.array(super_group_to_color[row['super_group']])
        offset = group_offsets[row['group']]
        adjusted = np.clip(base_color * offset, 0, 1)
        class_colors.append(tuple(adjusted))

    return class_colors

def save_projections(X, Y, output_dir):
    projections = {}
    print("Running PCA...")
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    projections['pca'] = pca_2d
    np.save(os.path.join(output_dir, "X_pca.npy"), pca_2d)

    X_sub, Y_sub = subsample_data(X, Y, max_samples=3000)
    np.save(os.path.join(output_dir, "X_sub.npy"), X_sub)
    np.save(os.path.join(output_dir, "Y_sub.npy"), Y_sub)    
    
    print("Running t-SNE...")
    tsne = TSNE( n_components=2, perplexity=30, max_iter=1000, init="pca", verbose=1, random_state=42)
    X_tsne = tsne.fit_transform(X_sub)
    projections['tsne'] = X_tsne
    np.save(os.path.join(output_dir, "X_tsne.npy"), X_tsne)

    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_jobs=10, verbose=True)
    X_umap = reducer.fit_transform(X_sub)
    projections['umap'] = X_umap
    np.save(os.path.join(output_dir, "X_umap.npy"), X_umap)
    
    return projections

def plot_projection(X_2d, Y_single, class_names, class_colors, method_name, output_dir):
    plt.figure(figsize=(10, 8))
    for class_idx in np.unique(Y_single):
        mask = Y_single == class_idx
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    label=class_names[class_idx],
                    s=15, alpha=0.7,
                    color=class_colors[class_idx])
    plt.title(f"{method_name} Projection of Embeddings Colored by Group/Super-Group")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plot_path = os.path.join(output_dir, f"embedding_clusters_by_group_{method_name.lower()}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"{method_name} plot saved to {format_clickable_path(plot_path)}")

def visualize_saved_projections(Y_full, class_names, feature_key, output_dir):
    group_csv_path = os.path.join(CFG.CURRENT_DIR, "class_grouping_table.csv")
    class_colors = load_class_group_colors(group_csv_path, class_names)

    projection_sources = {
        "pca": {
            "X_path": os.path.join(output_dir, "X_pca.npy"),
            "Y_source": "full",
        },
        "tsne": {
            "X_path": os.path.join(output_dir, "X_tsne.npy"),
            "Y_path": os.path.join(output_dir, "Y_sub.npy"),
            "Y_source": "sub",
        },
        "umap": {
            "X_path": os.path.join(output_dir, "X_umap.npy"),
            "Y_path": os.path.join(output_dir, "Y_sub.npy"),
            "Y_source": "sub",
        },
    }

    for method, config in projection_sources.items():
        if not os.path.exists(config["X_path"]):
            print(f"Projection for {method.upper()} not found.")
            continue

        X_proj = np.load(config["X_path"])

        if config["Y_source"] == "full":
            Y_single = np.argmax(Y_full, axis=1)
        elif config["Y_source"] == "sub":
            if not os.path.exists(config["Y_path"]):
                print(f"Subsampled Y for {method.upper()} not found.")
                continue
            Y_sub = np.load(config["Y_path"])
            Y_single = np.argmax(Y_sub, axis=1)

        plot_projection(X_proj, Y_single, class_names, class_colors, method.upper(), output_dir)

def compute_label_alignment(X, Y, feature_key):
    Y_single = np.argmax(Y, axis=1)
    
    if X.shape[1] > 20 :
        X_reduced = PCA(n_components=20, random_state=42).fit_transform(X)
    else:
        X_reduced = X
        
    kmeans = KMeans(n_clusters=Y.shape[1], n_init=10, random_state=42)
    cluster_assignments = kmeans.fit_predict(X_reduced)

    nmi = normalized_mutual_info_score(Y_single, cluster_assignments)
    ari = adjusted_rand_score(Y_single, cluster_assignments)
    
    return {"feature_key": feature_key, "NMI": nmi, "ARI": ari}

def main():
    class_names = load_class_names(CFG.LABELS_DIR)
    common_ids = get_common_file_ids(CFG.FEATURES_DIR, CFG.LABELS_DIR)

    set_feature_keys = CFG.FEATURE_KEY_SETS.keys()
    simple_feature_keys = CFG.resolve_all_simple_feature_keys()

    print(f"Detected set feature keys: {set_feature_keys}")
    
    print(f"\nLoading all features and labels...")
    result = load_all_features_and_labels(CFG.FEATURES_DIR, CFG.LABELS_DIR, common_ids, class_names, required_keys=simple_feature_keys)
    
    summary_path = os.path.join(CFG.SUMMARY_DIR, "feature_analysis_summary.csv")
    os.makedirs(CFG.SUMMARY_DIR, exist_ok=True)
    
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = pd.DataFrame(columns=["feature_key", "NMI", "ARI"])

    for set_feature_key in set_feature_keys:
        print(f"\nProcessing set feature key: {set_feature_key}")
        
        if set_feature_key in summary_df["feature_key"].values:
            print(f"Skipping {set_feature_key}, already processed.")
            continue
        
        resolved_simple_keys = CFG.resolve_feature_keys(set_feature_key)
        
        print(f"Resolved simple keys for set : {set_feature_key} -  {resolved_simple_keys}")
        
        try:
            X = np.concatenate([result["features"][k] for k in resolved_simple_keys], axis=1)
            Y = result["labels"]
            print(f"X shape: {X.shape}, Y shape: {Y.shape}")

            output_dir = os.path.join(CFG.SUMMARY_DIR, f"feature_{set_feature_key}")
            os.makedirs(output_dir, exist_ok=True)

            print(f"Compute label alignment metrics...")
            label_metrics = compute_label_alignment(X, Y, set_feature_key)
            print(f"Label alignment metrics for {set_feature_key}: NMI: {label_metrics['NMI']}, ARI: {label_metrics['ARI']}")
            if set_feature_key in summary_df["feature_key"].values:
                summary_df.loc[summary_df["feature_key"] == set_feature_key, ["NMI", "ARI"]] = label_metrics["NMI"], label_metrics["ARI"]
            else:
                summary_df.loc[len(summary_df)] = {
                    "feature_key": set_feature_key,
                    "NMI": label_metrics["NMI"],
                    "ARI": label_metrics["ARI"]
                }            
            summary_df.to_csv(summary_path, index=False)

            #save_projections(X, Y, output_dir)
            #visualize_saved_projections(Y, class_names, set_feature_key, output_dir)
        except Exception as e:
            print(f"Skipping {set_feature_key} due to error: {e}")

if __name__ == "__main__":
    main()