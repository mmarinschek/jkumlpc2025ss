import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
from matplotlib.animation import FuncAnimation
import time

def select_samples_with_most_labels(labels_dir, common_ids, top_n=10):
    label_counts = []
    for idx in common_ids:
        labels_path = os.path.join(labels_dir, f"{idx}_labels.npz")
        if not os.path.exists(labels_path):
            continue

        labels_npz = np.load(labels_path)
        label_matrix = np.stack([labels_npz[cls].mean(axis=-1) for cls in labels_npz.keys()], axis=-1)
        label_binary = (label_matrix > 0).astype(int)

        # Count number of classes that are active at least once
        active_labels = (label_binary.sum(axis=0) > 0).sum()
        label_counts.append((idx, active_labels))

    # Sort descending by number of active labels and return top N IDs
    sorted_by_label_diversity = sorted(label_counts, key=lambda x: -x[1])
    return [idx for idx, _ in sorted_by_label_diversity[:top_n]]


def visualize_audio_features_labels(audio_dir, features_dir, labels_dir, common_ids, class_names, feature_key="mfcc", n=10, top_k=10, annotations_df=None):

    for idx in common_ids[:n]:
        frame_duration_sec = 0.120
        audio_path = os.path.join(audio_dir, f"{idx}.mp3")
        features_path = os.path.join(features_dir, f"{idx}.npz")
        labels_path = os.path.join(labels_dir, f"{idx}_labels.npz")
                
        print(f"Processing ID: {idx} - Audio: {audio_path}, Features: {features_path}, Labels: {labels_path}")

        if not all(os.path.exists(p) for p in [audio_path, features_path, labels_path]):
            print(f"Skipping ID {idx}: missing audio, features, or labels.")
            continue
        
        def on_close(event):
            sd.stop()

        # Load Audio
        y, sr = librosa.load(audio_path, sr=None)
        duration_sec = len(y) / sr
        
        # Load Features
        features_npz = np.load(features_path)
        if feature_key not in features_npz:
            print(f"Skipping ID {idx}: feature key '{feature_key}' not found.")
            continue
        features = features_npz[feature_key]
        time_steps = features.shape[0]
        
        # Load Labels
        labels_npz = np.load(labels_path)
                
        label_matrix = np.stack([labels_npz[cls].mean(axis=-1) for cls in class_names], axis=-1)
        label_binary = (label_matrix > 0).astype(int)

        # Top-K Classes
        class_activity = label_binary.sum(axis=0)
        active_classes_idx = np.where(class_activity > 0)[0]

        if len(active_classes_idx) == 0:
            print(f"No active classes found for sample ID {idx}. Skipping visualization.")
            continue

        # Sort by activity and select top_k
        sorted_active_idx = active_classes_idx[np.argsort(-class_activity[active_classes_idx])]
        top_classes_idx = sorted_active_idx[:top_k]
        top_classes = [class_names[i] for i in top_classes_idx]

        time_axis_audio = np.linspace(0, duration_sec, len(y))
        time_axis_features = np.arange(time_steps) * frame_duration_sec

        fig, axs = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
        
        fig.canvas.mpl_connect('close_event', on_close)

        # Static plots
        axs[0].plot(time_axis_audio, y, color="gray")
        axs[0].set_title(f"Audio Waveform - Sample ID: {idx}")
        axs[0].set_ylabel("Amplitude")

        axs[1].plot(time_axis_features, np.mean(features, axis=1), color="blue")
        axs[1].set_title(f"Feature Timeline ({feature_key})")
        axs[1].set_ylabel("Mean Feature Value")

        axs[2].imshow(
            label_binary[:, top_classes_idx].T,
            aspect='auto',
            cmap='Blues',
            extent=[0, duration_sec, 0, len(top_classes)],
            vmin=0,
            vmax=1
        )
        axs[2].set_yticks(np.arange(len(top_classes)) + 0.5)
        axs[2].set_yticklabels(top_classes)
        axs[2].set_title("Ground Truth Labels Over Time")
        axs[2].set_xlabel("Time (seconds)")

        # 🔸 Add annotation spans if available
        if annotations_df is not None:
            row = annotations_df[annotations_df["file_id"] == int(idx)]
            if not row.empty:
                try:
                    onsets = eval(row.iloc[0]["onset"])
                    offsets = eval(row.iloc[0]["offset"])
                    categories = eval(row.iloc[0]["categories"])
                    for start, end, cat_str in zip(onsets, offsets, categories):
                        cat_list = eval(cat_str)
                        label = cat_list[0] if isinstance(cat_list, list) and cat_list else cat_str
                        axs[2].axvspan(start, end, color="orange", alpha=0.3)
                        axs[2].text((start + end) / 2, len(top_classes) + 0.5, label, ha="center", fontsize=8, rotation=45)
                except Exception as e:
                    print(f"Annotation overlay failed for {idx}: {e}")

        # Animated cursor
        cursor_lines = [ax.axvline(0, color="red", linestyle="--") for ax in axs]
        
        def start_playback_with_cursor_sync():
            playback_start_time = time.time()

            def update_cursor(frame_idx):
                elapsed_time = time.time() - playback_start_time
                current_time = min(elapsed_time, duration_sec)

                for line in cursor_lines:
                    line.set_xdata([current_time])

                if current_time >= duration_sec and ani.event_source:
                    ani.event_source.stop()

                return cursor_lines

            return update_cursor
        
        update_cursor = start_playback_with_cursor_sync()

        ani = FuncAnimation(
            fig,
            update_cursor,
            frames=np.arange(time_steps),
            interval=frame_duration_sec * 1000, #Convert to milliseconds
            blit=True
        )
    
        sd.play(y, sr)
        plt.show()
        sd.wait()  # Ensure audio finishes before continuing