import os
import pandas as pd
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
from matplotlib.animation import FuncAnimation
import time

from data_loader import load_class_names
from configuration import ProjectConfig as CFG

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

class AudioLabelSegment:
    def __init__(self, class_name, annotator_index, onset_sec, offset_sec):
        self.class_name = class_name
        self.annotator_index = annotator_index
        self.onset_sec = onset_sec
        self.offset_sec = offset_sec

    def __repr__(self):
        return (f"AudioLabelSegment(class={self.class_name}, "
                f"annotator={self.annotator_index}, "
                f"onset={self.onset_sec:.2f}s, offset={self.offset_sec:.2f}s)")

def extract_segments_per_annotator(labels_npz_path, class_names, frame_duration_sec=0.120):
    """
    Extracts active segments (onset, offset) for each class and annotator.

    Returns:
        List of AudioLabelSegment objects.
    """
    labels_npz = np.load(labels_npz_path)
    segments = []

    for cls in class_names:
        if cls not in labels_npz:
            continue

        label_matrix = labels_npz[cls]  # Shape: [time_steps, annotators]

        for annotator_idx in range(label_matrix.shape[1]):
            label_sequence = label_matrix[:, annotator_idx]
            in_segment = False
            start_idx = None

            for i, val in enumerate(label_sequence):
                if val > 0 and not in_segment:
                    start_idx = i
                    in_segment = True
                elif val == 0 and in_segment:
                    end_idx = i
                    segments.append(AudioLabelSegment(
                        class_name=cls,
                        annotator_index=annotator_idx,
                        onset_sec=start_idx * frame_duration_sec,
                        offset_sec=end_idx * frame_duration_sec
                    ))
                    in_segment = False

            if in_segment:
                segments.append(AudioLabelSegment(
                    class_name=cls,
                    annotator_index=annotator_idx,
                    onset_sec=start_idx * frame_duration_sec,
                    offset_sec=len(label_sequence) * frame_duration_sec
                ))

    return segments

def plot_segments_panel(ax, segments: List[AudioLabelSegment], duration_sec: float, title: str):
    """
    Plots only the used classes from the segments, sorted alphabetically.

    Args:
        ax: Matplotlib axis object.
        segments: List of AudioLabelSegment instances.
        duration_sec: Total duration of the audio.
        title: Title of the subplot.
    """
    # Determine class set from segments
    used_classes = sorted(set(seg.class_name for seg in segments))

    ax.set_title(title)
    ax.set_xlim(0, duration_sec)
    ax.set_ylim(0, len(used_classes))
    ax.set_yticks(np.arange(len(used_classes)) + 0.5)
    ax.set_yticklabels(used_classes)

    for seg in segments:
        y_idx = used_classes.index(seg.class_name)
        ax.axvspan(seg.onset_sec, seg.offset_sec,
                   ymin=(y_idx / len(used_classes)),
                   ymax=((y_idx + 1) / len(used_classes)),
                   color='orange' if 'annotation' in title.lower() else 'steelblue',
                   alpha=0.5)
        label = seg.class_name
        if seg.annotator_index is not None:
            label += f" (A{seg.annotator_index})"

def extract_annotation_segments_from_csv(annotations_csv_path: str, file_id: str) -> List[AudioLabelSegment]:
    """
    Extracts annotation segments from the original annotations.csv for a given file ID.

    Args:
        annotations_csv_path: Path to annotations.csv.
        file_id: ID of the audio file (string or int).

    Returns:
        List of AudioLabelSegment instances representing annotated regions.
    """
    df = pd.read_csv(annotations_csv_path)

    if "filename" not in df.columns or "text" not in df.columns or "onset" not in df.columns or "offset" not in df.columns:
        raise ValueError("Required columns ['filename', 'text', 'onset', 'offset'] not found.")

    df["filename_clean"] = df["filename"].str.replace(".mp3", "", regex=False)
    file_rows = df[df["filename_clean"] == str(file_id)]

    if file_rows.empty:
        return []

    segments = []
    for _, row in file_rows.iterrows():        
        class_name = row["text"].strip() if isinstance(row["text"], str) else "Unknown"
        onset = float(row["onset"])
        offset = float(row["offset"])
        annotator_index = int(row["annotator"]) if "annotator" in row and pd.notna(row["annotator"]) else None

        segments.append(AudioLabelSegment(class_name, annotator_index, onset, offset))

    return segments

def visualize_audio_features_labels(common_ids, n=10, feature_key="mfcc", display_annotations:bool=False):

    audio_dir=CFG.AUDIO_DIR
    features_dir=CFG.FEATURES_DIR
    labels_dir=CFG.LABELS_DIR

    for idx in common_ids[:n]:
        frame_duration_sec = 0.120
        audio_path = os.path.join(audio_dir, f"{idx}.mp3")
        features_path = os.path.join(features_dir, f"{idx}.npz")
        labels_path = os.path.join(labels_dir, f"{idx}_labels.npz")

        class_names = load_class_names(CFG.LABELS_DIR)

        annotation_segments = extract_annotation_segments_from_csv(annotations_csv_path=os.path.join(CFG.DATA_DIR, "annotations.csv"),file_id=idx)
        print(f"Annotation segments: {annotation_segments}")
        label_segments = extract_segments_per_annotator(labels_path, class_names, frame_duration_sec)
        print(f"Label segments: {label_segments}")
                
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

        time_axis_audio = np.linspace(0, duration_sec, len(y))
        time_axis_features = np.arange(time_steps) * frame_duration_sec

        fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        fig.canvas.mpl_connect('close_event', on_close)
        plt.subplots_adjust(left=0.25)  # Adjust as needed (0.25 works well)

        # [0] Audio waveform
        axs[0].plot(time_axis_audio, y, color="gray")
        axs[0].set_title(f"Audio Waveform - Sample ID: {idx}")
        axs[0].set_ylabel("Amplitude")

        # [1] Feature plot
        axs[1].plot(time_axis_features, np.mean(features, axis=1), color="blue")
        axs[1].set_title(f"Feature Timeline ({feature_key})")
        axs[1].set_ylabel("Mean Feature Value")

        # [2] Annotations (orange)
        plot_segments_panel(axs[2], annotation_segments, duration_sec, "Human Annotations")

        # [3] Label Segments (blueish)
        plot_segments_panel(axs[3], label_segments, duration_sec, "Per-Annotator Label Segments")

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