import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support, accuracy_score
from models import ModelResult
from utils import format_clickable_path
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


class ModelEvaluationResult:
    def __init__(self, model_name, roc_auc_micro, balanced_accuracy=None):
        self.model_name = model_name
        self.roc_auc_micro = roc_auc_micro
        self.balanced_accuracy = balanced_accuracy

    def to_dict(self):
        return {
            "Model Name": self.model_name,
            "ROC-AUC Micro": self.roc_auc_micro,
            "Balanced Accuracy": self.balanced_accuracy
        }

def plot_aggregated_confusion_matrix(Y_test, Y_pred, model_result, output_dir):
    y_true_flat = Y_test.flatten()
    y_pred_flat = Y_pred.flatten()

    cm = confusion_matrix(y_true_flat, y_pred_flat)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Global Aggregated Confusion Matrix")

    filename = f"{model_result.model_name()}_aggregated_confusion_matrix.png"
    cm_path = os.path.join(output_dir, filename)
    plt.savefig(cm_path)
    plt.close()

    print(f"Aggregated confusion matrix saved to {format_clickable_path(cm_path)}")


def plot_per_class_confusion_matrix(Y_test, Y_pred, class_names, model_result, output_dir):
    n_classes = len(class_names)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for true_labels, pred_labels in zip(Y_test, Y_pred):
        true_idx = np.where(true_labels == 1)[0]
        pred_idx = np.where(pred_labels == 1)[0]

        for t in true_idx:
            for p in pred_idx:
                confusion[t, p] += 1

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion, xticklabels=class_names, yticklabels=class_names,
                 cmap="Blues", square=True)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Per-Class Confusion Matrix (Label Co-occurrence)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    filename = f"{model_result.model_name()}_per_class_confusion_matrix.png"
    cm_path = os.path.join(output_dir, filename)
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    print(f"Per-class confusion matrix saved to {format_clickable_path(cm_path)}")

class ModelEvaluationResult:
    def __init__(self, model_name, roc_auc_micro, balanced_accuracy=None):
        self.model_name = model_name
        self.roc_auc_micro = roc_auc_micro
        self.balanced_accuracy = balanced_accuracy
        self.best_epoch = None

    def to_dict(self):
        return {
            "Model Name": self.model_name,
            "Best Epoch": self.best_epoch,
            "ROC-AUC Micro": self.roc_auc_micro,
            "Balanced Accuracy": self.balanced_accuracy
        }
        
class ModelEvaluationTracker:
    def __init__(self, output_dir, summary_dir):
        self.output_dir = output_dir
        self.summary_dir = summary_dir
        self.summary_csv_path = os.path.join(summary_dir, "model_evaluation_summary.csv")
        self.evaluation_results = []
        self.existing_names = set()

        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.summary_csv_path):
            summary_df = pd.read_csv(self.summary_csv_path)
            self.evaluation_results = summary_df.to_dict(orient="records")
            self.existing_names = set(row["Model Name"] for _, row in summary_df.iterrows())

    def already_evaluated(self, model_name):
        return model_name in self.existing_names

    def add_result(self, eval_result):
        model_name = eval_result.model_name
        self.evaluation_results.append(eval_result.to_dict())
        self.existing_names.add(model_name)
        self._save()

    def _save(self):
        summary_df = pd.DataFrame(self.evaluation_results)
        summary_df.to_csv(self.summary_csv_path, index=False)
        print(f"Summary saved to {format_clickable_path(self.summary_csv_path)}")

def evaluate_model(model_result : ModelResult, X_test, Y_test, class_names, output_dir) -> ModelEvaluationResult :
    Y_pred = model_result.predict(X_test)
    y_test_flat = Y_test.flatten()
    y_pred_flat = Y_pred.flatten()

    try:
        y_test_probs = model_result.predict_proba(X_test)
        auc_micro = roc_auc_score(Y_test, y_test_probs, average="micro")
        print(f"\nROC-AUC Micro (Primary Selection Metric): {auc_micro:.4f}")
    except (ValueError, AttributeError):
        auc_micro = None
        print("ROC-AUC Micro could not be calculated.")

    balanced_acc = balanced_accuracy_score(y_test_flat, y_pred_flat)
    print(f"Balanced Accuracy (For Reporting Only): {balanced_acc:.4f}")

    plot_aggregated_confusion_matrix(Y_test, Y_pred, model_result, output_dir)
    plot_per_class_confusion_matrix(Y_test, Y_pred, class_names, model_result, output_dir)

    return ModelEvaluationResult(
        model_result.model_name(),
        auc_micro,
        balanced_acc
    )