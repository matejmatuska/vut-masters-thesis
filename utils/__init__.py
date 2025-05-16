import copy

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


class EarlyStopping:
    def __init__(self, patience, tolerance=0.0):
        """
        Helper to stop training when validation loss does not improve.

        Best model state and additional custom data at the best loss are stored.

        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param tolerance: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.tolerance = tolerance
        self.best_loss = float("inf") - tolerance * 10
        self.wait = 0
        self.stop = False
        self._data_at_best = None
        self._best_model_state = None

    def __call__(self, current_loss, model, **kwargs) -> bool:
        assert not self.stop, "Early stopping has been triggered already."
        if current_loss < self.best_loss + self.tolerance:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self._data_at_best = kwargs
                self._best_model_state = copy.deepcopy(model.state_dict())

        if current_loss < self.best_loss - self.tolerance:
            self.best_loss = current_loss
            self._data_at_best = kwargs
            self._best_model_state = model.state_dict()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True
        return self.stop

    @property
    def min_loss(self):
        return self.best_loss

    @property
    def best_model_state(self):
        return self._best_model_state

    @property
    def data_at_best(self):
        if self.stop:
            return self._data_at_best
        raise ValueError("Early stopping has not been triggered yet.")

    def __bool__(self):
        return self.stop


def log_conf_matrix(cm, artifact_file):
    plt.figure(figsize=(8, 6))
    ticks = np.arange(0, cm.shape[0])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=ticks, yticklabels=ticks
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    mlflow.log_figure(plt.gcf(), artifact_file)
    plt.close()


def log_class_stats(all_preds, all_labels, suffix):
    conf_matrix = confusion_matrix(all_labels, all_preds)
    log_conf_matrix(conf_matrix, artifact_file=f"confusion_matrix_{suffix}.pdf")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    mlflow.log_dict(report_dict, artifact_file=f"classification_report_{suffix}.json")


def compute_class_weights(dataset):
    labels = torch.cat([data.y for data in dataset], dim=0).numpy()

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights_tensor
