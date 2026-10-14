# mlp_experiment.py：逐轮增量训练 MLP，并记录每轮的训练损失与验证损失。

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, log_loss
from sklearn.neural_network import MLPClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RANDOM_STATE = 42
N_EPOCHS = 40


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    X_train = load_npz(DATA_DIR / "train_tfidf.npz")
    X_valid = load_npz(DATA_DIR / "valid_tfidf.npz")
    y_train = pd.read_csv(DATA_DIR / "train_split_clean.csv")["target"]
    y_valid = pd.read_csv(DATA_DIR / "valid_split_clean.csv")["target"]
    classes = np.unique(y_train)

    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=1,
        random_state=RANDOM_STATE,
    )

    train_losses = []
    valid_losses = []
    for _ in range(N_EPOCHS):
        model.partial_fit(X_train, y_train, classes=classes)
        train_losses.append(float(model.loss_))
        valid_losses.append(float(log_loss(y_valid, model.predict_proba(X_valid), labels=classes)))

    best_epoch = int(np.argmin(valid_losses))
    valid_prediction = model.predict(X_valid)
    train_prediction = model.predict(X_train)

    metrics = {
        "model": "MLPClassifier",
        "parameters": {
            "hidden_layer_sizes": [100], "activation": "relu", "solver": "adam",
            "alpha": 0.001, "learning_rate_init": 0.001, "epochs": N_EPOCHS,
            "random_state": RANDOM_STATE,
        },
        "feature_source": "train_tfidf.npz / valid_tfidf.npz (5000-dimensional word TF-IDF)",
        "epochs_run": N_EPOCHS,
        "best_epoch_by_valid_loss": best_epoch + 1,
        "final_train_loss": train_losses[-1],
        "final_valid_loss": valid_losses[-1],
        "best_valid_loss": valid_losses[best_epoch],
        "train_accuracy": float(accuracy_score(y_train, train_prediction)),
        "train_macro_f1": float(f1_score(y_train, train_prediction, average="macro")),
        "valid_accuracy": float(accuracy_score(y_valid, valid_prediction)),
        "valid_macro_f1": float(f1_score(y_valid, valid_prediction, average="macro")),
    }
    (RESULTS_DIR / "mlp_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "mlp_validation_report.txt").write_text(
        classification_report(y_valid, valid_prediction, zero_division=0), encoding="utf-8"
    )

    epochs = range(1, N_EPOCHS + 1)
    plt.figure(figsize=(7.4, 4.6))
    plt.plot(epochs, train_losses, "o-", color="#1f77b4", linewidth=2, markersize=3, label="Training loss")
    plt.plot(epochs, valid_losses, "s-", color="#d62728", linewidth=2, markersize=3, label="Validation loss")
    plt.axvline(best_epoch + 1, color="grey", linestyle="--", alpha=0.7)
    plt.text(best_epoch + 1, max(valid_losses) * 0.96,
             f"best epoch {best_epoch + 1}", ha="center", va="top", fontsize=9)
    plt.xlabel("Epoch")
    plt.ylabel("Log loss")
    plt.title("MLP Training vs Validation Loss")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mlp_loss_curve.png", dpi=200)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_valid, valid_prediction, cmap="Blues", values_format="d")
    plt.title("MLP Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mlp_confusion_matrix.png", dpi=200)
    plt.close()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("已保存 mlp_loss_curve.png、mlp_confusion_matrix.png、mlp_metrics.json 和 mlp_validation_report.txt")


if __name__ == "__main__":
    main()
