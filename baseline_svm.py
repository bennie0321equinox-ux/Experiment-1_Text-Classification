# baseline_svm.py：训练线性 SVM 基线模型并在验证集上评估。

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def main() -> None:
    X_train = load_npz(DATA_DIR / "train_tfidf.npz")
    X_valid = load_npz(DATA_DIR / "valid_tfidf.npz")
    y_train = pd.read_csv(DATA_DIR / "train_split_clean.csv")["target"]
    y_valid = pd.read_csv(DATA_DIR / "valid_split_clean.csv")["target"]

    model = LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    train_prediction = model.predict(X_train)
    valid_prediction = model.predict(X_valid)

    metrics = {
        "model": "LinearSVC",
        "parameters": {"C": 1.0, "max_iter": 5000, "random_state": RANDOM_STATE},
        "train_accuracy": float(accuracy_score(y_train, train_prediction)),
        "valid_accuracy": float(accuracy_score(y_valid, valid_prediction)),
        "train_macro_f1": float(f1_score(y_train, train_prediction, average="macro")),
        "valid_macro_f1": float(f1_score(y_valid, valid_prediction, average="macro")),
        "fit_seconds": fit_seconds,
        "train_shape": list(X_train.shape),
        "valid_shape": list(X_valid.shape),
    }

    with (RESULTS_DIR / "baseline_svm_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    report = classification_report(y_valid, valid_prediction, zero_division=0)
    (RESULTS_DIR / "baseline_svm_classification_report.txt").write_text(report, encoding="utf-8")

    ConfusionMatrixDisplay.from_predictions(
        y_valid,
        valid_prediction,
        cmap="Blues",
        values_format="d",
    )
    plt.title("Linear SVM baseline: validation confusion matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_svm_confusion_matrix.png", dpi=180)
    plt.close()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n验证集分类报告：")
    print(report)
    print("已保存基线指标、分类报告和验证集混淆矩阵。")


if __name__ == "__main__":
    main()
