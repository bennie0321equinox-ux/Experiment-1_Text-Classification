# learning_curve_experiment.py：训练样本量对 Macro-F1 影响的学习曲线实验。

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RANDOM_STATE = 42

TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]
CV_FOLDS = 5


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train_split_clean.csv").fillna("")
    text = train["text_clean"]
    target = train["target"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95,
                                  sublinear_tf=True, lowercase=False,
                                  strip_accents="unicode")),
        ("clf", LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)),
    ])

    sizes, train_scores, valid_scores = learning_curve(
        pipeline, text, target,
        train_sizes=TRAIN_SIZES,
        cv=CV_FOLDS,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    plt.figure(figsize=(7.4, 4.6))
    plt.plot(sizes, train_mean, "o-", color="#1f77b4", linewidth=2, label="Train Macro-F1")
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, color="#1f77b4", alpha=0.15)
    plt.plot(sizes, valid_mean, "s-", color="#d62728", linewidth=2, label="CV Macro-F1")
    plt.fill_between(sizes, valid_mean - valid_std, valid_mean + valid_std, color="#d62728", alpha=0.15)
    plt.xlabel("Number of training samples")
    plt.ylabel("Macro-F1")
    plt.title("Learning Curve: TF-IDF + LinearSVC (5-fold CV on train split)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "learning_curve_linear_svc.png", dpi=200)
    plt.close()

    record = {
        "feature": "word TF-IDF max_features=5000, min_df=2, max_df=0.95, sublinear_tf=True",
        "model": "LinearSVC C=1.0, max_iter=5000",
        "cv_folds": CV_FOLDS,
        "train_sizes": [int(size) for size in sizes],
        "train_macro_f1_mean": [float(value) for value in train_mean],
        "train_macro_f1_std": [float(value) for value in train_std],
        "cv_macro_f1_mean": [float(value) for value in valid_mean],
        "cv_macro_f1_std": [float(value) for value in valid_std],
        "note": "TfidfVectorizer is inside the pipeline, so each fold refits on its own training part only.",
    }
    (RESULTS_DIR / "learning_curve_linear_svc.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("已保存学习曲线到 results/。")


if __name__ == "__main__":
    main()
