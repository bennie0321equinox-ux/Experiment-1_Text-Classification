# final_svc_validation_analysis.py：复现最终模型的验证结果，并生成实验报告所需图表。

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import PredefinedSplit, validation_curve
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RANDOM_STATE = 42

C_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
CHAR_FEATURE_WEIGHT = 0.25


def build_features(train_text: pd.Series, valid_text: pd.Series):
    word = TfidfVectorizer(max_features=80000, min_df=1, max_df=0.95, sublinear_tf=True,
                           lowercase=False, strip_accents="unicode", ngram_range=(1, 1))
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           max_features=150000, sublinear_tf=True, lowercase=False,
                           strip_accents="unicode")
    train_features = hstack([
        word.fit_transform(train_text),
        char.fit_transform(train_text) * CHAR_FEATURE_WEIGHT,
    ], format="csr")
    valid_features = hstack([
        word.transform(valid_text),
        char.transform(valid_text) * CHAR_FEATURE_WEIGHT,
    ], format="csr")
    return train_features, valid_features


def plot_validation_curve(train_features, valid_features, y_train, y_valid) -> dict:
    X_all = vstack([train_features, valid_features], format="csr")
    y_all = np.concatenate([np.asarray(y_train), np.asarray(y_valid)])
    test_fold = np.concatenate([
        np.full(len(y_train), -1, dtype=int),
        np.zeros(len(y_valid), dtype=int),
    ])
    cv = PredefinedSplit(test_fold)

    train_scores, valid_scores = validation_curve(
        LinearSVC(max_iter=12000, random_state=RANDOM_STATE),
        X_all, y_all,
        param_name="C",
        param_range=C_GRID,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    best_index = int(np.argmax(valid_mean))

    plt.figure(figsize=(7.4, 4.6))
    plt.semilogx(C_GRID, train_mean, "o-", color="#1f77b4", linewidth=2, label="Train Macro-F1")
    plt.semilogx(C_GRID, valid_mean, "s-", color="#d62728", linewidth=2, label="Validation Macro-F1")
    plt.axvline(C_GRID[best_index], color="grey", linestyle="--", alpha=0.7)
    plt.text(C_GRID[best_index] * 1.15, 0.93,
             f"best C={C_GRID[best_index]}\nF1={valid_mean[best_index]:.4f}",
             ha="left", va="center", fontsize=9, color="dimgray")
    plt.xlabel("LinearSVC regularization strength C (log scale)")
    plt.ylabel("Macro-F1")
    plt.title("Validation Curve: LinearSVC (word + char TF-IDF)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "validation_curve_linear_svc.png", dpi=200)
    plt.close()

    return {
        "param_name": "C",
        "param_range": list(C_GRID),
        "train_macro_f1": [float(value) for value in train_mean],
        "valid_macro_f1": [float(value) for value in valid_mean],
        "best_C": float(C_GRID[best_index]),
        "best_valid_macro_f1": float(valid_mean[best_index]),
        "note": "Vectorizers fit on train split only; validation split never used for fitting.",
    }


def collect_comparison_scores(final_score: float) -> dict:
    def read_json(name: str):
        path = RESULTS_DIR / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    scores = {
        "Naive Bayes": 0.8352,
        "MLP": 0.8442,
        "Logistic Regression": 0.8821,
        "LinearSVC word TF-IDF": 0.8922,
    }

    mlp = read_json("mlp_metrics.json")
    if mlp:
        scores["MLP"] = mlp["valid_macro_f1"]

    model_comparison = read_json("model_comparison_results.json")
    if model_comparison:
        for item in model_comparison.get("ranked_results", []):
            if item["model"].startswith("MultinomialNB"):
                scores["Naive Bayes"] = item["valid_macro_f1"]
                break

    quick_test = read_json("logistic_regression_quick_test.json")
    if quick_test and quick_test.get("ranked_results"):
        scores["Logistic Regression"] = quick_test["ranked_results"][0]["valid_macro_f1"]

    char_feature = read_json("linear_svc_char_feature_results.json")
    if char_feature:
        scores["LinearSVC word TF-IDF"] = char_feature["word_only_baseline_this_run"]["valid_macro_f1"]

    scores["LinearSVC word + char TF-IDF"] = final_score
    return scores


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train_split_clean.csv").fillna("")
    valid = pd.read_csv(DATA_DIR / "valid_split_clean.csv").fillna("")

    train_features, valid_features = build_features(train["text_clean"], valid["text_clean"])

    model = LinearSVC(C=1.0, max_iter=12000, random_state=RANDOM_STATE)
    model.fit(train_features, train["target"])
    pred = model.predict(valid_features)
    metrics = {
        "model": "LinearSVC word plus character TF-IDF",
        "parameters": {"C": 1.0, "max_iter": 12000, "random_state": RANDOM_STATE,
                       "word_max_features": 80000, "char_max_features": 150000,
                       "char_ngram_range": [3, 5], "char_weight": CHAR_FEATURE_WEIGHT},
        "valid_accuracy": float(accuracy_score(valid["target"], pred)),
        "valid_macro_f1": float(f1_score(valid["target"], pred, average="macro")),
        "train_accuracy": float(accuracy_score(train["target"], model.predict(train_features))),
        "train_macro_f1": float(f1_score(train["target"], model.predict(train_features), average="macro")),
    }
    (RESULTS_DIR / "final_svc_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "final_svc_validation_report.txt").write_text(
        classification_report(valid["target"], pred, zero_division=0), encoding="utf-8"
    )

    ConfusionMatrixDisplay.from_predictions(valid["target"], pred, cmap="Blues", values_format="d")
    plt.title("Best LinearSVC Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_svc_confusion_matrix.png", dpi=200)
    plt.close()

    curve = plot_validation_curve(train_features, valid_features, train["target"], valid["target"])
    (RESULTS_DIR / "validation_curve_linear_svc.json").write_text(
        json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    comparison = pd.DataFrame(
        list(collect_comparison_scores(metrics["valid_macro_f1"]).items()),
        columns=["model", "macro_f1"],
    )
    comparison.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)
    plot_data = comparison.sort_values("macro_f1")
    plt.figure(figsize=(8.6, 4.8))
    bars = plt.barh(plot_data["model"], plot_data["macro_f1"], color="#337ab7")
    plt.xlim(0.80, 0.91)
    plt.xlabel("Validation Macro-F1")
    plt.title("Validation Performance Comparison")
    for bar, score in zip(bars, plot_data["macro_f1"]):
        plt.text(score + 0.001, bar.get_y() + bar.get_height() / 2, f"{score:.4f}", va="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_model_comparison.png", dpi=200)
    plt.close()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(curve, ensure_ascii=False, indent=2))
    print("已保存最终模型分类报告、混淆矩阵、验证曲线和模型对比图到 results/。")


if __name__ == "__main__":
    main()
