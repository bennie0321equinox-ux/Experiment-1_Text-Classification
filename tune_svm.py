# tune_svm.py：线性 SVM 的第一轮参数与 TF-IDF 特征调优。

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def save_initial_baseline() -> dict:
    train_df = pd.read_csv(DATA_DIR / "train_split_clean.csv").fillna("")
    valid_df = pd.read_csv(DATA_DIR / "valid_split_clean.csv").fillna("")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
        strip_accents="unicode",
        ngram_range=(1, 1),
    )
    X_train = vectorizer.fit_transform(train_df["text_clean"])
    X_valid = vectorizer.transform(valid_df["text_clean"])

    model = LinearSVC(C=0.5, max_iter=5000, random_state=RANDOM_STATE)
    model.fit(X_train, train_df["target"])

    valid_pred = model.predict(X_valid)
    metrics = {
        "model": "LinearSVC",
        "baseline_type": "initial_before_optimization",
        "params": {"C": 0.5, "max_iter": 5000, "random_state": RANDOM_STATE},
        "feature": {
            "max_features": 5000,
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "lowercase": False,
            "ngram_range": [1, 1],
        },
        "train_accuracy": float(accuracy_score(train_df["target"], model.predict(X_train))),
        "valid_accuracy": float(accuracy_score(valid_df["target"], valid_pred)),
        "train_macro_f1": float(f1_score(train_df["target"], model.predict(X_train), average="macro")),
        "valid_macro_f1": float(f1_score(valid_df["target"], valid_pred, average="macro")),
        "classification_report": classification_report(valid_df["target"], valid_pred, zero_division=0),
    }

    (RESULTS_DIR / "initial_baseline_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "initial_baseline_validation_report.txt").write_text(metrics["classification_report"], encoding="utf-8")
    return metrics


def run_tuning() -> dict:
    train_df = pd.read_csv(DATA_DIR / "train_split_clean.csv").fillna("")
    valid_df = pd.read_csv(DATA_DIR / "valid_split_clean.csv").fillna("")

    results = []
    for ngram_range in [(1, 1), (1, 2)]:
        for max_features in [5000, 8000]:
            for C in [0.3, 0.5, 0.7, 1.0, 2.0]:
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    lowercase=False,
                    strip_accents="unicode",
                    ngram_range=ngram_range,
                )
                X_train = vectorizer.fit_transform(train_df["text_clean"])
                X_valid = vectorizer.transform(valid_df["text_clean"])

                model = LinearSVC(C=C, max_iter=5000, random_state=RANDOM_STATE)
                model.fit(X_train, train_df["target"])
                valid_pred = model.predict(X_valid)

                result = {
                    "ngram_range": list(ngram_range),
                    "max_features": max_features,
                    "C": C,
                    "train_accuracy": float(accuracy_score(train_df["target"], model.predict(X_train))),
                    "valid_accuracy": float(accuracy_score(valid_df["target"], valid_pred)),
                    "train_macro_f1": float(f1_score(train_df["target"], model.predict(X_train), average="macro")),
                    "valid_macro_f1": float(f1_score(valid_df["target"], valid_pred, average="macro")),
                    "classification_report": classification_report(valid_df["target"], valid_pred, zero_division=0),
                }
                results.append(result)

    ranked = sorted(results, key=lambda item: (item["valid_macro_f1"], item["valid_accuracy"]), reverse=True)
    best = ranked[0]
    (RESULTS_DIR / "tuned_model_results.json").write_text(json.dumps({"ranked_results": ranked}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "best_tuned_model_summary.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"best": best, "all_results": ranked}


def main() -> None:
    baseline = save_initial_baseline()
    tuning = run_tuning()
    print("=== 初始基线 ===")
    print(json.dumps({
        "model": baseline["model"],
        "params": baseline["params"],
        "valid_accuracy": baseline["valid_accuracy"],
        "valid_macro_f1": baseline["valid_macro_f1"],
    }, ensure_ascii=False, indent=2))
    print("\n=== 优化后最佳结果 ===")
    print(json.dumps({
        "ngram_range": tuning["best"]["ngram_range"],
        "max_features": tuning["best"]["max_features"],
        "C": tuning["best"]["C"],
        "valid_accuracy": tuning["best"]["valid_accuracy"],
        "valid_macro_f1": tuning["best"]["valid_macro_f1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
