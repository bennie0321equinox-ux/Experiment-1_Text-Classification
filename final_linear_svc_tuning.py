# final_linear_svc_tuning.py：线性 SVM 的最后一轮精确调优。

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


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_split_clean.csv").fillna("")
    valid_df = pd.read_csv(DATA_DIR / "valid_split_clean.csv").fillna("")
    return train_df, valid_df


def evaluate_case(train_df, valid_df, max_features, min_df, max_df, ngram_range, C, class_weight=None):
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        lowercase=False,
        strip_accents="unicode",
        ngram_range=ngram_range,
    )
    X_train = vec.fit_transform(train_df["text_clean"])
    X_valid = vec.transform(valid_df["text_clean"])

    model = LinearSVC(C=C, max_iter=12000, random_state=RANDOM_STATE, class_weight=class_weight)
    model.fit(X_train, train_df["target"])
    y_pred = model.predict(X_valid)

    return {
        "max_features": max_features,
        "min_df": min_df,
        "max_df": max_df,
        "ngram_range": list(ngram_range),
        "C": C,
        "class_weight": class_weight,
        "train_accuracy": float(accuracy_score(train_df["target"], model.predict(X_train))),
        "valid_accuracy": float(accuracy_score(valid_df["target"], y_pred)),
        "train_macro_f1": float(f1_score(train_df["target"], model.predict(X_train), average="macro")),
        "valid_macro_f1": float(f1_score(valid_df["target"], y_pred, average="macro")),
        "classification_report": classification_report(valid_df["target"], y_pred, zero_division=0),
    }


def main() -> None:
    train_df, valid_df = load_data()

    cases = []
    for max_features in [50000, 80000, 100000]:
        for min_df in [1, 2, 3]:
            for ngram_range in [(1, 1), (1, 2), (1, 3)]:
                for C in [0.7, 1.0, 1.5, 2.0, 3.0]:
                    cases.append((max_features, min_df, 0.95, ngram_range, C, None))
                    cases.append((max_features, min_df, 0.95, ngram_range, C, "balanced"))

    results = []
    for max_features, min_df, max_df, ngram_range, C, class_weight in cases:
        result = evaluate_case(train_df, valid_df, max_features, min_df, max_df, ngram_range, C, class_weight)
        results.append(result)

    ranked = sorted(results, key=lambda x: (x["valid_macro_f1"], x["valid_accuracy"]), reverse=True)
    best = ranked[0]

    (RESULTS_DIR / "linear_svc_final_tuning_results.json").write_text(
        json.dumps({"ranked_results": ranked[:20]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "linear_svc_final_best_summary.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "linear_svc_final_best_report.txt").write_text(best["classification_report"], encoding="utf-8")

    print("=== 最优配置 ===")
    print(json.dumps({
        "max_features": best["max_features"],
        "min_df": best["min_df"],
        "max_df": best["max_df"],
        "ngram_range": best["ngram_range"],
        "C": best["C"],
        "class_weight": best["class_weight"],
        "valid_accuracy": best["valid_accuracy"],
        "valid_macro_f1": best["valid_macro_f1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
