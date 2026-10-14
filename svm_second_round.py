# svm_second_round.py：线性 SVM 的第二轮参数与 TF-IDF 特征调优。

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


def load_split_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(DATA_DIR / "train_split_clean.csv").fillna("")
    valid_df = pd.read_csv(DATA_DIR / "valid_split_clean.csv").fillna("")
    return train_df, valid_df


def build_and_evaluate(train_df: pd.DataFrame, valid_df: pd.DataFrame, ngram_range: tuple[int, int], max_features: int, min_df: int, C: float):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
        strip_accents="unicode",
        ngram_range=ngram_range,
    )
    X_train = vectorizer.fit_transform(train_df["text_clean"])
    X_valid = vectorizer.transform(valid_df["text_clean"])

    model = LinearSVC(C=C, max_iter=8000, random_state=RANDOM_STATE)
    model.fit(X_train, train_df["target"])

    valid_pred = model.predict(X_valid)
    train_pred = model.predict(X_train)

    result = {
        "ngram_range": list(ngram_range),
        "max_features": max_features,
        "min_df": min_df,
        "C": C,
        "max_iter": 8000,
        "train_accuracy": float(accuracy_score(train_df["target"], train_pred)),
        "valid_accuracy": float(accuracy_score(valid_df["target"], valid_pred)),
        "train_macro_f1": float(f1_score(train_df["target"], train_pred, average="macro")),
        "valid_macro_f1": float(f1_score(valid_df["target"], valid_pred, average="macro")),
        "classification_report": classification_report(valid_df["target"], valid_pred, zero_division=0),
    }
    return result


def main() -> None:
    train_df, valid_df = load_split_data()

    results = []
    for ngram_range in [(1, 1), (1, 2)]:
        for max_features in [5000, 8000, 12000, 15000]:
            for min_df in [1, 2, 3]:
                for C in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                    result = build_and_evaluate(train_df, valid_df, ngram_range, max_features, min_df, C)
                    results.append(result)

    ranked = sorted(results, key=lambda item: (item["valid_macro_f1"], item["valid_accuracy"]), reverse=True)
    best = ranked[0]

    (RESULTS_DIR / "second_round_tuning_results.json").write_text(
        json.dumps({"ranked_results": ranked}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "second_round_best_summary.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "second_round_best_validation_report.txt").write_text(best["classification_report"], encoding="utf-8")

    print("=== 第二轮调优：Top 10 ===")
    for idx, item in enumerate(ranked[:10], start=1):
        print(
            f"{idx}. ngram={item['ngram_range']}, max_features={item['max_features']}, min_df={item['min_df']}, "
            f"C={item['C']}, valid_acc={item['valid_accuracy']:.4f}, valid_macro_f1={item['valid_macro_f1']:.4f}"
        )
    print("\n=== 最优配置 ===")
    print(json.dumps({
        "ngram_range": best["ngram_range"],
        "max_features": best["max_features"],
        "min_df": best["min_df"],
        "C": best["C"],
        "valid_accuracy": best["valid_accuracy"],
        "valid_macro_f1": best["valid_macro_f1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
