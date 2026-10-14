# model_comparison.py：对比朴素贝叶斯、逻辑回归与线性 SVM。

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(DATA_DIR / "train_split_clean.csv").fillna("")
    valid_df = pd.read_csv(DATA_DIR / "valid_split_clean.csv").fillna("")
    test_df = pd.read_csv(DATA_DIR / "test_clean.csv").fillna("")
    return train_df, valid_df, test_df


def build_vectorizer(train_text: pd.Series):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(train_text)
    return vectorizer, X_train


def evaluate_model(model_name: str, model, X_train, y_train, X_valid, y_valid) -> dict:
    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)

    metrics = {
        "model": model_name,
        "valid_accuracy": float(accuracy_score(y_valid, valid_pred)),
        "valid_macro_f1": float(f1_score(y_valid, valid_pred, average="macro")),
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "train_macro_f1": float(f1_score(y_train, model.predict(X_train), average="macro")),
        "classification_report": classification_report(y_valid, valid_pred, zero_division=0),
    }
    return metrics


def main() -> None:
    train_df, valid_df, test_df = load_data()

    vectorizer, X_train = build_vectorizer(train_df["text_clean"])
    X_valid = vectorizer.transform(valid_df["text_clean"])
    X_test = vectorizer.transform(test_df["text_clean"])

    y_train = train_df["target"]
    y_valid = valid_df["target"]

    candidates = []

    for C in [0.5, 1.0, 2.0, 5.0, 10.0]:
        model = LinearSVC(C=C, max_iter=5000, random_state=RANDOM_STATE)
        result = evaluate_model(f"LinearSVC_C{C}", model, X_train, y_train, X_valid, y_valid)
        result["params"] = {"C": C, "max_iter": 5000, "random_state": RANDOM_STATE}
        candidates.append(result)

    for C in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        model = LogisticRegression(
            C=C,
            max_iter=4000,
            random_state=RANDOM_STATE,
        )
        result = evaluate_model(f"LogisticRegression_C{C}", model, X_train, y_train, X_valid, y_valid)
        result["params"] = {"C": C, "max_iter": 4000, "solver": "lbfgs", "random_state": RANDOM_STATE}
        candidates.append(result)

    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0]:
        model = MultinomialNB(alpha=alpha)
        result = evaluate_model(f"MultinomialNB_alpha{alpha}", model, X_train, y_train, X_valid, y_valid)
        result["params"] = {"alpha": alpha}
        candidates.append(result)

    ranked = sorted(candidates, key=lambda item: (item["valid_macro_f1"], item["valid_accuracy"]), reverse=True)

    best = ranked[0]
    best_summary = {
        "best_model": best["model"],
        "params": best["params"],
        "valid_accuracy": best["valid_accuracy"],
        "valid_macro_f1": best["valid_macro_f1"],
        "train_accuracy": best["train_accuracy"],
        "train_macro_f1": best["train_macro_f1"],
    }

    (RESULTS_DIR / "model_comparison_results.json").write_text(
        json.dumps({"ranked_results": ranked}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "best_model_summary.json").write_text(
        json.dumps(best_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "best_model_validation_report.txt").write_text(best["classification_report"], encoding="utf-8")

    full_train_text = pd.concat([train_df["text_clean"], valid_df["text_clean"]], ignore_index=True)
    full_train_target = pd.concat([y_train, y_valid], ignore_index=True)
    final_vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
        strip_accents="unicode",
    )
    X_full_train = final_vectorizer.fit_transform(full_train_text)

    model_name = best["model"]
    if model_name.startswith("LinearSVC"):
        C = best["params"]["C"]
        final_model = LinearSVC(C=C, max_iter=5000, random_state=RANDOM_STATE)
    elif model_name.startswith("LogisticRegression"):
        C = best["params"]["C"]
        final_model = LogisticRegression(
            C=C,
            max_iter=4000,
            random_state=RANDOM_STATE,
        )
    else:
        alpha = best["params"]["alpha"]
        final_model = MultinomialNB(alpha=alpha)

    final_model.fit(X_full_train, full_train_target)
    X_test_final = final_vectorizer.transform(test_df["text_clean"])
    test_pred = final_model.predict(X_test_final)

    prediction_df = pd.DataFrame(
        {
            "text": test_df["text"],
            "predicted_target": test_pred,
        }
    )
    prediction_df.to_csv(RESULTS_DIR / "test_predictions_baseline_model.csv", index=False)

    print("=== 模型比较结果 ===")
    for idx, item in enumerate(ranked, start=1):
        print(f"{idx}. {item['model']} | valid_acc={item['valid_accuracy']:.4f} | valid_macro_f1={item['valid_macro_f1']:.4f}")
    print("\n=== 最优模型 ===")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2))
    print("\n已保存：model_comparison_results.json, best_model_summary.json, test_predictions.csv")


if __name__ == "__main__":
    main()
