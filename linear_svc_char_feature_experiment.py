# linear_svc_char_feature_experiment.py：词级与字符级 TF-IDF 的融合对比实验。

import json
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def score_model(train_df, valid_df, word_train, word_valid, char_train=None, char_valid=None, char_weight=0.0, C=1.0):
    if char_train is None:
        train_features = word_train
        valid_features = word_valid
    else:
        train_features = hstack([word_train, char_train * char_weight], format="csr")
        valid_features = hstack([word_valid, char_valid * char_weight], format="csr")

    model = LinearSVC(C=C, max_iter=12000, random_state=RANDOM_STATE)
    model.fit(train_features, train_df["target"])
    predictions = model.predict(valid_features)
    return {
        "C": C,
        "char_weight": char_weight,
        "train_feature_count": int(train_features.shape[1]),
        "valid_accuracy": float(accuracy_score(valid_df["target"], predictions)),
        "valid_macro_f1": float(f1_score(valid_df["target"], predictions, average="macro")),
        "classification_report": classification_report(valid_df["target"], predictions, zero_division=0),
    }


def main():
    train_df = pd.read_csv(DATA_DIR / "train_split_clean.csv").fillna("")
    valid_df = pd.read_csv(DATA_DIR / "valid_split_clean.csv").fillna("")
    train_text = train_df["text_clean"]
    valid_text = valid_df["text_clean"]

    word_vectorizer = TfidfVectorizer(
        max_features=80000,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
        strip_accents="unicode",
        ngram_range=(1, 1),
    )
    word_train = word_vectorizer.fit_transform(train_text)
    word_valid = word_vectorizer.transform(valid_text)

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150000,
        sublinear_tf=True,
        lowercase=False,
        strip_accents="unicode",
    )
    char_train = char_vectorizer.fit_transform(train_text)
    char_valid = char_vectorizer.transform(valid_text)

    results = [score_model(train_df, valid_df, word_train, word_valid)]
    for char_weight in (0.25, 0.5, 1.0):
        for C in (0.7, 1.0, 1.5, 2.0):
            results.append(
                score_model(
                    train_df,
                    valid_df,
                    word_train,
                    word_valid,
                    char_train,
                    char_valid,
                    char_weight=char_weight,
                    C=C,
                )
            )

    ranked = sorted(results, key=lambda result: (result["valid_macro_f1"], result["valid_accuracy"]), reverse=True)
    baseline = results[0]
    best = ranked[0]
    output = {
        "experiment": "LinearSVC word and character TF-IDF feature combination",
        "validation_protocol": "Vectorizers fit on train_split_clean.csv only; validation split transformed only.",
        "word_tfidf": {
            "max_features": 80000,
            "min_df": 1,
            "max_df": 0.95,
            "ngram_range": [1, 1],
            "sublinear_tf": True,
            "strip_accents": "unicode",
        },
        "char_tfidf": {
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "min_df": 2,
            "max_features": 150000,
            "sublinear_tf": True,
            "strip_accents": "unicode",
        },
        "previous_best_linear_svc": {
            "valid_accuracy": 0.8921302578018996,
            "valid_macro_f1": 0.8922064172607055,
        },
        "word_only_baseline_this_run": baseline,
        "best_result": best,
        "improved_over_previous_best": best["valid_macro_f1"] > 0.8922064172607055,
        "ranked_results": ranked,
    }
    output_path = RESULTS_DIR / "linear_svc_char_feature_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": output_path.name,
        "word_only_baseline": {
            "valid_accuracy": baseline["valid_accuracy"],
            "valid_macro_f1": baseline["valid_macro_f1"],
        },
        "best": {
            "C": best["C"],
            "char_weight": best["char_weight"],
            "valid_accuracy": best["valid_accuracy"],
            "valid_macro_f1": best["valid_macro_f1"],
        },
        "improved_over_previous_best": output["improved_over_previous_best"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
