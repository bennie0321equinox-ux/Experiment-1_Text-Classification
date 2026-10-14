# tfidf_features.py：生成基线对照用的 5000 维词级 TF-IDF 特征。

import json
from pathlib import Path

import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MAX_FEATURES = 5000


def load_clean_text(filename: str, has_target: bool) -> tuple[pd.Series, pd.Series | None]:
    data = pd.read_csv(DATA_DIR / filename)
    if "text_clean" not in data.columns:
        raise ValueError(f"{filename} 缺少 text_clean 列")
    if has_target and "target" not in data.columns:
        raise ValueError(f"{filename} 缺少 target 列")
    target = data["target"] if has_target else None
    return data["text_clean"].fillna(""), target


def main() -> None:
    train_text, train_target = load_clean_text("train_split_clean.csv", has_target=True)
    valid_text, valid_target = load_clean_text("valid_split_clean.csv", has_target=True)
    test_text, _ = load_clean_text("test_clean.csv", has_target=False)

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
    )

    train_matrix = vectorizer.fit_transform(train_text)
    valid_matrix = vectorizer.transform(valid_text)
    test_matrix = vectorizer.transform(test_text)

    save_npz(DATA_DIR / "train_tfidf.npz", train_matrix)
    save_npz(DATA_DIR / "valid_tfidf.npz", valid_matrix)
    save_npz(DATA_DIR / "test_tfidf.npz", test_matrix)

    feature_names = vectorizer.get_feature_names_out()
    pd.DataFrame({"feature": feature_names}).to_csv(
        RESULTS_DIR / "tfidf_features.csv", index=False
    )

    idf_ranking = sorted(
        zip(feature_names, vectorizer.idf_),
        key=lambda item: item[1],
        reverse=True,
    )
    report = {
        "vectorizer": {
            "max_features": MAX_FEATURES,
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "lowercase": False,
            "fit_on": "train_split_clean.csv",
        },
        "train_shape": list(train_matrix.shape),
        "valid_shape": list(valid_matrix.shape),
        "test_shape": list(test_matrix.shape),
        "train_nonzero": int(train_matrix.nnz),
        "valid_nonzero": int(valid_matrix.nnz),
        "test_nonzero": int(test_matrix.nnz),
        "train_sparsity": 1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]),
        "valid_sparsity": 1 - valid_matrix.nnz / (valid_matrix.shape[0] * valid_matrix.shape[1]),
        "test_sparsity": 1 - test_matrix.nnz / (test_matrix.shape[0] * test_matrix.shape[1]),
        "highest_idf_features": [
            {"feature": feature, "idf": float(score)}
            for feature, score in idf_ranking[:20]
        ],
        "target_counts": {
            "train": train_target.value_counts().sort_index().astype(int).to_dict(),
            "valid": valid_target.value_counts().sort_index().astype(int).to_dict(),
        },
    }
    with (RESULTS_DIR / "tfidf_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("已保存 train_tfidf.npz、valid_tfidf.npz、test_tfidf.npz")
    print("已保存 tfidf_features.csv 和 tfidf_report.json")


if __name__ == "__main__":
    main()
