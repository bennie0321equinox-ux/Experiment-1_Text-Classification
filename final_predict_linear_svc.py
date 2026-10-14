# final_predict_linear_svc.py：用全部有标签数据拟合最终模型，生成提交用的预测文件。

import json
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from prepare_data import clean_text


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

WORD_TFIDF_CONFIG = {
    "max_features": 80000,
    "min_df": 1,
    "max_df": 0.95,
    "sublinear_tf": True,
    "lowercase": False,
    "strip_accents": "unicode",
    "ngram_range": (1, 1),
}
CHAR_TFIDF_CONFIG = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_features": 150000,
    "sublinear_tf": True,
    "lowercase": False,
    "strip_accents": "unicode",
}
CHAR_FEATURE_WEIGHT = 0.25
SVM_CONFIG = {"C": 1.0, "max_iter": 12000, "random_state": RANDOM_STATE}


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data_unlabeled.csv")
    if not {"text", "target"}.issubset(train_df.columns):
        raise ValueError("train_data.csv 必须有 text 和 target 两列")
    if list(test_df.columns) != ["text"]:
        raise ValueError("test_data_unlabeled.csv 必须只含 text 列")

    train_text = train_df["text"].fillna("").map(clean_text)
    test_text = test_df["text"].fillna("").map(clean_text)

    word_vectorizer = TfidfVectorizer(**WORD_TFIDF_CONFIG)
    char_vectorizer = TfidfVectorizer(**CHAR_TFIDF_CONFIG)
    X_train_word = word_vectorizer.fit_transform(train_text)
    X_test_word = word_vectorizer.transform(test_text)
    X_train_char = char_vectorizer.fit_transform(train_text)
    X_test_char = char_vectorizer.transform(test_text)

    X_train = hstack([X_train_word, X_train_char * CHAR_FEATURE_WEIGHT], format="csr")
    X_test = hstack([X_test_word, X_test_char * CHAR_FEATURE_WEIGHT], format="csr")
    model = LinearSVC(**SVM_CONFIG)
    model.fit(X_train, train_df["target"])
    predictions = model.predict(X_test)

    output_path = BASE_DIR / "predictions.csv"
    pd.DataFrame(predictions).to_csv(output_path, index=False, header=False)

    metadata = {
        "purpose": "final test-set prediction after validation-based model selection",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "word_tfidf": {**WORD_TFIDF_CONFIG, "ngram_range": list(WORD_TFIDF_CONFIG["ngram_range"])},
        "char_tfidf": {**CHAR_TFIDF_CONFIG, "ngram_range": list(CHAR_TFIDF_CONFIG["ngram_range"])},
        "char_feature_weight": CHAR_FEATURE_WEIGHT,
        "linear_svc": SVM_CONFIG,
        "validation_result_used_for_selection": {"accuracy": 0.8948439620081411, "macro_f1": 0.895023216770858},
        "output_format": "one predicted label per row; no header",
    }
    (RESULTS_DIR / "final_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已用全部 {len(train_df)} 条有标签文本训练最终模型。")
    print(f"词特征数：{X_train_word.shape[1]}；字符特征数：{X_train_char.shape[1]}；合并后：{X_train.shape[1]}。")
    print(f"已生成 {output_path.name}：{len(predictions)} 行、无表头、每行一个预测类别。")


if __name__ == "__main__":
    main()
