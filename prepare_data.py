# prepare_data.py：划分训练子集与验证集，并清洗新闻文本。

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TRAIN_PATH = DATA_DIR / "train_data.csv"
TEST_PATH = DATA_DIR / "test_data_unlabeled.csv"


def clean_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    separator = re.search(r"\n[ \t]*\n", text)
    if separator:
        text = text[separator.end():]

    text = re.sub(r"(?m)^[ \t]*>.*$", " ", text)
    text = re.sub(r"(?im)^[ \t]*Lines:[^\n]*$", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def add_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["text"] = result["text"].fillna("")
    result["text_clean"] = result["text"].map(clean_text)
    return result


def build_exploration_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_lengths = train_df["text"].astype(str).str.len()
    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_columns": list(train_df.columns),
        "test_columns": list(test_df.columns),
        "missing_values": train_df.isna().sum().to_dict(),
        "duplicate_train_text": int(train_df["text"].duplicated().sum()),
        "label_distribution": {
            str(label): int(count)
            for label, count in train_df["target"].value_counts().sort_index().items()
        },
        "text_length": {
            "min": int(train_lengths.min()),
            "median": float(train_lengths.median()),
            "mean": float(train_lengths.mean()),
            "max": int(train_lengths.max()),
        },
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
    }


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required_train_columns = {"text", "target"}
    if not required_train_columns.issubset(train_df.columns):
        raise ValueError("训练集必须包含 text 和 target 列")
    if list(test_df.columns) != ["text"]:
        raise ValueError("测试集必须只包含 text 列")

    report = build_exploration_report(train_df, test_df)
    with (RESULTS_DIR / "exploration_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    train_split, valid_split = train_test_split(
        train_df,
        test_size=VALIDATION_SIZE,
        stratify=train_df["target"],
        random_state=RANDOM_STATE,
    )

    add_clean_text(train_split).to_csv(DATA_DIR / "train_split_clean.csv", index=False)
    add_clean_text(valid_split).to_csv(DATA_DIR / "valid_split_clean.csv", index=False)
    add_clean_text(test_df).to_csv(DATA_DIR / "test_clean.csv", index=False)

    train_split[["text", "target"]].to_csv(DATA_DIR / "train_split_raw.csv", index=False)
    valid_split[["text", "target"]].to_csv(DATA_DIR / "valid_split_raw.csv", index=False)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"训练子集：{len(train_split)} 条")
    print(f"验证集：{len(valid_split)} 条")
    print(f"测试集：{len(test_df)} 条（未参与划分）")
    print("已生成清洗后的 train_split_clean.csv、valid_split_clean.csv、test_clean.csv")


if __name__ == "__main__":
    main()
