# logistic_regression_quick_test.py：逻辑回归的快速对照实验。

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(DATA_DIR / 'train_split_clean.csv').fillna('')
valid_df = pd.read_csv(DATA_DIR / 'valid_split_clean.csv').fillna('')

results = []
for ngram_range in [(1,1), (1,2)]:
    for max_features in [50000, 80000]:
        for C in [0.5, 1.0, 2.0, 5.0, 10.0]:
            vec = TfidfVectorizer(
                max_features=max_features,
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                lowercase=False,
                strip_accents='unicode',
                ngram_range=ngram_range,
            )
            X_train = vec.fit_transform(train_df['text_clean'])
            X_valid = vec.transform(valid_df['text_clean'])
            model = LogisticRegression(C=C, solver='lbfgs', max_iter=4000, random_state=42)
            model.fit(X_train, train_df['target'])
            pred = model.predict(X_valid)
            results.append({
                'ngram_range': list(ngram_range),
                'max_features': max_features,
                'C': C,
                'valid_accuracy': float(accuracy_score(valid_df['target'], pred)),
                'valid_macro_f1': float(f1_score(valid_df['target'], pred, average='macro')),
                'classification_report': classification_report(valid_df['target'], pred, zero_division=0),
            })

ranked = sorted(results, key=lambda r: (r['valid_macro_f1'], r['valid_accuracy']), reverse=True)
print(json.dumps(ranked[:10], ensure_ascii=False, indent=2))
(RESULTS_DIR / 'logistic_regression_quick_test.json').write_text(json.dumps({'ranked_results': ranked}, ensure_ascii=False, indent=2), encoding='utf-8')
