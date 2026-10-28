"""
Train and evaluate 5 classifiers on the UCI Bank Marketing dataset.

Run from the project root:
    python model/train_models.py

Outputs
-------
model/<name>.joblib      full pipeline (preprocessing + classifier) per model
model/metrics.json       test-set metrics + tuned hyper-parameters
model/metrics_table.md   same metrics as a markdown table (for the README)
test_data.csv            held-out 20% test split (19 raw features + y)
"""

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess import (  # noqa: E402
    RAW_FEATURES, TARGET, build_preprocessor, clean_raw, encode_target,
)

RANDOM_STATE = 42
DATA_PATH = ROOT / "data" / "bank_marketing.csv"
MODEL_DIR = ROOT / "model"

# Model name -> (estimator, small hyper-parameter grid tuned with 5-fold CV).
# Only ~11% of clients subscribe, so class_weight="balanced" is offered to the
# models that support it and CV decides whether to use it.
MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        {"clf__C": [0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"clf__max_depth": [5, 8, 12],
         "clf__min_samples_leaf": [20, 50, 100],
         "clf__class_weight": [None, "balanced"]},
    ),
    "KNN": (
        KNeighborsClassifier(n_jobs=-1),
        {"clf__n_neighbors": [11, 25, 51], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-5, 1e-2, 1e-1]},
    ),
    "Support Vector Machine": (
        # Linear SVM scales well to 33k rows; it has no native probabilities,
        # so it is wrapped in CalibratedClassifierCV to allow AUC computation.
        CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, max_iter=5000), ensemble=False, cv=3
        ),
        {"clf__estimator__C": [0.01, 0.1, 1.0],
         "clf__estimator__class_weight": [None, "balanced"]},
    ),
}

FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = clean_raw(pd.read_csv(DATA_PATH))
    print(f"Loaded {df.shape[0]} rows; using {len(RAW_FEATURES)} input features "
          f"('duration' excluded as leakage)")

    X = df[RAW_FEATURES]
    y = encode_target(df[TARGET])
    print(f"Positive class (subscribed) rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Save the untouched test split so the Streamlit app can be evaluated on it
    test_df = X_test.copy()
    test_df[TARGET] = pd.Series(y_test, index=X_test.index).map({1: "yes", 0: "no"})
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows -> test_data.csv\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, best_params = {}, {}

    for name, (estimator, grid) in MODELS.items():
        start = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]
        results[name] = evaluate(y_test, y_pred, y_prob)
        best_params[name] = {
            k.replace("clf__", "").replace("estimator__", ""): v
            for k, v in search.best_params_.items()
        }

        joblib.dump(best, MODEL_DIR / FILE_NAMES[name], compress=3)
        m = results[name]
        print(f"{name:<24} acc={m['Accuracy']:.4f} auc={m['AUC']:.4f} "
              f"f1={m['F1']:.4f} mcc={m['MCC']:.4f}  ({time.time() - start:.0f}s) "
              f"params={best_params[name]}")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"metrics": results, "best_params": best_params,
                   "model_files": FILE_NAMES}, f, indent=2)

    table = pd.DataFrame(results).T.round(4)
    table.index.name = "ML Model Name"
    (MODEL_DIR / "metrics_table.md").write_text(table.to_markdown())
    print("\n" + table.to_string())


if __name__ == "__main__":
    main()
