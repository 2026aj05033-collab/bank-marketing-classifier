"""
Shared preprocessing for the Bank Marketing (term-deposit) project.

Both the training script (model/train_models.py) and the Streamlit app (app.py)
import from here, so an uploaded test CSV is transformed exactly the same way
the training data was.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "y"

# `duration` (length of the last call) is deliberately excluded: it is only known
# after the call has ended, so using it would leak the answer (see UCI notes).
LEAKY_COLUMNS = ["duration"]

# The 19 raw input features a test CSV must contain (plus the target column y).
RAW_FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp_var_rate", "cons_price_idx", "cons_conf_idx",
    "euribor3m", "nr_employed",
]

# Columns after feature engineering
NUMERIC_COLS = [
    "age", "campaign", "days_since_prev", "previous", "emp_var_rate",
    "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
    "loan_burden",
]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan", "contact",
    "month", "day_of_week", "poutcome", "prev_contacted", "age_group",
]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning shared by training and the app."""
    df = df.copy()
    df.columns = [c.strip().replace(".", "_") for c in df.columns]
    return df


def encode_target(series: pd.Series) -> np.ndarray:
    """Map yes/no (or 1/0) target to 1/0."""
    s = series.astype(str).str.strip().str.lower()
    return s.map({"yes": 1, "no": 0, "1": 1, "0": 0}).astype(int).to_numpy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few domain-driven features on top of the raw columns."""
    out = df.copy()

    # pdays = 999 means "never contacted in a previous campaign" -> split into
    # a flag plus a cleaned numeric column (999 would otherwise distort scaling)
    out["prev_contacted"] = np.where(out["pdays"] == 999, "no", "yes")
    out["days_since_prev"] = out["pdays"].where(out["pdays"] != 999, -1)

    # Number of existing loans (housing + personal) the client holds
    out["loan_burden"] = (out["housing"] == "yes").astype(int) + (out["loan"] == "yes").astype(int)

    out["age_group"] = pd.cut(
        out["age"], bins=[0, 25, 35, 45, 60, 200],
        labels=["<=25", "26-35", "36-45", "46-60", "60+"],
    ).astype(str)

    return out[NUMERIC_COLS + CATEGORICAL_COLS]


def build_preprocessor() -> Pipeline:
    """Feature engineering + imputation + scaling + one-hot encoding."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_features, validate=False)),
        ("columns", columns),
    ])
