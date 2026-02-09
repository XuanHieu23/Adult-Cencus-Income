#!/usr/bin/env python3
# preprocessing.py — Adult Income preprocessing utilities

import os
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_series_counts(s: pd.Series, out_csv: str) -> None:
    df = pd.DataFrame(
        {
            "count": s.value_counts(dropna=False),
            "proportion": s.value_counts(normalize=True, dropna=False),
        }
    )
    df.to_csv(out_csv)


def normalize_adult_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common Adult column variants into canonical names.

    Canonical names used by this project:
      - education-num
      - capital.gain
      - capital.loss
      - hours.per.week
      - marital.status
      - native.country
      - fnlwgt
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        # education-num variants
        "education.num": "education-num",
        "education_num": "education-num",

        # capital gain/loss variants
        "capital-gain": "capital.gain",
        "capital_gain": "capital.gain",
        "capital-loss": "capital.loss",
        "capital_loss": "capital.loss",

        # hours per week variants
        "hours-per-week": "hours.per.week",
        "hours_per_week": "hours.per.week",

        # marital status variants
        "marital-status": "marital.status",
        "marital_status": "marital.status",

        # native country variants
        "native-country": "native.country",
        "native_country": "native.country",

        # rare typo
        "fnlwg": "fnlwgt",
    }

    existing = set(df.columns)
    final_map = {k: v for k, v in rename_map.items() if k in existing}
    return df.rename(columns=final_map)


def safe_strip_categoricals(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(str).str.strip()
    return df


def validate_columns(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    label_col: str = "income",
) -> None:
    missing = [c for c in (categorical_cols + numeric_cols + [label_col]) if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV does not match expected Adult columns after normalization. Missing: "
            + ", ".join(missing)
            + "\nAvailable columns: " + ", ".join(df.columns)
        )


def load_and_clean_adult(
    csv_path: str,
    categorical_cols: List[str],
    numeric_cols: List[str],
    label_col: str = "income",
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_adult_columns(df)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'. Please check your CSV.")

    validate_columns(df, categorical_cols, numeric_cols, label_col=label_col)

    # strip spaces (important for Adult variants like " Private")
    df = safe_strip_categoricals(df, categorical_cols + [label_col])

    # replace '?' tokens -> Unknown (keep rows)
    for c in ["workclass", "occupation", "native.country"]:
        if c in df.columns:
            df[c] = df[c].replace("?", "Unknown")

    # drop duplicates if requested
    if drop_duplicates and int(df.duplicated().sum()) > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    return df


def split_xy_train_test(
    df: pd.DataFrame,
    label_col: str = "income",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop"
    )