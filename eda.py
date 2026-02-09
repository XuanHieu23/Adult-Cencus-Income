#!/usr/bin/env python3
# eda.py — EDA + correlation heatmap + optional PCA visualization (no seaborn)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.decomposition import PCA


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_numeric_histograms(df: pd.DataFrame, out_dir: str, numeric_cols: List[str]) -> None:
    ensure_dir(out_dir)
    for col in numeric_cols:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        x = df[col].dropna()
        plt.hist(x, bins=40)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{col}.png"), dpi=200)
        plt.close()


def save_categorical_bars(df: pd.DataFrame, out_dir: str, categorical_cols: List[str], top_k: int = 10) -> None:
    ensure_dir(out_dir)
    for col in categorical_cols:
        if col not in df.columns:
            continue
        vc = df[col].astype(str).value_counts().head(top_k)
        plt.figure(figsize=(8, 4))
        plt.bar(vc.index.astype(str), vc.values)
        plt.title(f"Top {top_k} categories of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"bar_{col}.png"), dpi=200)
        plt.close()


def save_label_distribution_plot(y: pd.Series, out_path: str, title: str = "Label distribution") -> None:
    vc = y.value_counts()
    plt.figure(figsize=(5, 4))
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_correlation_heatmap_numeric(df: pd.DataFrame, out_path: str, numeric_cols: List[str]) -> None:
    """
    Correlation heatmap for numeric features only (stable + interpretable).
    """
    cols = [c for c in numeric_cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.colorbar(label="Correlation")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pca_2d_plot_from_preprocessed(
    X_preprocessed,
    y: pd.Series,
    out_path: str,
    max_points: int = 6000,
    seed: int = 42,
) -> None:
    """
    Optional PCA 2D visualization on preprocessed matrix (after one-hot + scaling).
    Works for both sparse/dense.
    """
    rng = np.random.default_rng(seed)
    n = X_preprocessed.shape[0]
    idx = np.arange(n)
    if n > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    X_sub = X_preprocessed[idx]
    y_sub = y.iloc[idx].astype(str).values

    # convert sparse -> dense for PCA (subsample keeps memory safe)
    try:
        X_sub = X_sub.toarray()
    except Exception:
        pass

    pca = PCA(n_components=2, random_state=seed)
    X2 = pca.fit_transform(X_sub)

    plt.figure(figsize=(6, 5))
    classes = np.unique(y_sub)
    for cls in classes:
        mask = (y_sub == cls)
        plt.scatter(X2[mask, 0], X2[mask, 1], s=8, alpha=0.6, label=str(cls))
    plt.title("PCA (2D) of Preprocessed Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_eda(
    df: pd.DataFrame,
    y: pd.Series,
    out_dir: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    top_k: int = 10,
) -> None:
    ensure_dir(out_dir)
    save_numeric_histograms(df, out_dir, numeric_cols)
    save_categorical_bars(df, out_dir, categorical_cols, top_k=top_k)
    save_label_distribution_plot(y, os.path.join(out_dir, "label_distribution_bar.png"),
                                 title="Income label distribution")
    save_correlation_heatmap_numeric(df, os.path.join(out_dir, "corr_heatmap_numeric.png"), numeric_cols)