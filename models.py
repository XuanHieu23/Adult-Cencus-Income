#!/usr/bin/env python3
# models.py — Main runner: Adult Income (DT/RF/KNN/SVM) + EDA + KNN CV + timing + outputs

import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from preprocessing import (
    ensure_dir,
    save_series_counts,
    load_and_clean_adult,
    split_xy_train_test,
    build_preprocessor,
)
from eda import run_eda, save_pca_2d_plot_from_preprocessed


def time_fit_predict(pipe: Pipeline, X_train, y_train, X_test) -> tuple[float, float]:
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pipe.predict(X_test)
    pred_t = time.perf_counter() - t0

    return float(train_t), float(pred_t)


def evaluate_and_save(
    name: str,
    pipe: Pipeline,
    X_train, y_train,
    X_test, y_test,
    out_dir: str
) -> dict:
    # timing
    train_t, pred_t = time_fit_predict(pipe, X_train, y_train, X_test)

    # predict + metrics
    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    err = 1.0 - acc

    # save classification report
    rep_dict = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(rep_dict).T.to_csv(os.path.join(out_dir, f"{name}_classification_report.csv"))

    # save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot(values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_confusion_matrix.png"), dpi=200)
    plt.close()

    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print(f"Test Accuracy: {acc:.6f}")
    print(f"Classification Error (1-Accuracy): {err:.6f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "model": name,
        "test_accuracy": acc,
        "classification_error": err,
        "train_time_sec": train_t,
        "predict_time_sec": pred_t
    }


def knn_kfold_select_k(
    preprocessor,
    X_train, y_train,
    out_dir: str,
    cv_folds: int = 5,
    seed: int = 42,
    k_candidates=None
) -> pd.DataFrame:
    if k_candidates is None:
        k_candidates = list(range(1, 26, 2))  # odd k

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows = []

    for k in k_candidates:
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=k)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        rows.append({
            "k": k,
            "cv_mean_accuracy": float(np.mean(scores)),
            "cv_std_accuracy": float(np.std(scores)),
            "cv_mean_error": float(1.0 - np.mean(scores)),
        })

    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "knn_kfold_cv_k_selection.csv"), index=False)

    # plot error vs k
    plt.figure(figsize=(6, 4))
    plt.plot(df["k"], df["cv_mean_error"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Mean CV classification error")
    plt.title(f"KNN: {cv_folds}-fold CV error vs k (train)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "knn_cv_error_vs_k.png"), dpi=200)
    plt.close()

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="adult.csv", help="Adult Income CSV (must contain 'income').")
    parser.add_argument("--out", default="outputs_final_project", help="Output folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--make_eda", action="store_true", help="Save EDA plots")
    parser.add_argument("--make_pca", action="store_true", help="Optional: save PCA 2D plot")
    args = parser.parse_args()

    ensure_dir(args.out)
    eda_dir = os.path.join(args.out, "eda")
    ensure_dir(eda_dir)

    # Columns (canonical after normalization)
    categorical_cols = [
        "workclass", "education", "marital.status", "occupation",
        "relationship", "race", "sex", "native.country"
    ]
    numeric_cols = [
        "age", "fnlwgt", "education-num",
        "capital.gain", "capital.loss", "hours.per.week"
    ]

    # -----------------------------
    # A-B) Load + Clean
    # -----------------------------
    df = load_and_clean_adult(
        csv_path=args.data,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        label_col="income",
        drop_duplicates=True
    )

    print("=== BASIC INFORMATION ===")
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)

    # quick checks (for your log)
    print("\n=== CHECK DUPLICATES ===")
    print("Duplicated rows after cleaning:", int(df.duplicated().sum()))

    print("\n=== LABEL DISTRIBUTION (after cleaning) ===")
    print(df["income"].value_counts())
    print(df["income"].value_counts(normalize=True))

    # -----------------------------
    # C) Split 80/20 stratify
    # -----------------------------
    X_train, X_test, y_train, y_test = split_xy_train_test(
        df, label_col="income", test_size=args.test_size, seed=args.seed
    )

    print("\n=== TRAIN / TEST SPLIT ===")
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)
    print("\nTrain label distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nTest label distribution:")
    print(y_test.value_counts(normalize=True))

    # Save label distributions (csv)
    save_series_counts(df["income"], os.path.join(args.out, "label_distribution_full.csv"))
    save_series_counts(y_train, os.path.join(args.out, "label_distribution_train.csv"))
    save_series_counts(y_test, os.path.join(args.out, "label_distribution_test.csv"))

    # -----------------------------
    # D) Preprocess
    # -----------------------------
    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    # -----------------------------
    # EDA (plots + correlation heatmap)
    # -----------------------------
    if args.make_eda:
        run_eda(
            df=df,
            y=df["income"],
            out_dir=eda_dir,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            top_k=10
        )

    # Optional PCA on preprocessed matrix
    if args.make_pca:
        # fit_transform on TRAIN to avoid leakage; then show PCA on train only (clean)
        X_train_pre = preprocessor.fit_transform(X_train)
        save_pca_2d_plot_from_preprocessed(
            X_preprocessed=X_train_pre,
            y=y_train,
            out_path=os.path.join(eda_dir, "pca2d_train_preprocessed.png"),
            max_points=6000,
            seed=args.seed
        )

    # -----------------------------
    # Models (DT baseline, RF main, KNN compare, SVM linear/RBF)
    # -----------------------------
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=args.seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=args.seed, n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11),
        "SVM_Linear": SVC(kernel="linear", class_weight="balanced", random_state=args.seed),
        "SVM_RBF": SVC(kernel="rbf", class_weight="balanced", random_state=args.seed),
    }

    results = []
    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", clf),
        ])
        res = evaluate_and_save(name, pipe, X_train, y_train, X_test, y_test, args.out)
        results.append(res)

    results_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    results_df.to_csv(os.path.join(args.out, "model_comparison_testset.csv"), index=False)

    # timing file (separate, if you want)
    results_df[["model", "train_time_sec", "predict_time_sec"]].to_csv(
        os.path.join(args.out, "model_timing.csv"), index=False
    )

    # -----------------------------
    # KNN k-fold choose best k
    # -----------------------------
    knn_cv_df = knn_kfold_select_k(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        out_dir=args.out,
        cv_folds=args.cv_folds,
        seed=args.seed,
        k_candidates=list(range(1, 26, 2))
    )
    best_knn = knn_cv_df.loc[knn_cv_df["cv_mean_accuracy"].idxmax()]

    # -----------------------------
    # SUMMARY.txt
    # -----------------------------
    with open(os.path.join(args.out, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write("Adult Income Final Project - Summary\n\n")
        f.write(f"Data file: {args.data}\n")
        f.write(f"Total samples (after cleaning): {df.shape[0]}\n")
        f.write(f"Features: {df.shape[1]-1} | Label: income\n")
        f.write(f"Train/Test split: {int((1-args.test_size)*100)}/{int(args.test_size*100)} (stratified)\n\n")

        f.write("Model comparison on test set:\n")
        f.write(results_df[["model", "test_accuracy", "classification_error"]].to_string(index=False))
        f.write("\n\n")

        f.write("Running time (seconds):\n")
        f.write(results_df[["model", "train_time_sec", "predict_time_sec"]].to_string(index=False))
        f.write("\n\n")

        f.write(
            f"Best KNN by {args.cv_folds}-fold CV (train): "
            f"k={int(best_knn['k'])}, "
            f"mean_acc={best_knn['cv_mean_accuracy']:.4f}, "
            f"std={best_knn['cv_std_accuracy']:.4f}, "
            f"mean_error={best_knn['cv_mean_error']:.4f}\n"
        )

    print("\nALL DONE ✅")
    print(f"Outputs saved to: {args.out}")
    if args.make_eda:
        print(f"EDA saved to: {eda_dir}")


if __name__ == "__main__":
    main()