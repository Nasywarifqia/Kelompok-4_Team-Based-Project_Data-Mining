"""Verifikasi konsistensi dan signifikansi statistik 5 model menggunakan Stratified 5-Fold CV dan Paired t-test (koreksi Holm-Bonferroni)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "output"
OUT_DIR = ROOT / "evaluation" / "komparasi"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def holm_bonferroni(p_values):
    """Koreksi p-value bertahap Holm-Bonferroni untuk perbandingan berganda."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, p * (m - rank))
        running_max = max(running_max, val)
        adjusted[orig_idx] = running_max
    return adjusted


def build_models(n_features):
    categories = [[-1, 0, 1]] * n_features
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Decision Tree (CART)": DecisionTreeClassifier(
            criterion="gini", max_depth=12, min_samples_leaf=1, min_samples_split=10, random_state=42
        ),
        "Naive Bayes": make_pipeline(OrdinalEncoder(categories=categories), CategoricalNB(alpha=2.0)),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11, n_jobs=-1)),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000, random_state=42)
        ),
    }


def main():
    X = pd.concat([pd.read_csv(DATA_DIR / "X_train.csv"), pd.read_csv(DATA_DIR / "X_test.csv")], ignore_index=True)
    y = pd.concat([pd.read_csv(DATA_DIR / "y_train.csv"), pd.read_csv(DATA_DIR / "y_test.csv")], ignore_index=True).squeeze()

    models = build_models(X.shape[1])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_records = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        for name, base_model in models.items():
            clf = clone(base_model)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)[:, 1]
            fold_records.append({
                "Fold": fold_idx,
                "Model": name,
                "Accuracy": accuracy_score(y_val, y_pred),
                "Recall (Phishing)": recall_score(y_val, y_pred, pos_label=1),
                "F1-Score (Phishing)": f1_score(y_val, y_pred, pos_label=1),
                "ROC-AUC": roc_auc_score(y_val, y_proba),
            })

    folds_df = pd.DataFrame(fold_records)
    folds_df.round(4).to_csv(OUT_DIR / "cv_5fold_per_fold.csv", index=False)

    metrics = ["Accuracy", "Recall (Phishing)", "F1-Score (Phishing)", "ROC-AUC"]
    summary_rows = []
    for name in models:
        sub = folds_df[folds_df["Model"] == name]
        row = {"Model": name}
        for m in metrics:
            row[f"{m} Mean"] = round(float(sub[m].mean()), 4)
            row[f"{m} Std"] = round(float(sub[m].std(ddof=1)), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("F1-Score (Phishing) Mean", ascending=False).reset_index(drop=True)
    summary_df.to_csv(OUT_DIR / "cv_5fold_summary.csv", index=False)

    # Uji t-test berpasangan + koreksi Holm terhadap model terbaik per metrik (F1 & Recall)
    stat_rows = []
    for target_metric in ["F1-Score (Phishing)", "Recall (Phishing)"]:
        best_model = summary_df.sort_values(f"{target_metric} Mean", ascending=False).iloc[0]["Model"]
        best_scores = folds_df[folds_df["Model"] == best_model].sort_values("Fold")[target_metric].to_numpy()
        others = [m for m in models if m != best_model]
        raw_p, t_vals = [], []
        for other in others:
            other_scores = folds_df[folds_df["Model"] == other].sort_values("Fold")[target_metric].to_numpy()
            t_stat, p_val = ttest_rel(best_scores, other_scores)
            t_vals.append(float(t_stat))
            raw_p.append(float(p_val))
        adj_p = holm_bonferroni(raw_p)
        for other, t_val, p_r, p_h in zip(others, t_vals, raw_p, adj_p):
            stat_rows.append({
                "Metrik": target_metric,
                "Model Terbaik": best_model,
                "Pembanding": other,
                "t_statistic": round(t_val, 4),
                "p_value_raw": round(p_r, 4),
                "p_value_holm": round(p_h, 4),
                "Signifikan (alpha=0.05)": "Ya" if p_h < 0.05 else "Tidak (Setara)",
            })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(OUT_DIR / "statistical_test_holm.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    order = summary_df["Model"].tolist()
    for ax, metric in zip(axes, ["F1-Score (Phishing)", "Recall (Phishing)"]):
        data = [folds_df[folds_df["Model"] == m][metric].values for m in order]
        ax.boxplot(data, tick_labels=order, showmeans=True)
        ax.set_title(f"Distribusi 5-Fold CV — {metric}")
        ax.set_ylabel("Skor")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cv_5fold_boxplot.png", dpi=150)
    plt.close()

    print("=== RINGKASAN 5-FOLD CV ===")
    print(summary_df.to_string(index=False))
    print("\n=== UJI T-TEST BERPASANGAN + KOREKSI HOLM ===")
    print(stat_df.to_string(index=False))


if __name__ == "__main__":
    main()
