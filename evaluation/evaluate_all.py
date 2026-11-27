"""
Evaluasi komparatif 5 model deteksi phishing (PIC: Gisca — Evaluator & Benchmark).

Cara pakai (dari folder root repo):
    python evaluation/evaluate_all.py

Script ini memuat setiap model yang SUDAH ada di folder models/, memprediksi
data uji yang sama (dataset/output/X_test.csv), lalu membuat:
    evaluation/komparasi/leaderboard.csv        tabel perbandingan
    evaluation/komparasi/leaderboard.md         tabel versi laporan
    evaluation/komparasi/bar_metrik.png         grafik batang metrik
    evaluation/komparasi/roc_semua_model.png    kurva ROC semua model
    evaluation/komparasi/confusion_matrix_semua.png
Model yang belum ada (mis. Decision Tree / Random Forest) dilewati otomatis,
jadi cukup jalankan ulang script ini setelah semua anggota menyetor model.
"""
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "output"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "evaluation" / "komparasi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sama dengan katalog di app/app.py (needs_scaling = pakai scaler.pkl)
CATALOG = [
    ("Decision Tree (CART)", "decision_tree_model.pkl", False, "Laura"),
    ("Random Forest", "random_forest_model.pkl", False, "Wawa"),
    ("Naive Bayes", "naive_bayes_model.pkl", False, "Gisca"),
    ("K-Nearest Neighbors", "knn_model.pkl", True, "Irfan"),
    ("Logistic Regression", "logistic_regression_model.pkl", True, "Rapip"),
]


def main():
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    scaler = joblib.load(DATA_DIR / "scaler.pkl")
    X_test_scaled = scaler.transform(X_test)

    rows, preds = [], {}
    for name, fname, needs_scaling, pic in CATALOG:
        path = MODEL_DIR / fname
        if not path.exists():
            print(f"[lewati] {name}: {fname} belum ada di models/")
            continue
        model = joblib.load(path)
        X = X_test_scaled if needs_scaling else X_test
        t0 = time.perf_counter()
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        pred_time = time.perf_counter() - t0
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "Model": name,
            "PIC": pic,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (Phishing)": precision_score(y_test, y_pred, pos_label=1),
            "Recall (Phishing)": recall_score(y_test, y_pred, pos_label=1),
            "F1-Score (Phishing)": f1_score(y_test, y_pred, pos_label=1),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "False Negative": int(fn),
            "False Positive": int(fp),
            "Waktu Prediksi (s)": pred_time,
        })
        preds[name] = (y_pred, y_proba)
        print(f"[ok] {name}")

    if not rows:
        print("Belum ada model di folder models/.")
        return

    # Urutan peringkat: Recall Phishing (prioritas), lalu F1, lalu ROC-AUC
    lb = (pd.DataFrame(rows)
          .sort_values(["Recall (Phishing)", "F1-Score (Phishing)", "ROC-AUC"], ascending=False)
          .reset_index(drop=True))
    lb.insert(0, "Peringkat", range(1, len(lb) + 1))
    lb.round(4).to_csv(OUT_DIR / "leaderboard.csv", index=False)

    md = lb.copy()
    for c in ["Accuracy", "Precision (Phishing)", "Recall (Phishing)", "F1-Score (Phishing)"]:
        md[c] = (md[c] * 100).map("{:.2f}%".format)
    md["ROC-AUC"] = md["ROC-AUC"].map("{:.4f}".format)
    md["Waktu Prediksi (s)"] = md["Waktu Prediksi (s)"].map("{:.4f}".format)
    try:
        md_text = md.to_markdown(index=False)
    except ImportError:
        headers = " | ".join(md.columns)
        sep = " | ".join("---" for _ in md.columns)
        body = "\n".join(" | ".join(str(v) for v in row) for row in md.itertuples(index=False))
        md_text = f"| {headers} |\n| {sep} |\n" + "\n".join(f"| {line} |" for line in body.splitlines())
    (OUT_DIR / "leaderboard.md").write_text(md_text, encoding="utf-8")

    print("\nLEADERBOARD (diurutkan Recall Phishing -> F1 -> ROC-AUC)")
    print(lb.round(4).to_string(index=False))
    best = lb.iloc[0]
    print(f"\nModel terbaik (Recall Phishing tertinggi): {best['Model']} ({best['PIC']})")

    # 1) Grafik batang metrik
    metrik = ["Accuracy", "Precision (Phishing)", "Recall (Phishing)", "F1-Score (Phishing)", "ROC-AUC"]
    ax = lb.set_index("Model")[metrik].plot(kind="bar", figsize=(11, 5), rot=0, ylim=(0.85, 1.0))
    ax.set_title("Perbandingan Performa Model Deteksi Phishing (Data Uji)")
    ax.set_ylabel("Skor")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bar_metrik.png", dpi=150)
    plt.close()

    # 2) Kurva ROC semua model
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in lb["Model"]:
        RocCurveDisplay.from_predictions(y_test, preds[name][1], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Tebak acak (0.5)")
    ax.set_title("Kurva ROC — Semua Model")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_semua_model.png", dpi=150)
    plt.close()

    # 3) Confusion matrix semua model
    n = len(lb)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    axes = [axes] if n == 1 else axes
    for ax, name in zip(axes, lb["Model"]):
        ConfusionMatrixDisplay.from_predictions(
            y_test, preds[name][0], display_labels=["Legit (0)", "Phishing (1)"],
            cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(name, fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix_semua.png", dpi=150)
    plt.close()
    print(f"\nHasil disimpan di {OUT_DIR}")


if __name__ == "__main__":
    main()
