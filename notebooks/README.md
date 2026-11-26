# Panduan Pengerjaan Modeling (Fase 2)

Folder ini digunakan oleh masing-masing anggota untuk menaruh notebook eksperimen modeling deteksi situs web *phishing*.

## Pembagian File Notebook
1. `01_decision_tree_laura.ipynb` - **Laura** (Decision Tree CART)
2. `02_random_forest_wawa.ipynb` - **Wawa** (Random Forest)
3. `03_naive_bayes_gisca.ipynb` - **Gisca** (Naive Bayes)
4. `04_knn_irfan.ipynb` - **Irfan** (K-Nearest Neighbors)
5. `05_logistic_regression_rapip.ipynb` - **Rapip** (Logistic Regression)

## Aturan Membaca Data
Gunakan data bersama di folder `dataset/output/`. Jangan membagi atau mengacak ulang dataset.

```python
import pandas as pd
import joblib

X_train = pd.read_csv('../dataset/output/X_train.csv')
y_train = pd.read_csv('../dataset/output/y_train.csv').squeeze()
X_test = pd.read_csv('../dataset/output/X_test.csv')
y_test = pd.read_csv('../dataset/output/y_test.csv').squeeze()

# Khusus KNN dan Logistic Regression:
scaler = joblib.load('../dataset/output/scaler.pkl')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Luaran yang Wajib Disimpan
1. Simpan model terlatih ke `../models/<nama_model>.pkl` menggunakan `joblib.dump()`.
2. Catat metrik evaluasi pada `X_test` (`0 = Legitimate`, `1 = Phishing`):
   - Accuracy
   - Precision (Phishing / kelas 1)
   - Recall (Phishing / kelas 1 — metrik prioritas)
   - F1-Score (Phishing / kelas 1)
   - ROC-AUC Score
   - Waktu pelatihan (detik)
3. Setorkan metrik ke `evaluation/` untuk rekapitulasi tabel komparasi oleh **Gisca**.
