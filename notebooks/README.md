# Panduan Pengerjaan Modeling (Fase 2)

Folder ini digunakan oleh masing-masing anggota untuk menaruh notebook eksperimen modeling algoritma yang diteliti.

## Pembagian File Notebook
1. `01_decision_tree_laura.ipynb` - **Laura** (Decision Tree CART)
2. `02_random_forest_wawa.ipynb` - **Wawa** (Random Forest)
3. `03_naive_bayes_gisca.ipynb` - **Gisca** (Naive Bayes Gaussian)
4. `04_knn_irfan.ipynb` - **Irfan** (K-Nearest Neighbors)
5. `05_logistic_regression_rapip.ipynb` - **Rapip** (Logistic Regression)

## Aturan Membaca Data
Gunakan data bersama yang sudah bersih di folder `dataset_final_tbp/output/`. Jangan membagi atau mengacak ulang dataset.

```python
import pandas as pd
import joblib

# Load data bersama
X_train = pd.read_csv('../dataset_final_tbp/output/X_train.csv')
y_train = pd.read_csv('../dataset_final_tbp/output/y_train.csv').squeeze()
X_test = pd.read_csv('../dataset_final_tbp/output/X_test.csv')
y_test = pd.read_csv('../dataset_final_tbp/output/y_test.csv').squeeze()

# Catatan khusus KNN dan Logistic Regression:
# Gunakan scaler yang sudah di-fit pada X_train:
scaler = joblib.load('../dataset_final_tbp/output/scaler.pkl')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Luaran yang Wajib Disimpan
1. Simpan model terlatih ke folder `../models/<nama_model>.pkl` menggunakan `joblib.dump()`.
2. Catat nilai metrik pada data uji (`X_test`):
   - Accuracy
   - Precision (kelas 0 & kelas 1)
   - Recall (prioritas kelas 0 - Dropout)
   - F1-Score
   - ROC-AUC Score
   - Waktu pelatihan (detik)
3. Setorkan nilai metrik tersebut ke **Gisca** untuk rekapitulasi tabel evaluasi gabungan di folder `evaluation/`.
