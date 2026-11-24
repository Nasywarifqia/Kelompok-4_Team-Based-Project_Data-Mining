# Early Warning System Kelulusan Mahasiswa Berbasis Data Mining

Proyek **Team-Based Project (TBP)** Mata Kuliah Data Mining — S1 Informatika Universitas Sebelas Maret (Semester 5).

---

## 👥 Tim Pengembang (Kelompok 4)

| Nama Anggota | Peran / Tanggung Jawab Tahapan TBP | Algoritma yang Diteliti |
|---|---|---|
| **Laura** | **Project Lead & Data Prep** (Tahap 1 & 2: Problem Definition & Data Cleaning) | Decision Tree (CART) |
| **Wawa** | **Feature Engineer** (Tahap 3: Preprocessing, Encoding, & Train-Test Split) | Random Forest |
| **Gisca** | **Evaluator & Benchmark** (Tahap 5: Evaluasi Komparatif & Visualisasi Metrik) | Naive Bayes (Gaussian) |
| **Irfan** | **Deployment Engineer** (Tahap 6: Prototipe Dashboard Streamlit) | K-Nearest Neighbors (KNN) |
| **Rapip** | **Business Analyst & Report** (Tahap 7: Laporan Akhir, PPT, & Rekomendasi Bisnis) | Logistic Regression |

---

## 📁 Struktur Repositori

```text
├── dataset_final_tbp/        # Skrip preprocessing & output dataset bersama (X_train, X_test, dll.)
│   ├── output/               # X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.pkl
│   ├── preprocessing.py      # Skrip pipeline pembersihan dan split data
│   └── README.txt            # Dokumentasi dataset & checksum MD5
├── notebooks/                # Tempat pengerjaan notebook modeling masing-masing anggota
│   ├── 01_decision_tree_laura.ipynb
│   ├── 02_random_forest_wawa.ipynb
│   ├── 03_naive_bayes_gisca.ipynb
│   ├── 04_knn_irfan.ipynb
│   ├── 05_logistic_regression_rapip.ipynb
│   └── README.md
├── models/                   # Berkas model tersimpan (.pkl) dari ke-5 algoritma
│   └── README.md
├── evaluation/               # Skrip evaluasi komparatif, grafik ROC-AUC, dan leaderboard gabungan
│   └── README.md
├── app/                      # Aplikasi web dashboard interaktif (Streamlit)
│   ├── app.py                # Source code dashboard
│   ├── requirements.txt      # Dependensi Python
│   └── README.md
└── docs/                     # Dokumen laporan resmi, proposal, dan slide presentasi
    └── README.md
```

---

## 🚀 Alur Kerja Pengerjaan (Workflow)

1. **Fase 1 (Selesai):** Pembersihan dataset bersama dan pembagian train-test (80:20) di folder `dataset_final_tbp/output/`.
2. **Fase 2 (Sedang Berjalan):** Setiap anggota melatih algoritmanya masing-masing di folder `notebooks/` menggunakan dataset bersama, lalu menyimpan modelnya ke folder `models/`.
3. **Fase 3:** Gisca merekapitulasi metrik seluruh model di folder `evaluation/` untuk menentukan *Best Model*.
4. **Fase 4:** Irfan menanamkan *Best Model* ke dalam dashboard Streamlit di folder `app/`, dan Rapip menyusun Laporan Akhir serta Slide Presentasi di folder `docs/`.

---

## 💻 Cara Menjalankan Dashboard Streamlit

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
