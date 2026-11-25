# Sistem Deteksi Situs Web Phishing Berbasis Data Mining

Proyek **Team-Based Project (TBP)** Mata Kuliah Data Mining — S1 Informatika Universitas Sebelas Maret (Semester 5).

---

## 👥 Tim Pengembang (Kelompok 4)

| Nama Anggota | Peran / Tanggung Jawab Tahapan TBP | Algoritma yang Diteliti |
|---|---|---|
| **Laura** | **Project Lead & Data Prep** (Tahap 1 & 2: Problem Definition & Data Cleaning) | Decision Tree (CART) |
| **Wawa** | **Feature Engineer** (Tahap 3: Preprocessing, Deduplikasi, & Train-Test Split) | Random Forest |
| **Gisca** | **Evaluator & Benchmark** (Tahap 5: Evaluasi Komparatif & Visualisasi Metrik) | Naive Bayes |
| **Irfan** | **Deployment Engineer** (Tahap 6: Prototipe Dashboard Streamlit) | K-Nearest Neighbors (KNN) |
| **Rapip** | **Business Analyst & Report** (Tahap 7: Laporan Akhir, PPT, & Rekomendasi) | Logistic Regression |

---

## 📊 Dataset

* **Sumber:** [UCI Machine Learning Repository — Phishing Websites (ID 327)](https://archive.ics.uci.edu/dataset/327/phishing+websites)
* **Data Mentah:** 11.055 baris, 30 fitur indikator keamanan (URL, SSL, domain, HTML/JS), 0 *missing value*.
* **Data Bersih (Setelah Deduplikasi):** 5.849 baris unik (`2.830` Legitimate / `0`, `3.019` Phishing / `1`) untuk mencegah kebocoran data (*data leakage*) antara *train* dan *test*.

---

## 📁 Struktur Repositori

```text
├── dataset/                  # Dataset mentah UCI, preprocessing.py, dan hasil split di output/
│   ├── output/               # X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.pkl
│   ├── phishing_websites.csv # Dataset mentah (11.055 baris)
│   ├── preprocessing.py      # Skrip cleaning, deduplikasi, dan split 80:20
│   └── README.txt            # Dokumentasi preprocessing & checksum MD5
├── notebooks/                # Tempat pengerjaan notebook modeling masing-masing anggota
│   └── README.md
├── models/                   # Berkas model tersimpan (.pkl) dari ke-5 algoritma
│   └── README.md
├── evaluation/               # Skrip evaluasi komparatif, grafik ROC-AUC, dan leaderboard gabungan
│   └── README.md
├── app/                      # Aplikasi web dashboard interaktif (Streamlit)
│   ├── app.py                # Source code dashboard deteksi phishing
│   ├── requirements.txt      # Dependensi Python
│   └── README.md
└── docs/                     # Dokumen laporan resmi, proposal, dan slide presentasi
    └── README.md
```

---

## 💻 Cara Menjalankan Dashboard Streamlit

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
