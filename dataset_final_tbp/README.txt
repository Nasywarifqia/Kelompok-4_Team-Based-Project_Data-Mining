DATASET BERSAMA - EARLY WARNING SYSTEM KELULUSAN MAHASISWA
TBP Data Mining (Laura, Wawa, Agischa, Rizky, Irfan)
Disiapkan oleh: Agischa (Evaluator & Benchmark)

=====================================================================
1. ISI FOLDER
=====================================================================
preprocessing.py        Script yang menghasilkan semua file di output/
output/X_train.csv      Fitur data latih   (80.000 baris, 9 kolom)
output/X_test.csv       Fitur data uji     (20.000 baris, 9 kolom)
output/y_train.csv      Target data latih  (kolom: Status Kelulusan)
output/y_test.csv       Target data uji
output/encoding_map.json  Arti angka hasil encoding + parameter split
output/scaler.pkl       StandardScaler (fit di X_train) - HANYA untuk KNN & LogReg

=====================================================================
2. SUMBER DATA
=====================================================================
Kaggle: atifnirparmar/dataset-kelulusan-mahasiswa
File  : dataset_kelulusan_mahasiswa.csv (100.000 baris, 10 kolom)

Target "Status Kelulusan":  1 = Graduate (lulus),  0 = Dropout (tidak lulus)
Distribusi: 0 = 50.038 (50,04%), 1 = 49.962 (49,96%) -> seimbang

=====================================================================
3. LANGKAH CLEANING
=====================================================================
- Missing values : 0 di semua kolom -> tidak ada yang diisi/dihapus.
- Duplikat       : 0 baris -> tidak ada yang dihapus.
- Filter status  : target hanya berisi 0/1 -> tidak perlu filter.
- Konsistensi    : IPK, IPS Rata-rata, IPS Semester Akhir semua di rentang 0-4.
- Kolom          : SEMUA 9 fitur dipakai. Tidak ada kolom ID, tidak ada kolom
                   yang membocorkan target, dan tidak ada alasan kuat membuang kolom.
Jumlah baris tetap 100.000 (tidak ada data yang dihapus).

=====================================================================
4. LANGKAH PREPROCESSING
=====================================================================
a. Encoding (pemetaan manual, tidak di-fit dari data -> tanpa leakage)
   Pekerjaan Sambil Kuliah : Tidak=0, Ya=1
   Kategori Kehadiran      : Rendah=0, Sedang=1, Tinggi=2  (ordinal, ada urutan)
b. Pisah fitur (X) dan target (y = Status Kelulusan)
c. Train-test split: test_size=0.20, random_state=42, stratify=y
   -> proporsi kelas di train dan test sama (50,04% : 49,96%)
d. Scaling TIDAK diterapkan pada file CSV bersama, karena:
   - Decision Tree & Random Forest tidak butuh scaling (berbasis aturan pembagian).
   - Gaussian Naive Bayes menghitung mean & varians per fitur, tidak butuh scaling.
   - KNN (berbasis jarak) & Logistic Regression (berbasis gradien/regularisasi)
     BUTUH scaling -> pakai scaler.pkl, atau lebih aman buat Pipeline sendiri:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), KNeighborsClassifier())
        model.fit(X_train, y_train.values.ravel())
   Scaler hanya di-fit pada X_train, lalu dipakai untuk transform X_test.

=====================================================================
5. ATURAN UNTUK SEMUA ANGGOTA
=====================================================================
- Pakai file di output/ apa adanya. JANGAN split ulang, JANGAN acak ulang.
- Jangan pernah fit apa pun (scaler, model, seleksi fitur) pada X_test.
- Membaca y: y_train = pd.read_csv("y_train.csv").squeeze()
- Simpan untuk evaluasi bersama: y_pred dan y_proba (probabilitas kelas 1 = Graduate)
  pada X_test, serta waktu training.
- Cek file kalian sama dengan file asli (MD5):
     X_train.csv  a082607c430d44f0dcb3993748f8c6a6
     X_test.csv   4127b1a07685e0f73a400e51bdafb12c
     y_train.csv  f3d30dd28c1a8470aa98b1b58b34f6f1
     y_test.csv   67030176445e89021fc6ee58ee17bbbe
  (Dibuat dengan pandas 3.0.5 & scikit-learn 1.9.1; menjalankan ulang dengan versi
   lain bisa menghasilkan MD5 berbeda, jadi bagikan file CSV-nya, bukan hanya script.)

=====================================================================
6. CATATAN PENTING DARI EDA (untuk laporan)
=====================================================================
Korelasi tiap fitur dengan target ~0 (|r| < 0,01), mutual information <= 0,01,
dan uji cepat Random Forest hanya ~50% akurasi. IPK dan IPS Rata-rata juga
tidak berkorelasi (0,001) dan distribusi fitur rata -> indikasi data sintetis acak.
Tim memutuskan tetap memakai dataset ini; hasil model kemungkinan sekitar 50%
dan ROC-AUC sekitar 0,5. Ini perlu dibahas sebagai temuan di laporan/presentasi.
Detail: tbp-kelulusan/eda/LAPORAN_EDA.md
