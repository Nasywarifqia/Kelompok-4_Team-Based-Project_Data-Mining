DATASET BERSAMA - DETEKSI SITUS WEB PHISHING
TBP Data Mining Kelompok 4 (Laura, Wawa, Gisca, Irfan, Rapip)

1. ISI FOLDER
phishing_websites.csv     Dataset mentah UCI (11.055 baris, 31 kolom)
preprocessing.py          Script cleaning, deduplikasi, dan split data
output/X_train.csv        Fitur data latih   (4.679 baris, 30 kolom)
output/X_test.csv         Fitur data uji     (1.170 baris, 30 kolom)
output/y_train.csv        Target data latih  (0=Legitimate, 1=Phishing)
output/y_test.csv         Target data uji
output/encoding_map.json  Keterangan nilai fitur & mapping target
output/scaler.pkl         StandardScaler (fit di X_train) untuk KNN & LogReg

2. SUMBER DATA
UCI Machine Learning Repository: Phishing Websites (ID 327)
https://archive.ics.uci.edu/dataset/327/phishing+websites

3. LANGKAH CLEANING & PREPROCESSING
- Missing values : 0 di semua kolom.
- Duplikat       : 5.206 baris duplikat dihapus sebelum split -> tersisa 5.849 baris unik.
                   (Wajib dihapus agar baris identik tidak bocor antara X_train dan X_test).
- Target Mapping : Result asli (1=Legitimate, -1=Phishing) diubah menjadi biner:
                   0 = Legitimate (Aman), 1 = Phishing (Berbahaya).
                   Proporsi seimbang: 1 = 51,62% | 0 = 48,38%.
- Train-Test Split: 80:20, random_state=42, stratify=y.
- Scaling        : StandardScaler hanya di-fit pada X_train (disimpan di scaler.pkl).

4. ATURAN UNTUK SEMUA ANGGOTA
- Gunakan CSV di folder output/ apa adanya. JANGAN split atau acak ulang.
- Membaca target: y_train = pd.read_csv("../dataset/output/y_train.csv").squeeze()
- KNN & Logistic Regression wajib menggunakan scaler.pkl.
- Verifikasi MD5 file output/:
  X_train.csv : f685dec89642360bddbaeb0ec08bcd12
  X_test.csv  : d45191fd5fcc0dac3ab25da0b94cb049
  y_train.csv : 456a25ad82b876879c7d61b84ee4fb9e
  y_test.csv  : 1607f523d9ad3efaf35cf1b2357d00a3
