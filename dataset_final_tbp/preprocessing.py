"""
TAHAP B & C - Data Cleaning + Preprocessing (dataset bersama untuk 5 anggota)
Input  : dataset_kelulusan_mahasiswa.csv (file asli Kaggle, taruh di folder yang sama)
Output : folder output/ berisi X_train.csv, X_test.csv, y_train.csv, y_test.csv,
         encoding_map.json, scaler.pkl (opsional, untuk KNN & Logistic Regression)
Jalankan di VS Code:  python preprocessing.py
"""
import json, hashlib
from pathlib import Path

import pandas as pd                                    # olah tabel
import joblib                                          # simpan objek Python (.pkl)
from sklearn.model_selection import train_test_split   # bagi data train/test
from sklearn.preprocessing import StandardScaler       # scaling (z-score)

RAW_CSV = "dataset_kelulusan_mahasiswa.csv"
OUT = Path("output"); OUT.mkdir(exist_ok=True)
TARGET = "Status Kelulusan"
RANDOM_STATE = 42      # JANGAN DIUBAH: dipakai bersama 5 anggota
TEST_SIZE = 0.20       # JANGAN DIUBAH: rasio 80:20

# ---------- 1. Baca data mentah ----------
df = pd.read_csv(RAW_CSV)
print("Data mentah:", df.shape)

# ---------- 2. Cleaning (berdasarkan hasil EDA) ----------
# a. Missing value: EDA menemukan 0 -> tidak ada yang diisi/dihapus. Dicek ulang di sini.
assert df.isna().sum().sum() == 0, "Ada missing value! Periksa file CSV."
# b. Duplikat: EDA menemukan 0 -> tidak ada yang dihapus. Dicek ulang di sini.
assert df.duplicated().sum() == 0, "Ada baris duplikat! Periksa file CSV."
# c. Target hanya boleh 0 (Dropout) dan 1 (Graduate) -> tidak perlu filter.
assert set(df[TARGET].unique()) == {0, 1}, "Target berisi nilai selain 0/1!"
# d. Rentang IPK/IPS harus 0-4.
for c in ["IPK", "IPS Rata-rata", "IPS Semester Akhir"]:
    assert df[c].between(0, 4).all(), f"{c} di luar rentang 0-4"
# e. Semua 9 fitur dipakai: tidak ada kolom ID, tidak ada kolom yang bocor (leakage),
#    dan tidak ada alasan kuat untuk membuang kolom.

# ---------- 3. Encoding fitur kategorikal ----------
# Pemetaan ditulis manual (bukan di-fit dari data) -> tidak ada leakage dan hasil selalu sama.
ENCODING = {
    "Pekerjaan Sambil Kuliah": {"Tidak": 0, "Ya": 1},                 # biner
    "Kategori Kehadiran": {"Rendah": 0, "Sedang": 1, "Tinggi": 2},    # ordinal (ada urutan)
}
for col, mapping in ENCODING.items():
    df[col] = df[col].map(mapping)
    assert df[col].notna().all(), f"Ada kategori tak dikenal di {col}"

LABEL = {0: "Dropout", 1: "Graduate"}

# ---------- 4. Pisahkan fitur (X) dan target (y) ----------
X = df.drop(columns=TARGET)
y = df[TARGET]

# ---------- 5. Train-test split 80:20, stratify ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------- 6. Simpan dataset bersama (TANPA scaling) ----------
X_train.to_csv(OUT / "X_train.csv", index=False)
X_test.to_csv(OUT / "X_test.csv", index=False)
y_train.to_csv(OUT / "y_train.csv", index=False)
y_test.to_csv(OUT / "y_test.csv", index=False)

with open(OUT / "encoding_map.json", "w") as f:
    json.dump({"fitur": ENCODING, "target": {str(k): v for k, v in LABEL.items()},
               "random_state": RANDOM_STATE, "test_size": TEST_SIZE}, f, indent=2)

# ---------- 7. Scaler OPSIONAL (hanya untuk KNN & Logistic Regression) ----------
# Di-fit HANYA pada X_train agar informasi X_test tidak bocor.
scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, OUT / "scaler.pkl")

# ---------- 8. Ringkasan + sidik jari file (untuk cek kesamaan antar anggota) ----------
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("Proporsi target train:", y_train.value_counts(normalize=True).round(4).to_dict())
print("Proporsi target test :", y_test.value_counts(normalize=True).round(4).to_dict())
for name in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    md5 = hashlib.md5((OUT / name).read_bytes()).hexdigest()
    print(f"MD5 {name}: {md5}")
print("Selesai. File tersimpan di folder:", OUT.resolve())
