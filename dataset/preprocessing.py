import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "phishing_websites.csv"
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "Result"
RANDOM_STATE = 42
TEST_SIZE = 0.20

def main():
    df = pd.read_csv(RAW_CSV)
    print(f"Data mentah: {df.shape[0]} baris, {df.shape[1]} kolom")

    assert df.isna().sum().sum() == 0, "Ditemukan missing value!"
    assert set(df[TARGET_COL].unique()) == {-1, 1}, "Nilai target harus {-1, 1}!"

    dup_count = int(df.duplicated().sum())
    df_clean = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplikat dihapus: {dup_count} baris -> tersisa {df_clean.shape[0]} baris unik")

    # Mapping target: 0 = Legitimate (Aman), 1 = Phishing (Berbahaya)
    df_clean[TARGET_COL] = df_clean[TARGET_COL].map({1: 0, -1: 1})

    X = df_clean.drop(columns=TARGET_COL)
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    encoding_info = {
        "feature_values": {"-1": "Phishing indicator", "0": "Suspicious", "1": "Legitimate indicator"},
        "target_mapping": {"0": "Legitimate (Aman)", "1": "Phishing (Berbahaya)"},
        "raw_rows": int(df.shape[0]),
        "duplicates_removed": dup_count,
        "unique_rows": int(df_clean.shape[0]),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    with open(OUT_DIR / "encoding_map.json", "w", encoding="utf-8") as f:
        json.dump(encoding_info, f, indent=2)

    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print("Proporsi target train (0=Legitimate, 1=Phishing):", y_train.value_counts(normalize=True).round(4).to_dict())
    print("Proporsi target test  (0=Legitimate, 1=Phishing):", y_test.value_counts(normalize=True).round(4).to_dict())
    for name in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        md5 = hashlib.md5((OUT_DIR / name).read_bytes()).hexdigest()
        print(f"MD5 {name}: {md5}")

if __name__ == "__main__":
    main()
