from pathlib import Path
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Early Warning System Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "dataset_final_tbp" / "output"

def load_encoding():
    map_file = DATA_DIR / "encoding_map.json"
    if map_file.exists():
        with open(map_file, "r") as f:
            return json.load(f)
    return None

def load_scaler():
    scaler_file = DATA_DIR / "scaler.pkl"
    if scaler_file.exists():
        return joblib.load(scaler_file)
    return None

def get_available_models():
    models = {}
    model_files = {
        "Decision Tree (CART)": "decision_tree_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "K-Nearest Neighbors (KNN)": "knn_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl"
    }
    for name, filename in model_files.items():
        path = MODEL_DIR / filename
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models

st.title("🎓 Early Warning System Kelulusan Mahasiswa")
st.markdown("Sistem Prediksi Kelulusan & Peringatan Dini Berbasis Data Mining — **Kelompok 4**")

tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Kelulusan", "📊 Komparasi 5 Model", "ℹ️ Tentang Proyek"])

with tab1:
    st.subheader("Form Input Parameter Akademik Mahasiswa")
    st.write("Masukkan indikator capaian akademik untuk melakukan skrining status kelulusan.")

    col1, col2, col3 = st.columns(3)

    with col1:
        ipk = st.number_input("IPK Berjalan", min_value=0.0, max_value=4.0, value=3.20, step=0.01)
        ips_rata = st.number_input("IPS Rata-rata", min_value=0.0, max_value=4.0, value=3.15, step=0.01)
        ips_akhir = st.number_input("IPS Semester Akhir", min_value=0.0, max_value=4.0, value=3.10, step=0.01)

    with col2:
        ips_tren = st.number_input("Tren Nilai IPS (Semester Akhir - Rata-rata)", min_value=-4.0, max_value=4.0, value=float(round(ips_akhir - ips_rata, 2)), step=0.01)
        mk_tidak_lulus = st.number_input("Mata Kuliah Tidak Lulus", min_value=0, max_value=20, value=1, step=1)
        jumlah_cuti = st.number_input("Jumlah Cuti Akademik (Semester)", min_value=0, max_value=4, value=0, step=1)

    with col3:
        jumlah_semester = st.number_input("Jumlah Semester Ditempuh", min_value=1, max_value=14, value=8, step=1)
        pekerjaan = st.selectbox("Pekerjaan Sambil Kuliah", ["Tidak", "Ya"])
        kehadiran = st.selectbox("Kategori Kehadiran", ["Tinggi", "Sedang", "Rendah"])

    pekerjaan_enc = 1 if pekerjaan == "Ya" else 0
    kehadiran_map = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}
    kehadiran_enc = kehadiran_map[kehadiran]

    input_df = pd.DataFrame([{
        "IPK": ipk,
        "Mata Kuliah Tidak Lulus": mk_tidak_lulus,
        "Jumlah Cuti Akademik": jumlah_cuti,
        "Pekerjaan Sambil Kuliah": pekerjaan_enc,
        "Jumlah Semester": jumlah_semester,
        "IPS Rata-rata": ips_rata,
        "IPS Semester Akhir": ips_akhir,
        "IPS Tren": ips_tren,
        "Kategori Kehadiran": kehadiran_enc
    }])

    st.markdown("---")
    available_models = get_available_models()

    if available_models:
        selected_model_name = st.selectbox("Pilih Model untuk Prediksi:", list(available_models.keys()))
        selected_model = available_models[selected_model_name]

        if st.button("Jalankan Prediksi", type="primary"):
            scaler = load_scaler()
            data_to_predict = input_df
            if "KNN" in selected_model_name or "Logistic" in selected_model_name:
                if scaler:
                    data_to_predict = scaler.transform(input_df)

            pred = selected_model.predict(data_to_predict)[0]
            prob = None
            if hasattr(selected_model, "predict_proba"):
                prob = selected_model.predict_proba(data_to_predict)[0]

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                if pred == 1:
                    st.success("### Status Prediksi: LULUS (Graduate)")
                    st.write("Mahasiswa diprediksi memiliki rekam jejak akademik yang aman untuk menyelesaikan studi.")
                else:
                    st.error("### Status Prediksi: BERISIKO DROPOUT")
                    st.write("Terdeteksi indikasi risiko putus studi / ketidaksesuaian kelulusan akademik.")

            with col_res2:
                if prob is not None:
                    prob_grad = prob[1] * 100
                    prob_drop = prob[0] * 100
                    st.metric("Peluang Lulus", f"{prob_grad:.1f}%")
                    st.metric("Peluang Dropout", f"{prob_drop:.1f}%")

            st.markdown("#### Rekomendasi Tindak Lanjut Akademik:")
            if pred == 1:
                st.info("Pertahankan ritme akademik dan monitoring SKS secara berkala hingga tugas akhir.")
            else:
                st.warning("Perlu evaluasi intensif: jadwalkan bimbingan DPA, periksa tunggakan mata kuliah wajib, dan pantau kehadiran perkuliahan.")
    else:
        st.info("Model klasifikasi (.pkl) belum tersedia di folder `models/`. Selesaikan proses pelatihan model di folder `notebooks/` terlebih dahulu untuk mengaktifkan fitur prediksi interaktif.")

with tab2:
    st.subheader("Leaderboard Komparasi 5 Algoritma Klasifikasi")
    st.write("Perbandingan performa model pada data uji (X_test: 20.000 sampel data):")

    dummy_leaderboard = pd.DataFrame([
        {"Algoritma": "Decision Tree (CART)", "PIC": "Laura", "Accuracy": "-", "Precision": "-", "Recall (Dropout)": "-", "F1-Score": "-", "ROC-AUC": "-"},
        {"Algoritma": "Random Forest", "PIC": "Wawa", "Accuracy": "-", "Precision": "-", "Recall (Dropout)": "-", "F1-Score": "-", "ROC-AUC": "-"},
        {"Algoritma": "Naive Bayes (Gaussian)", "PIC": "Gisca", "Accuracy": "-", "Precision": "-", "Recall (Dropout)": "-", "F1-Score": "-", "ROC-AUC": "-"},
        {"Algoritma": "K-Nearest Neighbors (KNN)", "PIC": "Irfan", "Accuracy": "-", "Precision": "-", "Recall (Dropout)": "-", "F1-Score": "-", "ROC-AUC": "-"},
        {"Algoritma": "Logistic Regression", "PIC": "Rapip", "Accuracy": "-", "Precision": "-", "Recall (Dropout)": "-", "F1-Score": "-", "ROC-AUC": "-"}
    ])
    st.dataframe(dummy_leaderboard, use_container_width=True)
    st.caption("Nilai metrik akan diperbarui secara otomatis setelah tahap evaluasi mandiri selesai dikompilasi oleh Gisca di folder `evaluation/`.")

with tab3:
    st.subheader("Tentang Proyek Team-Based Project (TBP)")
    st.markdown("""
    **Mata Kuliah:** Data Mining (Informatika UNS - Semester 5)  
    **Kelompok 4:**
    - Laura (Project Lead & Data Prep) — *Decision Tree*
    - Wawa (Feature Engineer) — *Random Forest*
    - Gisca (Evaluator & Benchmark) — *Naive Bayes*
    - Irfan (Deployment Engineer) — *K-Nearest Neighbors & Streamlit*
    - Rapip (Business Analyst & Report) — *Logistic Regression*
    
    **Tujuan Proyek:**  
    Membangun model machine learning komparatif untuk mendeteksi risiko putus studi mahasiswa secara dini (*Early Warning System*) serta mengemasnya ke dalam prototipe dashboard interaktif.
    """)
