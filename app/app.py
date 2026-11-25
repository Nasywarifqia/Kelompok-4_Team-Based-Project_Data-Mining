import json
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Deteksi Situs Web Phishing — Kelompok 4",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "dataset" / "output"
EVAL_DIR = BASE_DIR / "evaluation"

FEATURE_COLS = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token",
    "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
    "Submitting_to_email", "Abnormal_URL", "Redirect", "on_mouseover",
    "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
    "DNSRecord", "web_traffic", "Page_Rank", "Google_Index",
    "Links_pointing_to_page", "Statistical_report",
]

SHORTENERS = ("bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly", "rebrand.ly")


def extract_url_features(url: str, ssl_trusted: bool, old_domain: bool) -> dict:
    """Ekstraksi indikator leksikal URL menjadi skema fitur UCI {-1: Phishing, 0: Suspicious, 1: Legitimate}."""
    features = {col: 1 for col in FEATURE_COLS}
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.netloc.lower()
    full_url = url.strip()

    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host.split(":")[0]):
        features["having_IP_Address"] = -1

    if len(full_url) >= 75:
        features["URL_Length"] = -1
    elif len(full_url) >= 54:
        features["URL_Length"] = 0

    if any(s in host for s in SHORTENERS):
        features["Shortining_Service"] = -1

    if "@" in full_url:
        features["having_At_Symbol"] = -1

    if full_url.rfind("//") > 7:
        features["double_slash_redirecting"] = -1

    if "-" in host:
        features["Prefix_Suffix"] = -1

    dot_count = host.replace("www.", "").count(".")
    if dot_count >= 3:
        features["having_Sub_Domain"] = -1
    elif dot_count == 2:
        features["having_Sub_Domain"] = 0

    if "https" in host:
        features["HTTPS_token"] = -1

    if parsed.scheme == "https" and ssl_trusted:
        features["SSLfinal_State"] = 1
    elif parsed.scheme == "https":
        features["SSLfinal_State"] = 0
    else:
        features["SSLfinal_State"] = -1

    if not old_domain:
        features["age_of_domain"] = -1
        features["Domain_registeration_length"] = -1

    return features


def load_scaler():
    scaler_file = DATA_DIR / "scaler.pkl"
    return joblib.load(scaler_file) if scaler_file.exists() else None


def get_available_models():
    models = {}
    model_files = {
        "Decision Tree (CART)": "decision_tree_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "K-Nearest Neighbors (KNN)": "knn_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl",
    }
    for name, filename in model_files.items():
        path = MODEL_DIR / filename
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models


st.title("🛡️ Sistem Deteksi Situs Web Phishing")
st.markdown("Deteksi Risiko Keamanan URL Berbasis Data Mining — **Kelompok 4 (Informatika UNS)**")

tab1, tab2, tab3 = st.tabs(["🔍 Deteksi Phishing", "📊 Komparasi 5 Model", "ℹ️ Tentang Proyek"])

with tab1:
    mode = st.radio(
        "Pilih Mode Pengujian:",
        ["Cek URL Otomatis", "Simulasi Manual Indikator Teknis"],
        horizontal=True,
    )

    if mode == "Cek URL Otomatis":
        url_input = st.text_input(
            "Masukkan URL yang ingin diperiksa:",
            value="http://login-verifikasi-akun-bank.xyz/auth?user=@test",
        )
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            ssl_trusted = st.checkbox("Sertifikat SSL terverifikasi & aktif > 1 tahun", value=False)
        with col_opt2:
            old_domain = st.checkbox("Umur domain sudah terdaftar > 6 bulan", value=False)

        feature_dict = extract_url_features(url_input, ssl_trusted, old_domain)
        with st.expander("Lihat Fitur yang Diekstrak dari URL"):
            st.dataframe(pd.DataFrame([feature_dict]), use_container_width=True)
    else:
        st.write("Atur indikator teknis utama (`-1` = Indikator Phishing, `0` = Mencurigakan, `1` = Aman):")
        feature_dict = {col: 1 for col in FEATURE_COLS}
        c1, c2, c3 = st.columns(3)
        with c1:
            feature_dict["SSLfinal_State"] = st.selectbox("SSL Final State", [-1, 0, 1], index=0)
            feature_dict["URL_of_Anchor"] = st.selectbox("URL of Anchor", [-1, 0, 1], index=0)
            feature_dict["Prefix_Suffix"] = st.selectbox("Prefix-Suffix (-) di Domain", [-1, 1], index=0)
        with c2:
            feature_dict["having_Sub_Domain"] = st.selectbox("Sub-domain Berlebihan", [-1, 0, 1], index=1)
            feature_dict["web_traffic"] = st.selectbox("Web Traffic Rank", [-1, 0, 1], index=0)
            feature_dict["age_of_domain"] = st.selectbox("Umur Domain", [-1, 1], index=0)
        with c3:
            feature_dict["Request_URL"] = st.selectbox("External Request URL", [-1, 1], index=0)
            feature_dict["Links_in_tags"] = st.selectbox("Links in Script/Meta Tags", [-1, 0, 1], index=1)
            feature_dict["having_IP_Address"] = st.selectbox("URL Memakai IP Address", [-1, 1], index=1)

    input_df = pd.DataFrame([feature_dict])[FEATURE_COLS]

    st.markdown("---")
    available_models = get_available_models()

    if available_models:
        selected_model_name = st.selectbox("Pilih Model Klasifikasi:", list(available_models.keys()))
        selected_model = available_models[selected_model_name]

        if st.button("Analisis Keamanan URL", type="primary"):
            data_to_predict = input_df
            if "KNN" in selected_model_name or "Logistic" in selected_model_name:
                scaler = load_scaler()
                if scaler is not None:
                    data_to_predict = scaler.transform(input_df)

            pred = int(selected_model.predict(data_to_predict)[0])
            prob = selected_model.predict_proba(data_to_predict)[0] if hasattr(selected_model, "predict_proba") else None

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if pred == 1:
                    st.error("### Status: TERINDIKASI PHISHING ⚠️")
                    st.write("URL memiliki pola karakteristik yang menyerupai situs penipuan/pencurian data.")
                else:
                    st.success("### Status: LEGITIMATE (AMAN) ✅")
                    st.write("URL tidak menunjukkan pola anomali utama dari situs phishing.")

            with col_r2:
                if prob is not None:
                    st.metric("Probabilitas Phishing", f"{prob[1] * 100:.1f}%")
                    st.metric("Probabilitas Aman", f"{prob[0] * 100:.1f}%")
    else:
        st.info("Model klasifikasi (.pkl) belum tersedia di folder `models/`. Selesaikan tahap modeling di `notebooks/` untuk mengaktifkan prediksi.")

with tab2:
    st.subheader("Leaderboard Komparasi 5 Algoritma")
    metric_files = sorted(EVAL_DIR.glob("*metrics*.json"))
    if metric_files:
        rows = []
        for mf in metric_files:
            with open(mf, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        placeholder = pd.DataFrame([
            {"Algoritma": "Decision Tree (CART)", "PIC": "Laura", "Accuracy": "-", "Recall (Phishing)": "-", "ROC-AUC": "-"},
            {"Algoritma": "Random Forest", "PIC": "Wawa", "Accuracy": "-", "Recall (Phishing)": "-", "ROC-AUC": "-"},
            {"Algoritma": "Naive Bayes", "PIC": "Gisca", "Accuracy": "-", "Recall (Phishing)": "-", "ROC-AUC": "-"},
            {"Algoritma": "K-Nearest Neighbors (KNN)", "PIC": "Irfan", "Accuracy": "-", "Recall (Phishing)": "-", "ROC-AUC": "-"},
            {"Algoritma": "Logistic Regression", "PIC": "Rapip", "Accuracy": "-", "Recall (Phishing)": "-", "ROC-AUC": "-"},
        ])
        st.dataframe(placeholder, use_container_width=True)

with tab3:
    st.subheader("Tentang Proyek Team-Based Project (TBP)")
    st.markdown("""
    **Topik:** Sistem Deteksi Situs Web Phishing Berbasis Data Mining  
    **Dataset:** UCI Machine Learning Repository — Phishing Websites (11.055 baris mentah $\\rightarrow$ 5.849 baris unik setelah deduplikasi, 30 fitur).  
    **Pembagian Tim Kelompok 4:**
    - **Laura:** Decision Tree (CART)
    - **Wawa:** Random Forest
    - **Gisca:** Naive Bayes & Evaluasi Gabungan
    - **Irfan:** K-Nearest Neighbors (KNN) & Dashboard Streamlit
    - **Rapip:** Logistic Regression & Laporan Akhir
    """)
