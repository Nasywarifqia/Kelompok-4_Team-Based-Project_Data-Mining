import json
import re
from pathlib import Path
from urllib.parse import urlparse

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Phishing Shield — Sistem Deteksi Keamanan URL",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
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

CSS_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1220px;
    }

    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        color: #ffffff;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.15);
    }

    .app-header h1 {
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 6px 0;
        color: #ffffff;
    }

    .app-header p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0;
        line-height: 1.5;
    }

    .tag-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.725rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .tag-cyan { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .tag-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .tag-slate { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(203, 213, 225, 0.2); }

    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    .card-title {
        font-size: 0.875rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .verdict-banner-phishing {
        background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.1);
    }

    .verdict-banner-safe {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.1);
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .status-badge-phishing { background: #ef4444; color: #ffffff; }
    .status-badge-safe { background: #10b981; color: #ffffff; }

    .metric-container {
        display: flex;
        flex-direction: column;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        font-variant-numeric: tabular-nums;
    }

    .signal-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 7px 11px;
        border-radius: 7px;
        margin-bottom: 6px;
        font-size: 0.825rem;
    }
    .signal-danger { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
    .signal-warning { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
    .signal-safe { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }

    .url-inspect-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 8px;
        margin-top: 10px;
    }
    .url-inspect-cell {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 9px 11px;
    }
    .url-inspect-cell .caption {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 2px;
    }
    .url-inspect-cell .val {
        font-size: 0.85rem;
        font-weight: 600;
        color: #0f172a;
        word-break: break-all;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 7px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.875rem;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: #eff6ff !important;
        color: #1d4ed8 !important;
    }
</style>
"""

st.markdown(CSS_STYLES, unsafe_allow_html=True)


def extract_url_features(url: str, ssl_trusted: bool, old_domain: bool, anchor_suspicious: bool = False, sfh_external: bool = False):
    clean_url = url.strip()
    parsed = urlparse(clean_url if "://" in clean_url else f"http://{clean_url}")
    host = parsed.netloc.lower()
    host_clean = host.split(":")[0]

    features = {col: 1 for col in FEATURE_COLS}
    anomalies = []
    suspicious = []
    safeties = []

    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host_clean))
    if is_ip:
        features["having_IP_Address"] = -1
        features["DNSRecord"] = -1
        anomalies.append("Hostname menggunakan alamat IP langsung")
    else:
        safeties.append("Hostname menggunakan nama domain valid")

    url_len = len(clean_url)
    if url_len >= 75:
        features["URL_Length"] = -1
        anomalies.append(f"Panjang URL ekstrem ({url_len} karakter ≥ 75)")
    elif url_len >= 54:
        features["URL_Length"] = 0
        suspicious.append(f"Panjang URL menengah ({url_len} karakter)")
    else:
        safeties.append(f"Panjang URL standar ({url_len} karakter < 54)")

    is_shortener = any(s in host for s in SHORTENERS)
    if is_shortener:
        features["Shortining_Service"] = -1
        features["URL_of_Anchor"] = 0
        anomalies.append("Menggunakan domain URL shortener")
    else:
        safeties.append("Bukan domain shortener")

    if "@" in clean_url:
        features["having_At_Symbol"] = -1
        anomalies.append("Terdapat simbol '@' untuk mengelabui browser")
    else:
        safeties.append("Bebas dari simbol pengabur '@'")

    if clean_url.rfind("//") > 7:
        features["double_slash_redirecting"] = -1
        anomalies.append("Ditemukan pola redirect double-slash '//'")
    else:
        safeties.append("Format double-slash protokol standar")

    if "-" in host:
        features["Prefix_Suffix"] = -1
        anomalies.append("Domain memuat tanda minus '-' (teknik typo-squatting)")
    else:
        safeties.append("Domain bersih tanpa tanda minus")

    sub_domain_str = host.replace("www.", "")
    dot_count = sub_domain_str.count(".")
    if dot_count >= 3:
        features["having_Sub_Domain"] = -1
        anomalies.append(f"Sub-domain berlapis berlebihan ({dot_count} titik)")
    elif dot_count == 2:
        features["having_Sub_Domain"] = 0
        suspicious.append("Dua tingkatan sub-domain terdeteksi")
    else:
        safeties.append("Struktur subdomain normal")

    if "https" in host:
        features["HTTPS_token"] = -1
        anomalies.append("Token 'https' dipalsukan pada label domain")
    else:
        safeties.append("Tanpa token https palsu pada domain")

    if parsed.scheme == "https" and ssl_trusted:
        features["SSLfinal_State"] = 1
        safeties.append("Koneksi HTTPS terenkripsi & sertifikat valid")
    elif parsed.scheme == "https":
        features["SSLfinal_State"] = 0
        suspicious.append("HTTPS aktif namun penerbit belum terverifikasi")
    else:
        features["SSLfinal_State"] = -1
        anomalies.append("Protokol HTTP tanpa enkripsi SSL/TLS")

    if not old_domain:
        features["age_of_domain"] = -1
        features["Domain_registeration_length"] = -1
        features["web_traffic"] = -1
        features["Page_Rank"] = -1
        features["Google_Index"] = -1
        features["Links_pointing_to_page"] = -1
        anomalies.append("Umur registrasi domain baru (< 6 bulan)")
    else:
        features["age_of_domain"] = 1
        features["Domain_registeration_length"] = 1
        features["web_traffic"] = 1
        features["Page_Rank"] = 1
        features["Google_Index"] = 1
        features["Links_pointing_to_page"] = 1
        safeties.append("Domain memiliki rekam jejak usia matang")

    if anchor_suspicious:
        features["URL_of_Anchor"] = -1
        anomalies.append("Banyak tautan anchor (<a>) kosong atau mengarah ke luar")

    if sfh_external:
        features["SFH"] = -1
        anomalies.append("Server Form Handler (SFH) mengirim kredensial ke server luar")

    if parsed.port and parsed.port not in (80, 443, 8080):
        features["port"] = -1
        anomalies.append(f"Menggunakan port layanan non-standar (:{parsed.port})")

    meta = {
        "scheme": parsed.scheme or "http",
        "host": host or "(tidak terdeteksi)",
        "path": parsed.path or "/",
        "query": parsed.query or "(tidak ada)",
        "length": url_len,
        "is_ip": is_ip,
        "is_shortened": is_shortener,
    }

    return features, meta, anomalies, suspicious, safeties


def load_scaler():
    scaler_file = DATA_DIR / "scaler.pkl"
    return joblib.load(scaler_file) if scaler_file.exists() else None


def get_available_models():
    models = {}
    catalog = {
        "K-Nearest Neighbors (k=11)": {"file": "knn_model.pkl", "needs_scaling": True, "pic": "Irfan"},
        "Random Forest": {"file": "random_forest_model.pkl", "needs_scaling": False, "pic": "Wawa"},
        "Decision Tree (CART)": {"file": "decision_tree_model.pkl", "needs_scaling": False, "pic": "Laura"},
        "Logistic Regression": {"file": "logistic_regression_model.pkl", "needs_scaling": True, "pic": "Rapip"},
        "Naive Bayes": {"file": "naive_bayes_model.pkl", "needs_scaling": False, "pic": "Gisca"},
    }
    for display_name, meta in catalog.items():
        path = MODEL_DIR / meta["file"]
        if path.exists():
            try:
                models[display_name] = {
                    "model": joblib.load(path),
                    "needs_scaling": meta["needs_scaling"],
                    "pic": meta["pic"],
                    "status": "Ready",
                }
            except Exception:
                pass
    return models


def load_all_metrics():
    rows = []
    metric_files = sorted(EVAL_DIR.glob("*metrics*.json"))
    for mf in metric_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
                rows.append({
                    "Algoritma": data.get("Algoritma", mf.stem),
                    "PIC": data.get("PIC", "-"),
                    "Accuracy": f"{data.get('Accuracy', 0) * 100:.2f}%",
                    "Precision": f"{data.get('Precision (Phishing)', 0) * 100:.2f}%",
                    "Recall Phishing": f"{data.get('Recall (Phishing)', 0) * 100:.2f}%",
                    "F1-Score": f"{data.get('F1-Score (Phishing)', 0) * 100:.2f}%",
                    "ROC-AUC": f"{data.get('ROC-AUC', 0):.4f}",
                    "Status": "Evaluated",
                    "raw_acc": data.get("Accuracy", 0),
                    "raw_rec": data.get("Recall (Phishing)", 0),
                    "raw_f1": data.get("F1-Score (Phishing)", 0),
                    "raw_auc": data.get("ROC-AUC", 0),
                })
        except Exception:
            pass

    known_roster = [
        ("K-Nearest Neighbors (k=11)", "Irfan"),
        ("Random Forest", "Wawa"),
        ("Decision Tree (CART)", "Laura"),
        ("Logistic Regression", "Rapip"),
        ("Naive Bayes", "Gisca"),
    ]
    existing_algos = {r["Algoritma"] for r in rows}
    for name, pic in known_roster:
        if not any(name.lower() in existing.lower() for existing in existing_algos):
            rows.append({
                "Algoritma": name,
                "PIC": pic,
                "Accuracy": "Pending",
                "Precision": "Pending",
                "Recall Phishing": "Pending",
                "F1-Score": "Pending",
                "ROC-AUC": "Pending",
                "Status": "In Development",
                "raw_acc": 0.0,
                "raw_rec": 0.0,
                "raw_f1": 0.0,
                "raw_auc": 0.0,
            })
    return pd.DataFrame(rows)


header_html = """
<div class="app-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                <span class="tag-pill tag-cyan">TBP Data Mining</span>
                <span class="tag-pill tag-emerald">UCI Phishing Dataset</span>
                <span class="tag-pill tag-slate">Kelompok 4 • Informatika UNS</span>
            </div>
            <h1>Phishing Shield — URL Threat Analyzer</h1>
            <p>Sistem inspeksi keamanan web real-time berbasis data mining dengan pipeline 30-fitur standar UCI.</p>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 0.78rem; color: #94a3b8; font-family: monospace;">Model Utama: KNN (k=11)</span><br>
            <span style="font-size: 0.78rem; color: #34d399; font-weight: 600;">Accuracy: 92.14% • ROC-AUC: 0.9760</span>
        </div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

tab_detect, tab_leaderboard, tab_about = st.tabs([
    "Investigasi & Deteksi URL",
    "Komparasi 5 Algoritma",
    "Arsitektur & Dokumentasi",
])

with tab_detect:
    col_input, col_config = st.columns([7, 3])

    with col_input:
        st.markdown("**Metode Input Data:**")
        input_mode = st.radio(
            "Metode Input:",
            ["Analisis URL Cepat", "Simulasi 30 Parameter UCI"],
            horizontal=True,
            label_visibility="collapsed",
        )

    with col_config:
        available_models = get_available_models()
        model_names = list(available_models.keys()) if available_models else ["Model Belum Tersedia"]
        selected_model_name = st.selectbox(
            "Model Inferensi:",
            model_names,
            help="Pilih algoritma machine learning yang akan menjalankan klasifikasi risiko.",
        )

    features_dict = {col: 1 for col in FEATURE_COLS}
    anomalies, suspicious, safeties = [], [], []
    meta = {}

    if input_mode == "Analisis URL Cepat":
        if "url_input" not in st.session_state:
            st.session_state["url_input"] = "http://login-verifikasi-bca-sekuriti.xyz/auth?user=@test&update=1"
        if "ssl_trusted" not in st.session_state:
            st.session_state["ssl_trusted"] = False
        if "old_domain" not in st.session_state:
            st.session_state["old_domain"] = False
        if "anchor_suspicious" not in st.session_state:
            st.session_state["anchor_suspicious"] = True
        if "sfh_external" not in st.session_state:
            st.session_state["sfh_external"] = True

        def load_preset(url, ssl, old, anchor, sfh):
            st.session_state["url_input"] = url
            st.session_state["ssl_trusted"] = ssl
            st.session_state["old_domain"] = old
            st.session_state["anchor_suspicious"] = anchor
            st.session_state["sfh_external"] = sfh

        st.caption("Pilihan Preset Uji Cepat:")
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("Bank Spoofing", use_container_width=True):
                load_preset("http://login-verifikasi-bca-sekuriti.xyz/auth?user=@test&update=1", False, False, True, True)
                st.rerun()
        with preset_cols[1]:
            if st.button("Google Safe", use_container_width=True):
                load_preset("https://www.google.com/search?q=cybersecurity+un", True, True, False, False)
                st.rerun()
        with preset_cols[2]:
            if st.button("Bitly Shortener", use_container_width=True):
                load_preset("https://bit.ly/update-keamanan-akun-2026", False, False, True, False)
                st.rerun()
        with preset_cols[3]:
            if st.button("Raw IP Host", use_container_width=True):
                load_preset("http://192.168.1.105/paypal/update-billing-verification.php?id=92834", False, False, True, True)
                st.rerun()

        url_input = st.text_input(
            "Alamat URL Sasaran:",
            value=st.session_state["url_input"],
            placeholder="Contoh: https://secure-login.bank.com/auth",
            help="Tempelkan URL lengkap termasuk protokol http:// atau https://",
        )

        opt_c1, opt_c2 = st.columns(2)
        with opt_c1:
            ssl_trusted = st.checkbox(
                "Sertifikat SSL diverifikasi otoritas terpercaya (CA)",
                value=st.session_state["ssl_trusted"],
            )
        with opt_c2:
            old_domain = st.checkbox(
                "Domain terdaftar lebih dari 6 bulan yang lalu",
                value=st.session_state["old_domain"],
            )

        with st.expander("Konfigurasi Parameter Lanjutan Halaman Web", expanded=False):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                anchor_suspicious = st.checkbox(
                    "Tautan anchor (<a>) mencurigakan atau mengarah ke luar domain",
                    value=st.session_state["anchor_suspicious"],
                )
            with col_adv2:
                sfh_external = st.checkbox(
                    "Formulir login mengirim data ke script/server pihak ketiga (SFH External)",
                    value=st.session_state["sfh_external"],
                )

        features_dict, meta, anomalies, suspicious, safeties = extract_url_features(
            url_input, ssl_trusted, old_domain, anchor_suspicious, sfh_external
        )

        with st.expander("Inspeksi Anatomi URL & Komponen Leksikal", expanded=True):
            inspect_html = f"""
            <div class="url-inspect-grid">
                <div class="url-inspect-cell">
                    <div class="caption">Protokol</div>
                    <div class="val"><code>{meta.get('scheme', '-').upper()}</code></div>
                </div>
                <div class="url-inspect-cell">
                    <div class="caption">Hostname / Host</div>
                    <div class="val"><code>{meta.get('host', '-')}</code></div>
                </div>
                <div class="url-inspect-cell">
                    <div class="caption">Panjang URL</div>
                    <div class="val">{meta.get('length', 0)} Karakter</div>
                </div>
                <div class="url-inspect-cell">
                    <div class="caption">Format Host</div>
                    <div class="val">{'Raw IP' if meta.get('is_ip') else 'Domain Name'}</div>
                </div>
                <div class="url-inspect-cell">
                    <div class="caption">Shortener URL</div>
                    <div class="val">{'Terdeteksi' if meta.get('is_shortened') else 'Tidak'}</div>
                </div>
            </div>
            """
            st.markdown(inspect_html, unsafe_allow_html=True)

            sig_c1, sig_c2 = st.columns(2)
            with sig_c1:
                st.caption("Pola Anomali & Risiko Terdeteksi:")
                if anomalies:
                    for a in anomalies:
                        st.markdown(f'<div class="signal-item signal-danger">⚠️ {a}</div>', unsafe_allow_html=True)
                if suspicious:
                    for s in suspicious:
                        st.markdown(f'<div class="signal-item signal-warning">🔍 {s}</div>', unsafe_allow_html=True)
                if not anomalies and not suspicious:
                    st.markdown('<div class="signal-item signal-safe">✓ Tidak ditemukan pola ancaman leksikal dasar</div>', unsafe_allow_html=True)
            with sig_c2:
                st.caption("Indikator Valid / Lolos Evaluasi:")
                for sf in safeties:
                    st.markdown(f'<div class="signal-item signal-safe">🛡️ {sf}</div>', unsafe_allow_html=True)

    else:
        st.markdown("<p style='font-size:0.875rem; color:#64748b;'>Sesuaikan parameter fitur UCI untuk melakukan simulasi teknis komprehensif:</p>", unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.markdown("**Struktur Alamat & Protokol**")
            features_dict["having_IP_Address"] = st.selectbox(
                "Host Berupa Alamat IP",
                [1, -1],
                format_func=lambda x: "Nama Domain Normal (+1)" if x == 1 else "Alamat IP Mentah (-1)",
            )
            features_dict["URL_Length"] = st.selectbox(
                "Panjang Karakter URL",
                [1, 0, -1],
                format_func=lambda x: "Pendek <54 kar (+1)" if x == 1 else ("Sedang 54-75 kar (0)" if x == 0 else "Panjang ≥75 kar (-1)"),
            )
            features_dict["Shortining_Service"] = st.selectbox(
                "Layanan Pemendek URL",
                [1, -1],
                format_func=lambda x: "Domain Langsung (+1)" if x == 1 else "Pemendek URL (-1)",
            )
            features_dict["Prefix_Suffix"] = st.selectbox(
                "Tanda Minus (-) pada Domain",
                [1, -1],
                format_func=lambda x: "Tanpa Minus (+1)" if x == 1 else "Memuat Tanda Minus (-1)",
            )
            features_dict["having_Sub_Domain"] = st.selectbox(
                "Hierarki Sub-domain",
                [1, 0, -1],
                format_func=lambda x: "Normal / 1 Titik (+1)" if x == 1 else ("2 Tingkat (0)" if x == 0 else "Banyak Tingkat ≥3 (-1)"),
            )
            features_dict["SSLfinal_State"] = st.selectbox(
                "Sertifikat Enkripsi SSL",
                [1, 0, -1],
                format_func=lambda x: "SSL Valid & Terpercaya (+1)" if x == 1 else ("SSL Self-signed / Ragu (0)" if x == 0 else "Tanpa SSL (-1)"),
            )

        with col_f2:
            st.markdown("**Konten & Struktur Halaman**")
            features_dict["URL_of_Anchor"] = st.selectbox(
                "Tautan Anchor (<a> tag)",
                [1, 0, -1],
                format_func=lambda x: "Mayoritas Normal (+1)" if x == 1 else ("Sebagian Mencurigakan (0)" if x == 0 else "Mayoritas Anomali/Kosong (-1)"),
            )
            features_dict["Request_URL"] = st.selectbox(
                "Aset External Request URL",
                [1, -1],
                format_func=lambda x: "Aset Internal Domain (+1)" if x == 1 else "Banyak Aset Domain Luar (-1)",
            )
            features_dict["Links_in_tags"] = st.selectbox(
                "Tautan dalam Script / Meta Tag",
                [1, 0, -1],
                format_func=lambda x: "Internal Normal (+1)" if x == 1 else ("Sebagian Eksternal (0)" if x == 0 else "Mayoritas Eksternal (-1)"),
            )
            features_dict["SFH"] = st.selectbox(
                "Server Form Handler (SFH)",
                [1, 0, -1],
                format_func=lambda x: "Action Handler Internal (+1)" if x == 1 else ("Action Handler Kosong (0)" if x == 0 else "Action Handler Domain Lain (-1)"),
            )
            features_dict["Submitting_to_email"] = st.selectbox(
                "Submit Data via mailto:",
                [1, -1],
                format_func=lambda x: "Server Backend (+1)" if x == 1 else "Kirim Form ke Email (-1)",
            )
            features_dict["Iframe"] = st.selectbox(
                "Iframe Tersembunyi",
                [1, -1],
                format_func=lambda x: "Tanpa Iframe Anonim (+1)" if x == 1 else "Memuat Iframe Tersembunyi (-1)",
            )

        with col_f3:
            st.markdown("**Reputasi Domain & Traffic**")
            features_dict["age_of_domain"] = st.selectbox(
                "Umur Domain Berjalan",
                [1, -1],
                format_func=lambda x: "Lama ≥6 Bulan (+1)" if x == 1 else "Baru <6 Bulan (-1)",
            )
            features_dict["DNSRecord"] = st.selectbox(
                "Catatan Entri DNS",
                [1, -1],
                format_func=lambda x: "DNS Terdaftar & Valid (+1)" if x == 1 else "DNS Record Tidak Ditemukan (-1)",
            )
            features_dict["web_traffic"] = st.selectbox(
                "Peringkat Web Traffic",
                [1, 0, -1],
                format_func=lambda x: "Traffic Tinggi / Top 100k (+1)" if x == 1 else ("Traffic Menengah (0)" if x == 0 else "Tanpa Traffic Tercatat (-1)"),
            )
            features_dict["Google_Index"] = st.selectbox(
                "Status Pengindeksan Google",
                [1, -1],
                format_func=lambda x: "Terindeks Google (+1)" if x == 1 else "Belum / Tidak Terindeks (-1)",
            )
            features_dict["Page_Rank"] = st.selectbox(
                "Google PageRank Score",
                [1, -1],
                format_func=lambda x: "PageRank Valid (+1)" if x == 1 else "PageRank Sangat Rendah (-1)",
            )
            features_dict["Statistical_report"] = st.selectbox(
                "Blacklist PhishTank / StopBadware",
                [1, -1],
                format_func=lambda x: "Bersih dari Laporan (+1)" if x == 1 else "Masuk Daftar Hitam Phishing (-1)",
            )

    input_df = pd.DataFrame([features_dict])[FEATURE_COLS]

    with st.expander("Vektor Lengkap 30 Fitur UCI (Format Numerik Machine Learning)"):
        st.dataframe(input_df, use_container_width=True, height=68)

    btn_analyze = st.button("Jalankan Inferensi Keamanan", type="primary", use_container_width=True)

    if btn_analyze:
        if selected_model_name in available_models:
            model_info = available_models[selected_model_name]
            active_model = model_info["model"]

            data_for_pred = input_df
            scaler_used = False
            if model_info["needs_scaling"]:
                scaler = load_scaler()
                if scaler is not None:
                    data_for_pred = scaler.transform(input_df)
                    scaler_used = True

            prediction = int(active_model.predict(data_for_pred)[0])
            probabilities = (
                active_model.predict_proba(data_for_pred)[0]
                if hasattr(active_model, "predict_proba")
                else None
            )

            is_phishing = prediction == 1
            phish_prob = probabilities[1] if probabilities is not None else (0.95 if is_phishing else 0.05)
            safe_prob = probabilities[0] if probabilities is not None else (1.0 - phish_prob)

            if is_phishing:
                banner_class = "verdict-banner-phishing"
                badge_class = "status-badge-phishing"
                status_text = "Situs Terindikasi Phishing"
                risk_level = "Tingkat Risiko Kritis"
                desc_text = (
                    "URL ini memiliki pola karakteristik leksikal dan struktural yang sangat identik "
                    "dengan situs penipuan, pemalsuan identitas (spoofing), atau pencurian kredensial pengguna."
                )
            else:
                banner_class = "verdict-banner-safe"
                badge_class = "status-badge-safe"
                status_text = "Situs Terverifikasi Aman (Legitimate)"
                risk_level = "Tingkat Risiko Minimal"
                desc_text = (
                    "Struktur URL, parameter enkripsi sertifikat, dan atribut domain konsisten "
                    "dengan pola situs resmi yang sah dan tidak memperlihatkan anomali manipulasi."
                )

            verdict_html = f"""
            <div class="{banner_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                    <span class="status-badge {badge_class}">{status_text}</span>
                    <span style="font-weight: 700; font-size: 0.9rem; color: {'#991b1b' if is_phishing else '#065f46'};">{risk_level}</span>
                </div>
                <p style="font-size: 0.95rem; line-height: 1.55; margin: 0 0 16px 0; color: #1e293b;">
                    {desc_text}
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;">
                    <div class="metric-container">
                        <span class="metric-label">Probabilitas Phishing</span>
                        <span class="metric-value" style="color: {'#dc2626' if is_phishing else '#64748b'};">{phish_prob * 100:.1f}%</span>
                    </div>
                    <div class="metric-container">
                        <span class="metric-label">Probabilitas Aman</span>
                        <span class="metric-value" style="color: {'#16a34a' if not is_phishing else '#64748b'};">{safe_prob * 100:.1f}%</span>
                    </div>
                    <div class="metric-container">
                        <span class="metric-label">Model Inferensi</span>
                        <span class="metric-value" style="font-size: 1.05rem;">{selected_model_name.split('(')[0].strip()}</span>
                    </div>
                    <div class="metric-container">
                        <span class="metric-label">Penskalaan Normalisasi</span>
                        <span class="metric-value" style="font-size: 1.05rem;">{'StandardScaler (Z-Score)' if scaler_used else 'Fitur Asli (Tree-based)'}</span>
                    </div>
                </div>
            </div>
            """
            st.markdown(verdict_html, unsafe_allow_html=True)

            chart_col, reco_col = st.columns([5, 5])

            with chart_col:
                st.markdown("**Distribusi Probabilitas Risiko:**")
                prob_data = pd.DataFrame([
                    {"Kategori": "Aman (Legitimate)", "Probabilitas": safe_prob},
                    {"Kategori": "Phishing (Berbahaya)", "Probabilitas": phish_prob},
                ])

                chart = (
                    alt.Chart(prob_data)
                    .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=26)
                    .encode(
                        x=alt.X("Probabilitas:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                        y=alt.Y("Kategori:N", title=None),
                        color=alt.Color("Kategori:N", scale=alt.Scale(domain=["Aman (Legitimate)", "Phishing (Berbahaya)"], range=["#10b981", "#ef4444"]), legend=None),
                        tooltip=[alt.Tooltip("Kategori:N"), alt.Tooltip("Probabilitas:Q", format=".2%")],
                    )
                    .properties(height=130)
                )
                st.altair_chart(chart, use_container_width=True)

            with reco_col:
                st.markdown("**Panduan Tindakan Pengguna:**")
                if is_phishing:
                    reco_items = [
                        "DILARANG memasukkan username, sandi akun perbankan, atau PIN pada halaman ini.",
                        "Segera tutup jendela tab browser dan jangan unduh berkas lampiran yang diminta.",
                        "Periksa apakah URL memuat kesalahan ketik tipuan (typo-squatting) pada domain utama.",
                    ]
                    for item in reco_items:
                        st.markdown(f"<div style='font-size:0.875rem; margin-bottom:6px; color:#991b1b;'>⛔ {item}</div>", unsafe_allow_html=True)
                else:
                    reco_items = [
                        "Domain memiliki parameter identitas yang konsisten dengan standar situs sah.",
                        "Pastikan ikon gembok pada browser aktif saat mentransaksikan data rahasia.",
                        "Tetap lakukan verifikasi dua langkah (2FA) pada layanan akun personal Anda.",
                    ]
                    for item in reco_items:
                        st.markdown(f"<div style='font-size:0.875rem; margin-bottom:6px; color:#166534;'>✅ {item}</div>", unsafe_allow_html=True)

        else:
            st.warning("Model belum diekspor ke folder `models/`. Silakan jalankan notebook modeling terlebih dahulu.")


with tab_leaderboard:
    st.markdown("### Leaderboard Komparasi 5 Algoritma")
    st.markdown(
        "Tolok ukur performa klasifikasi deteksi situs phishing diuji menggunakan data uji independen "
        "(1.170 baris unik terstratifikasi) yang bebas dari kebocoran data (*data leakage*)."
    )

    metrics_df = load_all_metrics()

    top_c1, top_c2, top_c3, top_c4 = st.columns(4)
    with top_c1:
        st.markdown("""
        <div class="metric-container">
            <span class="metric-label">Model Akurasi Tertinggi</span>
            <span class="metric-value">92.14%</span>
            <span style="font-size:0.75rem; color:#64748b; margin-top:2px;">K-Nearest Neighbors (k=11)</span>
        </div>
        """, unsafe_allow_html=True)
    with top_c2:
        st.markdown("""
        <div class="metric-container">
            <span class="metric-label">ROC-AUC Tertinggi</span>
            <span class="metric-value">0.9760</span>
            <span style="font-size:0.75rem; color:#64748b; margin-top:2px;">Pembeda Legitimate vs Phishing</span>
        </div>
        """, unsafe_allow_html=True)
    with top_c3:
        st.markdown("""
        <div class="metric-container">
            <span class="metric-label">Recall Phishing Prioritas</span>
            <span class="metric-value">93.05%</span>
            <span style="font-size:0.75rem; color:#64748b; margin-top:2px;">Minimasi False Negative Berbahaya</span>
        </div>
        """, unsafe_allow_html=True)
    with top_c4:
        st.markdown("""
        <div class="metric-container">
            <span class="metric-label">Total Sampel Uji</span>
            <span class="metric-value">1,170</span>
            <span style="font-size:0.75rem; color:#64748b; margin-top:2px;">Evaluasi 20% Stratified Holdout</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    display_df = metrics_df[["Algoritma", "PIC", "Accuracy", "Precision", "Recall Phishing", "F1-Score", "ROC-AUC", "Status"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    eval_data = metrics_df[metrics_df["Status"] == "Evaluated"]
    if not eval_data.empty:
        st.markdown("#### Visualisasi Metrik Model yang Telah Terevaluasi")
        chart_rows = []
        for _, row in eval_data.iterrows():
            chart_rows.extend([
                {"Algoritma": row["Algoritma"], "Metrik": "Accuracy", "Skor": row["raw_acc"]},
                {"Algoritma": row["Algoritma"], "Metrik": "Recall (Phishing)", "Skor": row["raw_rec"]},
                {"Algoritma": row["Algoritma"], "Metrik": "F1-Score", "Skor": row["raw_f1"]},
                {"Algoritma": row["Algoritma"], "Metrik": "ROC-AUC", "Skor": row["raw_auc"]},
            ])
        chart_eval_df = pd.DataFrame(chart_rows)

        comp_chart = (
            alt.Chart(chart_eval_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Metrik:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Skor:Q", title="Skor (0.0 - 1.0)", scale=alt.Scale(domain=[0.8, 1.0])),
                color=alt.Color("Algoritma:N", scale=alt.Scale(scheme="category10")),
                xOffset="Algoritma:N",
                tooltip=["Algoritma:N", "Metrik:N", alt.Tooltip("Skor:Q", format=".4f")],
            )
            .properties(height=260)
        )
        st.altair_chart(comp_chart, use_container_width=True)


with tab_about:
    st.markdown("### Arsitektur Rekayasa & Metodologi Proyek")

    st.markdown("""
    #### 1. Pipeline Pemrosesan Data Bebas Kebocoran (Data Leakage)
    Pada penelitian situs web phishing, banyak sampel URL memuat nilai fitur yang identik. Jika data mentah
    langsung dibagi menjadi *train* dan *test*, ribuan baris duplikat akan tersebar ke kedua himpunan
    sehingga menyebabkan model tampak berakurasi tinggi secara semu (*overfitting* artifisial).
    """)

    pipe_c1, pipe_c2, pipe_c3, pipe_c4 = st.columns(4)
    with pipe_c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">1. Dataset Mentah</div>
            <div style="font-size:1.4rem; font-weight:700; color:#0f172a;">11,055 Baris</div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:4px;">Sumber UCI ML Repository dengan 30 indikator keamanan dan 0 missing value.</p>
        </div>
        """, unsafe_allow_html=True)
    with pipe_c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">2. Deduplikasi Ketat</div>
            <div style="font-size:1.4rem; font-weight:700; color:#ef4444;">-5,206 Duplikat</div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:4px;">Pembersihan baris identik untuk menjamin integritas evaluasi generalisasi.</p>
        </div>
        """, unsafe_allow_html=True)
    with pipe_c3:
        st.markdown("""
        <div class="card">
            <div class="card-title">3. Dataset Unik</div>
            <div style="font-size:1.4rem; font-weight:700; color:#10b981;">5,849 Baris</div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:4px;">2.830 Legitimate (48.4%) dan 3.019 Phishing (51.6%) seimbang.</p>
        </div>
        """, unsafe_allow_html=True)
    with pipe_c4:
        st.markdown("""
        <div class="card">
            <div class="card-title">4. Stratified Split 80:20</div>
            <div style="font-size:1.4rem; font-weight:700; color:#2563eb;">4,679 : 1,170</div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:4px;">Pembagian train-test terstratifikasi dengan penskalaan StandardScaler independen.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 2. Pembagian Peran Tim (Kelompok 4)")
    roster_cols = st.columns(5)
    members = [
        {"name": "Laura", "role": "Project Lead & Data Prep", "algo": "Decision Tree (CART)", "desc": "Pembersihan dataset, deduplikasi, dan perumusan target."},
        {"name": "Wawa", "role": "Feature Engineer", "algo": "Random Forest", "desc": "Eksplorasi korelasi fitur dan pemodelan ensemble Random Forest."},
        {"name": "Gisca", "role": "Evaluator & Benchmark", "algo": "Naive Bayes", "desc": "Evaluasi komparatif 5 model, ROC-AUC, dan koordinasi metrik."},
        {"name": "Irfan", "role": "Deployment Engineer", "algo": "K-Nearest Neighbors", "desc": "Optimasi hiperparameter KNN dan arsitektur dashboard Streamlit."},
        {"name": "Rapip", "role": "Business Analyst & Report", "algo": "Logistic Regression", "desc": "Pemodelan regresi logistik, analisis koefisien, dan laporan akhir."},
    ]
    for col, m in zip(roster_cols, members):
        with col:
            st.markdown(f"""
            <div class="card" style="height: 100%;">
                <div style="font-weight:700; font-size:1.05rem; color:#0f172a;">{m['name']}</div>
                <div style="font-size:0.75rem; color:#2563eb; font-weight:600; margin-bottom:8px;">{m['role']}</div>
                <div class="tag-pill tag-slate" style="margin-bottom:8px;">{m['algo']}</div>
                <p style="font-size:0.78rem; color:#64748b; line-height:1.4; margin:0;">{m['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    st.markdown("""
    #### 3. Taksonomi 30 Fitur UCI
    Fitur dalam sistem diklasifikasikan ke dalam 4 domain utama:
    - **Fitur Berbasis Alamat URL (Address Bar):** Pengecekan IP address, panjang karakter URL, pemendek tautan (*URL shortener*), simbol `@`, *double slash redirect*, tanda minus `-` pada nama domain, serta kedalaman *subdomain*.
    - **Fitur Anomali & Manipulasi Permintaan (Abnormal Request):** Keberadaan token HTTPS palsu pada label domain, pengalihan *request* aset gambar/skrip ke server luar, dan manipulasi *port* koneksi.
    - **Fitur Struktur Konten HTML & Skrip:** Analisis persentase tautan *anchor*, integritas Server Form Handler (SFH), pencegahan klik kanan, manipulasi status mouseover, serta keberadaan *hidden iframe*.
    - **Fitur Reputasi & Jejak Domain:** Umur domain, status registrasi DNS, peringkat *traffic*, Google PageRank, dan pencatatan pada basis data ancaman PhishTank / StopBadware.
    """)
