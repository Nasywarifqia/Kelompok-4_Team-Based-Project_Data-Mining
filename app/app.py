import json
import re
import socket
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PhishGuard — URL Threat Inspection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "dataset" / "output"
EVAL_DIR = BASE_DIR / "evaluation"
KOMPARASI_DIR = EVAL_DIR / "komparasi"

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
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #161615;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums;
    }

    .stApp {
        background-color: #F7F5F0;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    .masthead {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        flex-wrap: wrap;
        gap: 16px;
        padding-bottom: 20px;
        margin-bottom: 24px;
        border-bottom: 1px solid #DCD7CC;
    }

    .masthead-kicker {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #C84B31;
        margin-bottom: 6px;
    }

    .masthead-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.15;
        color: #161615;
        margin: 0;
    }

    .masthead-sub {
        font-size: 0.9rem;
        color: #66635C;
        margin: 6px 0 0 0;
    }

    .masthead-stat {
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #66635C;
        line-height: 1.6;
    }

    .masthead-stat strong {
        color: #161615;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        border-bottom: 1px solid #DCD7CC;
        background: transparent;
        margin-bottom: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 2px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #66635C;
        background: transparent;
        border: none;
        transition: color 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stTabs [aria-selected="true"] {
        color: #161615 !important;
        border-bottom: 2px solid #C84B31 !important;
    }

    .stButton > button {
        border-radius: 4px;
        border: 1px solid #D5CFC4;
        background-color: #FFFFFF;
        color: #161615;
        font-weight: 500;
        font-size: 0.84rem;
        padding: 0.45rem 0.9rem;
        box-shadow: none !important;
        transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button:hover {
        border-color: #161615;
        background-color: #EFECE6;
        color: #161615;
    }

    .stButton > button[kind="primary"] {
        background-color: #161615;
        border-color: #161615;
        color: #F7F5F0;
        font-weight: 600;
        letter-spacing: 0.01em;
        padding: 0.65rem 1.25rem;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #C84B31;
        border-color: #C84B31;
        color: #FFFFFF;
    }

    .panel {
        background: #FFFFFF;
        border: 1px solid #DCD7CC;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .panel-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #66635C;
        margin-bottom: 12px;
    }

    .telemetry-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.83rem;
    }

    .telemetry-table td {
        padding: 8px 0;
        border-bottom: 1px solid #EFECE6;
    }

    .telemetry-table tr:last-child td {
        border-bottom: none;
    }

    .telemetry-key {
        color: #66635C;
        width: 42%;
    }

    .telemetry-val {
        font-family: 'JetBrains Mono', monospace;
        color: #161615;
        text-align: right;
        word-break: break-all;
    }

    .signal-row {
        font-size: 0.82rem;
        padding: 6px 10px;
        margin-bottom: 6px;
        border-radius: 4px;
        border-left: 3px solid;
    }

    .signal-danger {
        background: #F6EAEA;
        border-left-color: #9B2226;
        color: #6E1619;
    }

    .signal-warn {
        background: #F5EFE4;
        border-left-color: #B45309;
        color: #7C4D00;
    }

    .signal-ok {
        background: #E9F0EC;
        border-left-color: #1B4332;
        color: #1B4332;
    }

    .verdict-box {
        background: #FFFFFF;
        border: 1px solid #DCD7CC;
        border-left-width: 5px;
        border-radius: 6px;
        padding: 24px;
        margin: 16px 0 20px 0;
    }

    .verdict-phishing { border-left-color: #9B2226; }
    .verdict-safe { border-left-color: #1B4332; }

    .verdict-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 3px 8px;
        border-radius: 3px;
        display: inline-block;
    }

    .tag-phishing { background: #F6EAEA; color: #9B2226; }
    .tag-safe { background: #E9F0EC; color: #1B4332; }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border: 1px solid #DCD7CC;
        border-radius: 6px;
        background: #FFFFFF;
        margin-bottom: 20px;
    }

    .kpi-cell {
        padding: 16px 20px;
        border-right: 1px solid #DCD7CC;
    }

    .kpi-cell:last-child {
        border-right: none;
    }

    .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #66635C;
        margin-bottom: 6px;
    }

    .kpi-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 600;
        color: #161615;
        font-variant-numeric: tabular-nums;
    }

    .kpi-note {
        font-size: 0.76rem;
        color: #66635C;
        margin-top: 4px;
    }

    .step-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0;
        border: 1px solid #DCD7CC;
        border-radius: 6px;
        background: #FFFFFF;
        margin: 12px 0 24px 0;
    }

    .step-item {
        padding: 16px 18px;
        border-right: 1px solid #DCD7CC;
    }

    .step-item:last-child {
        border-right: none;
    }

    @media (max-width: 768px) {
        .kpi-grid, .step-strip { grid-template-columns: repeat(2, 1fr); }
    }
</style>
"""

st.markdown(CSS_STYLES, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def probe_live_host(url: str) -> dict:
    """Melakukan pemeriksaan DNS, sertifikat SSL, dan HTML ringan secara otomatis (timeout 2.5 detik)."""
    parsed = urlparse(url.strip() if "://" in url else f"http://{url.strip()}")
    host = parsed.netloc.split(":")[0].lower()
    result = {
        "reachable": False,
        "dns_valid": False,
        "ssl_valid": False,
        "cert_days_old": None,
        "external_sfh": False,
        "suspicious_anchors": False,
        "has_iframe": False,
        "has_mailto": False,
        "probe_note": "Offline / Unreachable (lexical fallback)",
    }
    if not host:
        return result

    try:
        socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        result["dns_valid"] = True
    except OSError:
        result["probe_note"] = "DNS lookup failed (NXDOMAIN)"
        return result

    if parsed.scheme == "https":
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((host, parsed.port or 443), timeout=2.5) as sock:
                with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                    cert = ssock.getpeercert()
                    result["ssl_valid"] = True
                    not_before = cert.get("notBefore")
                    if not_before:
                        issued = datetime.strptime(not_before, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
                        result["cert_days_old"] = (datetime.now(timezone.utc) - issued).days
        except Exception:
            result["ssl_valid"] = False

    try:
        req = Request(parsed.geturl(), headers={"User-Agent": "Mozilla/5.0 PhishGuard/1.0"})
        with urlopen(req, timeout=2.5) as resp:
            result["reachable"] = True
            ctype = resp.headers.get("Content-Type", "")
            if "text/html" in ctype.lower():
                html = resp.read(200_000).decode("utf-8", errors="ignore").lower()
                result["has_iframe"] = "<iframe" in html
                result["has_mailto"] = "mailto:" in html
                forms = re.findall(r'<form[^>]+action=["\']([^"\']+)["\']', html)
                if any(a.startswith("http") and host not in a for a in forms):
                    result["external_sfh"] = True
                anchors = re.findall(r'<a[^>]+href=["\']([^"\']+)["\']', html)
                if anchors:
                    ext = sum(1 for a in anchors if a.startswith("#") or "javascript:" in a or (a.startswith("http") and host not in a))
                    if ext / len(anchors) > 0.5:
                        result["suspicious_anchors"] = True
        result["probe_note"] = "Live DNS + TLS/HTML verified"
    except Exception:
        result["probe_note"] = "DNS resolved; HTTP fetch blocked/timed out"

    return result


def extract_url_features(url: str, probe: dict):
    clean_url = url.strip()
    parsed = urlparse(clean_url if "://" in clean_url else f"http://{clean_url}")
    host = parsed.netloc.lower()
    host_clean = host.split(":")[0]

    features = {col: 1 for col in FEATURE_COLS}
    anomalies, suspicious, safeties = [], [], []

    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host_clean))
    if is_ip:
        features["having_IP_Address"] = -1
        anomalies.append("Hostname menggunakan alamat IP langsung")
    else:
        safeties.append("Hostname menggunakan nama domain")

    url_len = len(clean_url)
    if url_len >= 75:
        features["URL_Length"] = -1
        anomalies.append(f"Panjang URL ekstrem ({url_len} karakter)")
    elif url_len >= 54:
        features["URL_Length"] = 0
        suspicious.append(f"Panjang URL menengah ({url_len} karakter)")
    else:
        safeties.append(f"Panjang URL standar ({url_len} karakter)")

    is_shortener = any(s in host for s in SHORTENERS)
    if is_shortener:
        features["Shortining_Service"] = -1
        features["URL_of_Anchor"] = 0
        anomalies.append("Menggunakan layanan URL shortener")
    else:
        safeties.append("Bukan domain shortener")

    if "@" in clean_url:
        features["having_At_Symbol"] = -1
        anomalies.append("Terdapat simbol '@' dalam URL")
    else:
        safeties.append("Bebas dari simbol '@'")

    if clean_url.rfind("//") > 7:
        features["double_slash_redirecting"] = -1
        anomalies.append("Pola redirect double-slash '//' terdeteksi")
    else:
        safeties.append("Struktur slash protokol normal")

    if "-" in host:
        features["Prefix_Suffix"] = -1
        anomalies.append("Domain memuat tanda hubung '-'")
    else:
        safeties.append("Domain tanpa tanda hubung '-'")

    dot_count = host.replace("www.", "").count(".")
    if dot_count >= 3:
        features["having_Sub_Domain"] = -1
        anomalies.append(f"Sub-domain berlapis ({dot_count} titik)")
    elif dot_count == 2:
        features["having_Sub_Domain"] = 0
        suspicious.append("Dua tingkatan sub-domain terdeteksi")
    else:
        safeties.append("Struktur sub-domain normal")

    if "https" in host:
        features["HTTPS_token"] = -1
        anomalies.append("Token 'https' disisipkan pada nama domain")

    if parsed.scheme == "https" and probe.get("ssl_valid"):
        features["SSLfinal_State"] = 1
        safeties.append("TLS/SSL aktif & sertifikat CA valid (live)")
    elif parsed.scheme == "https":
        features["SSLfinal_State"] = 0
        suspicious.append("HTTPS namun sertifikat CA tidak terverifikasi")
    else:
        features["SSLfinal_State"] = -1
        anomalies.append("Protokol HTTP tanpa enkripsi TLS/SSL")

    if not probe.get("dns_valid"):
        features["DNSRecord"] = -1
        features["age_of_domain"] = -1
        features["Domain_registeration_length"] = -1
        features["web_traffic"] = -1
        features["Page_Rank"] = -1
        features["Google_Index"] = -1
        anomalies.append("DNS tidak teresolusi / domain tidak aktif")
    else:
        features["DNSRecord"] = 1
        safeties.append("DNS record aktif terverifikasi")
        if not probe.get("ssl_valid"):
            features["age_of_domain"] = -1
            features["Domain_registeration_length"] = -1
            features["web_traffic"] = 0
            suspicious.append("Tanpa bukti sertifikat TLS jangka panjang")

    if probe.get("suspicious_anchors"):
        features["URL_of_Anchor"] = -1
        anomalies.append("Mayoritas tautan anchor mengarah ke luar")
    if probe.get("external_sfh"):
        features["SFH"] = -1
        anomalies.append("Form handler mengarah ke server eksternal")
    if probe.get("has_iframe"):
        features["Iframe"] = -1
        anomalies.append("Halaman memuat tag <iframe>")
    if probe.get("has_mailto"):
        features["Submitting_to_email"] = -1
        anomalies.append("Form menggunakan skema mailto:")

    if parsed.port and parsed.port not in (80, 443, 8080):
        features["port"] = -1
        anomalies.append(f"Port non-standar (:{parsed.port})")

    meta = {
        "scheme": (parsed.scheme or "http").upper(),
        "host": host or "-",
        "length": f"{url_len} chars",
        "host_type": "Raw IPv4" if is_ip else "Domain Name",
        "probe_status": probe.get("probe_note", "-"),
    }
    return features, meta, anomalies, suspicious, safeties


def load_scaler():
    scaler_file = DATA_DIR / "scaler.pkl"
    return joblib.load(scaler_file) if scaler_file.exists() else None


def get_available_models():
    catalog = {
        "Random Forest": {"file": "random_forest_model.pkl", "needs_scaling": False, "pic": "Wawa"},
        "K-Nearest Neighbors (k=11)": {"file": "knn_model.pkl", "needs_scaling": True, "pic": "Irfan"},
        "Naive Bayes (CategoricalNB)": {"file": "naive_bayes_model.pkl", "needs_scaling": False, "pic": "Gisca"},
        "Decision Tree (CART)": {"file": "decision_tree_model.pkl", "needs_scaling": False, "pic": "Laura"},
        "Logistic Regression": {"file": "logistic_regression_model.pkl", "needs_scaling": True, "pic": "Rapip"},
    }
    models = {}
    for display_name, meta in catalog.items():
        path = MODEL_DIR / meta["file"]
        if path.exists():
            try:
                models[display_name] = {
                    "model": joblib.load(path),
                    "needs_scaling": meta["needs_scaling"],
                    "pic": meta["pic"],
                }
            except Exception:
                pass
    return models


def load_all_metrics():
    rows = []
    for mf in sorted(EVAL_DIR.glob("*metrics*.json")):
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
                rows.append({
                    "Algoritma": data.get("Algoritma", mf.stem),
                    "PIC": data.get("PIC", "-"),
                    "Accuracy": data.get("Accuracy", 0.0),
                    "Precision": data.get("Precision (Phishing)", 0.0),
                    "Recall Phishing": data.get("Recall (Phishing)", 0.0),
                    "F1-Score": data.get("F1-Score (Phishing)", 0.0),
                    "ROC-AUC": data.get("ROC-AUC", 0.0),
                    "Train Time (s)": data.get("Train Time (s)", 0.0),
                })
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["F1-Score", "ROC-AUC"], ascending=False).reset_index(drop=True)
    return df


metrics_df = load_all_metrics()
best_summary = "5 Models Ready"
if not metrics_df.empty:
    top = metrics_df.iloc[0]
    best_summary = f"Best F1: <strong>{top['Algoritma']}</strong> ({top['F1-Score']*100:.2f}% • AUC {top['ROC-AUC']:.4f})"

st.markdown(f"""
<div class="masthead">
    <div>
        <div class="masthead-kicker">Kelompok 4 • Informatika UNS • UCI Phishing Benchmark</div>
        <h1 class="masthead-title">PhishGuard URL Inspector</h1>
        <p class="masthead-sub">Inspeksi keamanan URL otomatis (leksikal, DNS/TLS live probe, dan 30 fitur standar UCI).</p>
    </div>
    <div class="masthead-stat">
        <div>Dataset: <strong>5,849 unique URLs</strong> (80:20 split)</div>
        <div>{best_summary}</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_benchmark, tab_method = st.tabs([
    "URL Inspector",
    "Model Benchmark",
    "Methodology & Team",
])

with tab_detect:
    top_l, top_r = st.columns([7, 3])
    with top_l:
        input_mode = st.radio(
            "Mode Input",
            ["Live URL Inspector", "30-Feature Manual Matrix"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with top_r:
        available_models = get_available_models()
        model_names = list(available_models.keys()) if available_models else ["No Model Found"]
        selected_model_name = st.selectbox("Model Inferensi", model_names, label_visibility="collapsed")

    features_dict = {col: 1 for col in FEATURE_COLS}
    anomalies, suspicious, safeties, meta = [], [], [], {}

    if input_mode == "Live URL Inspector":
        if "url_box" not in st.session_state:
            st.session_state["url_box"] = "http://login-verifikasi-bca-sekuriti.xyz/auth?user=@test&update=1"

        col_main, col_telemetry = st.columns([6, 4], gap="large")

        with col_main:
            st.caption("PRESET UJI CEPAT")
            p_cols = st.columns(4)
            presets = [
                ("Bank Spoofing", "http://login-verifikasi-bca-sekuriti.xyz/auth?user=@test&update=1"),
                ("Google Safe", "https://www.google.com"),
                ("Shortener", "https://bit.ly/update-keamanan-akun-2026"),
                ("Raw IP Host", "http://192.168.1.105/paypal/update-billing.php?id=928"),
            ]
            for col, (label, preset_url) in zip(p_cols, presets):
                with col:
                    if st.button(label, use_container_width=True):
                        st.session_state["url_box"] = preset_url
                        st.rerun()

            url_input = st.text_input(
                "Target URL",
                key="url_box",
                placeholder="https://example.com/login",
            )

            live_probe_enabled = st.checkbox("Aktifkan Live DNS / TLS / HTML Probe otomatis", value=True)
            if live_probe_enabled:
                probe_res = probe_live_host(url_input)
            else:
                probe_res = {
                    "reachable": False,
                    "dns_valid": False,
                    "ssl_valid": False,
                    "probe_note": "Live probe disabled (lexical only)",
                }

            features_dict, meta, anomalies, suspicious, safeties = extract_url_features(url_input, probe_res)
            run_btn = st.button("Analyze URL Risk", type="primary", use_container_width=True)

        with col_telemetry:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-label">Telemetry & Live Probe</div>
                <table class="telemetry-table">
                    <tr><td class="telemetry-key">Protocol</td><td class="telemetry-val">{meta.get('scheme', '-')}</td></tr>
                    <tr><td class="telemetry-key">Host</td><td class="telemetry-val">{meta.get('host', '-')}</td></tr>
                    <tr><td class="telemetry-key">Length</td><td class="telemetry-val">{meta.get('length', '-')}</td></tr>
                    <tr><td class="telemetry-key">Host Format</td><td class="telemetry-val">{meta.get('host_type', '-')}</td></tr>
                    <tr><td class="telemetry-key">Probe Status</td><td class="telemetry-val">{meta.get('probe_status', '-')}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Signal Breakdown", expanded=True):
                for a in anomalies:
                    st.markdown(f'<div class="signal-row signal-danger">RISK — {a}</div>', unsafe_allow_html=True)
                for s in suspicious:
                    st.markdown(f'<div class="signal-row signal-warn">WARN — {s}</div>', unsafe_allow_html=True)
                for sf in safeties[:4]:
                    st.markdown(f'<div class="signal-row signal-ok">PASS — {sf}</div>', unsafe_allow_html=True)
    else:
        st.caption("Sesuaikan indikator UCI secara manual (-1: Phishing, 0: Mencurigakan, 1: Aman)")
        c1, c2, c3 = st.columns(3)
        with c1:
            features_dict["having_IP_Address"] = st.selectbox("Host Berupa IP", [1, -1], format_func=lambda x: "Domain (+1)" if x == 1 else "IP Address (-1)")
            features_dict["URL_Length"] = st.selectbox("Panjang URL", [1, 0, -1], format_func=lambda x: "<54 (+1)" if x == 1 else ("54-75 (0)" if x == 0 else ">=75 (-1)"))
            features_dict["Shortining_Service"] = st.selectbox("URL Shortener", [1, -1], format_func=lambda x: "Tidak (+1)" if x == 1 else "Ya (-1)")
            features_dict["Prefix_Suffix"] = st.selectbox("Tanda '-' Domain", [1, -1], format_func=lambda x: "Tidak (+1)" if x == 1 else "Ada (-1)")
            features_dict["having_Sub_Domain"] = st.selectbox("Sub-domain", [1, 0, -1], format_func=lambda x: "Normal (+1)" if x == 1 else ("2 Level (0)" if x == 0 else ">=3 Level (-1)"))
            features_dict["SSLfinal_State"] = st.selectbox("Status SSL", [1, 0, -1], format_func=lambda x: "Valid CA (+1)" if x == 1 else ("Ragu (0)" if x == 0 else "Tanpa SSL (-1)"))
        with c2:
            features_dict["URL_of_Anchor"] = st.selectbox("Anchor Tag", [1, 0, -1])
            features_dict["Request_URL"] = st.selectbox("External Request URL", [1, -1])
            features_dict["Links_in_tags"] = st.selectbox("Links in Script/Meta", [1, 0, -1])
            features_dict["SFH"] = st.selectbox("Server Form Handler", [1, 0, -1])
            features_dict["Submitting_to_email"] = st.selectbox("Submit via mailto:", [1, -1])
            features_dict["Iframe"] = st.selectbox("Hidden Iframe", [1, -1])
        with c3:
            features_dict["age_of_domain"] = st.selectbox("Umur Domain", [1, -1], format_func=lambda x: ">=6 Bulan (+1)" if x == 1 else "<6 Bulan (-1)")
            features_dict["DNSRecord"] = st.selectbox("DNS Record", [1, -1])
            features_dict["web_traffic"] = st.selectbox("Web Traffic", [1, 0, -1])
            features_dict["Google_Index"] = st.selectbox("Google Index", [1, -1])
            features_dict["Page_Rank"] = st.selectbox("PageRank", [1, -1])
            features_dict["Statistical_report"] = st.selectbox("Blacklist PhishTank", [1, -1])

        run_btn = st.button("Analyze Feature Matrix", type="primary", use_container_width=True)

    input_df = pd.DataFrame([features_dict])[FEATURE_COLS]
    with st.expander("Inspect 30-Feature Vector"):
        st.dataframe(input_df, use_container_width=True, height=70)

    if run_btn and selected_model_name in available_models:
        info = available_models[selected_model_name]
        data_for_pred = input_df
        if info["needs_scaling"]:
            scaler = load_scaler()
            if scaler is not None:
                data_for_pred = scaler.transform(input_df)

        pred = int(info["model"].predict(data_for_pred)[0])
        proba = info["model"].predict_proba(data_for_pred)[0] if hasattr(info["model"], "predict_proba") else None
        is_phish = pred == 1
        phish_p = float(proba[1]) if proba is not None else (0.95 if is_phish else 0.05)
        safe_p = float(proba[0]) if proba is not None else (1.0 - phish_p)

        v_class = "verdict-phishing" if is_phish else "verdict-safe"
        t_class = "tag-phishing" if is_phish else "tag-safe"
        verdict_title = "Phishing Threat Detected" if is_phish else "Legitimate Domain Profile"
        verdict_desc = (
            "Model mendeteksi kombinasi indikator yang konsisten dengan situs spoofing atau pencurian kredensial."
            if is_phish
            else "Parameter URL, enkripsi, dan reputasi domain sesuai dengan profil situs web yang sah."
        )

        st.markdown(f"""
        <div class="verdict-box {v_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:10px;">
                <span class="verdict-tag {t_class}">{verdict_title}</span>
                <span class="mono" style="font-size:0.78rem; color:#66635C;">Model: {selected_model_name} • PIC: {info['pic']}</span>
            </div>
            <p style="font-size:0.92rem; color:#161615; margin:0 0 18px 0;">{verdict_desc}</p>
            <div style="display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:16px; border-top:1px solid #EFECE6; padding-top:14px;">
                <div>
                    <div class="kpi-label">Phishing Probability</div>
                    <div class="kpi-num" style="color:{'#9B2226' if is_phish else '#66635C'};">{phish_p*100:.1f}%</div>
                </div>
                <div>
                    <div class="kpi-label">Legitimate Probability</div>
                    <div class="kpi-num" style="color:{'#1B4332' if not is_phish else '#66635C'};">{safe_p*100:.1f}%</div>
                </div>
                <div>
                    <div class="kpi-label">Preprocessing</div>
                    <div class="kpi-num" style="font-size:1.05rem; padding-top:6px;">{'StandardScaler' if info['needs_scaling'] else 'Native Discrete'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_chart, c_action = st.columns([5, 5], gap="large")
        with c_chart:
            st.caption("PROBABILITY DISTRIBUTION")
            prob_df = pd.DataFrame([
                {"Class": "Legitimate", "Probability": safe_p},
                {"Class": "Phishing", "Probability": phish_p},
            ])
            bar = (
                alt.Chart(prob_df)
                .mark_bar(height=22, clip=True)
                .encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    y=alt.Y("Class:N", title=None),
                    color=alt.Color(
                        "Class:N",
                        scale=alt.Scale(domain=["Legitimate", "Phishing"], range=["#1B4332", "#C84B31"]),
                        legend=None,
                    ),
                    tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Probability:Q", format=".2%")],
                )
                .properties(height=110, background="#FFFFFF", padding={"left": 10, "right": 10, "top": 8, "bottom": 8})
                .configure_axis(labelColor="#161615", titleColor="#161615", gridColor="#EFECE6", domainColor="#DCD7CC")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(bar, use_container_width=True)

        with c_action:
            st.caption("RECOMMENDED ACTION")
            if is_phish:
                st.markdown("""
                - Jangan memasukkan kredensial login, OTP, atau informasi finansial.
                - Tutup halaman dan verifikasi alamat domain resmi melalui mesin pencari.
                - Periksa tanda typo-squatting atau pemendek tautan yang menyembunyikan tujuan asli.
                """)
            else:
                st.markdown("""
                - Parameter leksikal dan sertifikat berada dalam batas aman.
                - Tetap pastikan ikon gembok TLS aktif sebelum mengirim data sensitif.
                - Gunakan autentikasi dua faktor (2FA) pada akun penting.
                """)


with tab_benchmark:
    if not metrics_df.empty:
        top_acc = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]
        top_rec = metrics_df.sort_values("Recall Phishing", ascending=False).iloc[0]
        top_auc = metrics_df.sort_values("ROC-AUC", ascending=False).iloc[0]

        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-cell">
                <div class="kpi-label">Peak Accuracy</div>
                <div class="kpi-num">{top_acc['Accuracy']*100:.2f}%</div>
                <div class="kpi-note">{top_acc['Algoritma']} ({top_acc['PIC']})</div>
            </div>
            <div class="kpi-cell">
                <div class="kpi-label">Peak Phishing Recall</div>
                <div class="kpi-num">{top_rec['Recall Phishing']*100:.2f}%</div>
                <div class="kpi-note">{top_rec['Algoritma']} ({top_rec['PIC']})</div>
            </div>
            <div class="kpi-cell">
                <div class="kpi-label">Peak ROC-AUC</div>
                <div class="kpi-num">{top_auc['ROC-AUC']:.4f}</div>
                <div class="kpi-note">{top_auc['Algoritma']} ({top_auc['PIC']})</div>
            </div>
            <div class="kpi-cell">
                <div class="kpi-label">Holdout Test Set</div>
                <div class="kpi-num">1,170</div>
                <div class="kpi-note">20% Stratified Unique URLs</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("HOLDOUT TEST LEADERBOARD (1,170 UNIQUE SAMPLES)")
        disp = metrics_df.copy()
        for c in ("Accuracy", "Precision", "Recall Phishing", "F1-Score"):
            disp[c] = (disp[c] * 100).map("{:.2f}%".format)
        disp["ROC-AUC"] = disp["ROC-AUC"].map("{:.4f}".format)
        disp["Train Time (s)"] = disp["Train Time (s)"].map("{:.4f}s".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

        melted = metrics_df.melt(
            id_vars=["Algoritma"],
            value_vars=["Accuracy", "Recall Phishing", "F1-Score", "ROC-AUC"],
            var_name="Metric",
            value_name="Score",
        )
        palette = ["#161615", "#C84B31", "#1B4332", "#9A6B4C", "#66635C"]
        comp_chart = (
            alt.Chart(melted)
            .mark_bar(clip=True)
            .encode(
                x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0, labelPadding=8)),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.85, 1.0]), title="Score"),
                color=alt.Color("Algoritma:N", scale=alt.Scale(range=palette)),
                xOffset="Algoritma:N",
                tooltip=["Algoritma:N", "Metric:N", alt.Tooltip("Score:Q", format=".4f")],
            )
            .properties(height=290, background="#FFFFFF", padding={"left": 14, "right": 14, "top": 12, "bottom": 12})
            .configure_axis(labelColor="#161615", titleColor="#161615", gridColor="#EFECE6", domainColor="#DCD7CC")
            .configure_legend(labelColor="#161615", titleColor="#161615")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(comp_chart, use_container_width=True)
    else:
        st.info("Belum ada file `*_metrics.json` di folder `evaluation/`.")

    cv_file = KOMPARASI_DIR / "cv_5fold_summary.csv"
    holm_file = KOMPARASI_DIR / "statistical_test_holm.csv"
    if cv_file.exists() and holm_file.exists():
        st.markdown("---")
        c_cv, c_holm = st.columns(2, gap="large")
        with c_cv:
            st.caption("STRATIFIED 5-FOLD CROSS-VALIDATION (5,849 SAMPLES)")
            st.dataframe(pd.read_csv(cv_file), use_container_width=True, hide_index=True)
        with c_holm:
            st.caption("PAIRED T-TEST + HOLM-BONFERRONI CORRECTION")
            st.dataframe(pd.read_csv(holm_file), use_container_width=True, hide_index=True)


with tab_method:
    st.caption("DATA PIPELINE (ZERO LEAKAGE)")
    st.markdown("""
    <div class="step-strip">
        <div class="step-item">
            <div class="kpi-label">01. Raw UCI Data</div>
            <div class="kpi-num">11,055</div>
            <div class="kpi-note">30 discrete features, 0 missing values</div>
        </div>
        <div class="step-item">
            <div class="kpi-label">02. Deduplication</div>
            <div class="kpi-num">-5,206</div>
            <div class="kpi-note">Removed identical vectors before split</div>
        </div>
        <div class="step-item">
            <div class="kpi-label">03. Unique Corpus</div>
            <div class="kpi-num">5,849</div>
            <div class="kpi-note">48.4% Legitimate / 51.6% Phishing</div>
        </div>
        <div class="step-item">
            <div class="kpi-label">04. Stratified 80:20</div>
            <div class="kpi-num">4,679 : 1,170</div>
            <div class="kpi-note">Independent StandardScaler fit on train</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("TEAM ROSTER & RESPONSIBILITIES (KELOMPOK 4)")
    roster_df = pd.DataFrame([
        {"Anggota": "Laura", "Peran TBP": "Project Lead & Data Prep", "Algoritma": "Decision Tree (CART)", "Fokus": "Aturan keputusan if-then & pruning max_depth"},
        {"Anggota": "Wawa", "Peran TBP": "Feature Engineer", "Algoritma": "Random Forest", "Fokus": "Ensemble bagging (300 trees) & feature importance"},
        {"Anggota": "Gisca", "Peran TBP": "Evaluator & Benchmark", "Algoritma": "Naive Bayes (CategoricalNB)", "Fokus": "Evaluasi gabungan 5 model & skrip benchmark"},
        {"Anggota": "Irfan", "Peran TBP": "Deployment Engineer", "Algoritma": "K-Nearest Neighbors (k=11)", "Fokus": "Tuning k berbasis jarak & dashboard Streamlit"},
        {"Anggota": "Rapip", "Peran TBP": "Business Analyst & Report", "Algoritma": "Logistic Regression", "Fokus": "Analisis koefisien linear & laporan akhir TBP"},
    ])
    st.dataframe(roster_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("TAXONOMY OF 30 UCI SECURITY FEATURES")
    t1, t2 = st.columns(2, gap="large")
    with t1:
        st.markdown("""
        **Address Bar & Lexical (9 Fitur)**  
        IP address di host, panjang URL, layanan shortener, simbol `@`, redirect `//`, tanda `-` pada domain, kedalaman sub-domain, status SSL, dan masa registrasi domain.

        **Abnormal Request & HTML/JS (9 Fitur)**  
        Token `https` palsu pada domain, aset eksternal (`Request_URL`), tautan anchor `<a>`, skrip/meta tags, Server Form Handler (`SFH`), `mailto:`, `on_mouseover`, klik kanan, pop-up, dan hidden `iframe`.
        """)
    with t2:
        st.markdown("""
        **Domain Reputation & Traffic (6 Fitur)**  
        Umur domain (`age_of_domain`), catatan DNS (`DNSRecord`), peringkat `web_traffic`, `Page_Rank`, indeks Google (`Google_Index`), tautan balik, dan laporan blacklist PhishTank.

        **Why Deduplication Matters**  
        Karena fitur UCI sudah didiskretisasi ke `{-1, 0, 1}`, terdapat `5.206` baris duplikat. Pembersihan sebelum `train_test_split` menjamin evaluasi generalisasi yang jujur.
        """)
