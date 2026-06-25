import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight — Sales & Marketing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(BASE_DIR, "model")
DATA_PATH    = os.path.join(BASE_DIR, "dataset", "Sales - Marketing customer dataset.csv")
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# ─────────────────────────────────────────────────────────────
#  CSS — Palette 04
#  bg:#FFFDF5 | sidebar:#FFF | primary:#364C84 | accent:#95B1EE | highlight:#E7F1A8
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif !important; 
}

h1, h2, h3, .ph-h1, .cc-title {
    font-family: 'Outfit', sans-serif !important;
}

/* ─ App shell ─ */
.stApp                    { background:#FFFDF5 !important; }
header                    { background: transparent !important; }
footer, [data-testid="stMainMenu"], [data-testid="stDeployButton"] { visibility:hidden; }
[data-testid="stSidebarNav"] { display:none; }
.block-container          { padding:1.8rem 2.2rem !important; max-width:100% !important; }

/* ─ Sidebar ─ */
[data-testid="stSidebar"] {
    background:#FFFFFF !important;
    border-right:1px solid #EBF0FA !important;
}
[data-testid="stSidebar"] > div { padding:0 0.8rem !important; }

/* ─ Sidebar Logo ─ */
.sb-logo      { display:flex;align-items:center;gap:10px;padding:1.4rem 0.2rem 1.2rem;border-bottom:1px solid #EEF1F8;margin-bottom:0.9rem; }
.sb-icon      { width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg, #364C84 0%, #95B1EE 100%);display:flex;align-items:center;justify-content:center;font-size:1.1rem;color:#fff;font-weight:800;flex-shrink:0;box-shadow: 0 4px 10px rgba(54,76,132,0.2); }
.sb-name      { font-size:1.05rem;font-weight:700;color:#101828 !important;line-height:1.2;letter-spacing:-0.02em; }
.sb-sub       { font-size:0.68rem;color:#98A2B3 !important;margin-top:1px;font-weight: 500; }

/* ─ Sidebar labels ─ */
.sb-lbl       { font-size:0.65rem;font-weight:700;color:#98A2B3 !important;text-transform:uppercase;letter-spacing:.12em;margin:1.2rem 0 0.45rem 0.2rem; }

/* ─ Radio → nav items ─ */
[data-testid="stSidebar"] .stRadio label > div:first-of-type {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio label {
    border-radius: 10px !important;
    padding: 0.55rem 0.8rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #475467 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
    margin: 0 !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #EEF3FC !important;
    color: #364C84 !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #EEF3FC !important;
    color: #364C84 !important;
    font-weight: 600 !important;
    border-color: rgba(149, 177, 238, 0.3) !important;
}

/* ─ Sidebar user card ─ */
.sb-card { background:#F8F9FD;border:1px solid #EBF0FA;border-radius:12px;padding:0.9rem;font-size:0.78rem;color:#475467;line-height:1.7;margin-top:0.4rem;box-shadow: 0 2px 8px rgba(54,76,132,.02); }
.sb-card b { color:#364C84;font-weight:600; }

/* ─ Status Pill ─ */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
    border: 1px solid transparent;
    width: 95%;
}
.status-pill.ok {
    background: rgba(149, 177, 238, 0.12);
    color: #364C84;
    border-color: rgba(149, 177, 238, 0.25);
}
.status-pill.fail {
    background: rgba(240, 68, 56, 0.08);
    color: #F04438;
    border-color: rgba(240, 68, 56, 0.18);
}
.status-pill span { font-size: 0.8rem; }

/* ─ Page header ─ */
.ph      { margin-bottom:1.6rem; }
.ph-h1   { font-size:1.65rem;font-weight:700;color:#101828;margin:0;letter-spacing:-0.02em; }
.ph-sub  { font-size:0.88rem;color:#667085;margin-top:0.3rem; }

/* ─ KPI cards ─ */
.kcard {
    background:#FFFFFF; border-radius:16px;
    padding:1.3rem 1.4rem;
    border:1px solid #EBF0FA;
    box-shadow:0 4px 20px rgba(54,76,132,.03),0 1px 2px rgba(54,76,132,.04);
    transition:box-shadow .22s,transform .22s;
}
.kcard:hover { box-shadow:0 8px 30px rgba(54,76,132,.1);transform:translateY(-4px); }
.kcard.green {
    background:linear-gradient(135deg,#364C84 0%,#2D3F70 60%,#1B2D5E 100%);
    border:none;
    box-shadow:0 4px 20px rgba(54,76,132,0.15);
}
.kcard.green .k-lbl,.kcard.green .k-val,.kcard.green .k-sub { color:rgba(255,255,255,.9) !important; }
.kcard.green .k-val                                         { color:#FFFFFF !important; }
.kcard.green .k-ico                                         { background:rgba(255,255,255,.15) !important; }
.kcard.green .k-pill                                        { background:rgba(255,255,255,.2) !important;color:#FFFFFF !important; }

.kcard.left-red   { border-left: 4px solid #F04438 !important; }
.kcard.left-blue  { border-left: 4px solid #364C84 !important; }
.kcard.left-light { border-left: 4px solid #95B1EE !important; }

.k-ico  { width:42px;height:42px;border-radius:10px;background:#EEF3FC;display:flex;align-items:center;justify-content:center;font-size:1.2rem;margin-bottom:0.75rem; }
.k-lbl  { font-size:0.78rem;font-weight:600;color:#667085;margin-bottom:0.25rem;text-transform:uppercase;letter-spacing:0.04em; }
.k-val  { font-size:1.75rem;font-weight:700;color:#101828;line-height:1.15;letter-spacing:-0.02em; }
.k-sub  { font-size:0.75rem;margin-top:0.35rem;font-weight:500; }
.k-up   { color:#364C84; }
.k-dn   { color:#F04438; }
.k-pill {
    display:inline-flex;align-items:center;gap:3px;
    background:rgba(149,177,238,.18);color:#364C84;
    font-size:0.7rem;font-weight:700;padding:3px 9px;
    border-radius:999px;margin-top:0.35rem;
}
.k-pill.dn { background:rgba(240,68,56,.1);color:#F04438; }

/* ─ Chart / content cards ─ */
/* ─ Chart / content cards ─ */
div[data-testid="stVerticalBlockBorderOnly"] {
    background:#FFFFFF !important;
    border-radius:16px !important;
    padding:1.4rem 1.6rem !important;
    border:1px solid #EBF0FA !important;
    box-shadow:0 4px 20px rgba(54,76,132,.03),0 1px 2px rgba(54,76,132,.04) !important;
    margin-bottom:1rem !important;
}
.ccard {
    background:#FFFFFF;
    border-radius:16px;
    padding:1.4rem 1.6rem;
    border:1px solid #EBF0FA;
    box-shadow:0 4px 20px rgba(54,76,132,.03),0 1px 2px rgba(54,76,132,.04);
    margin-bottom:1rem;
}
.cc-title    { font-size:1rem;font-weight:600;color:#101828;margin-bottom:.2rem; }
.cc-sub      { font-size:0.8rem;color:#98A2B3;margin-bottom:1rem; }

/* ─ Prediction boxes ─ */
.pb         { border-radius:16px;padding:1.8rem 1.4rem;text-align:center;box-shadow: 0 4px 15px rgba(54,76,132,0.02); }
.pb-red     { background:#FFF1F3;border:1px solid #FECDD3; }
.pb-green   { background:#EEF3FC;border:1px solid #C5D5F5; }
.pb-lbl     { font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem; }
.pb-val     { font-size:3.2rem;font-weight:800;line-height:1;margin:.3rem 0; }
.pb-desc    { font-size:0.83rem;color:#475467; }

/* ─ Recommendation row ─ */
.rc {
    display: flex;
    align-items: center;
    gap: .75rem;
    background: #FFFFFF;
    border: 1px solid #EBF0FA;
    border-radius: 10px;
    padding: .75rem 1rem;
    font-size: .83rem;
    color: #344054;
    margin-bottom: .5rem;
    box-shadow: 0 2px 4px rgba(54, 76, 132, 0.02);
    transition: all 0.2s ease-in-out;
}
.rc:hover {
    transform: translateX(4px);
    border-color: #95B1EE;
    box-shadow: 0 4px 8px rgba(54, 76, 132, 0.06);
}
.rc span:first-child {
    font-size: 1.15rem;
    background: #EEF3FC;
    padding: 6px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    flex-shrink: 0;
}

/* ─ Form styling override ─ */
div[data-testid="stForm"] {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    border: 1px solid #EBF0FA !important;
    padding: 2rem !important;
    box-shadow: 0 4px 20px rgba(54,76,132,.03),0 1px 2px rgba(54,76,132,.04) !important;
}

/* ─ Progress bar ─ */
.stProgress > div > div > div > div { background:linear-gradient(90deg,#364C84,#95B1EE);border-radius:999px; }

/* ─ Buttons ─ */
.stButton > button,
.stFormSubmitButton > button {
    background:#364C84 !important;color:#fff !important;
    border:none !important;border-radius:10px !important;
    font-weight:600 !important;font-size:0.92rem !important;
    padding:.65rem 1.6rem !important;letter-spacing:.01em !important;
    box-shadow:0 3px 8px rgba(54,76,132,.2) !important;
    transition:all .2s !important;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background:#2D3F70 !important;
    box-shadow:0 5px 18px rgba(54,76,132,.35) !important;
    transform:translateY(-1px);
}

/* ─ Modern Segmented Tabs ─ */
.stTabs [data-baseweb="tab-list"] {
    background-color: #EEF1F8 !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 6px !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab-highlight-bar"] {
    display: none !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: #475467 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.2s ease-in-out !important;
    margin: 0 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #364C84 !important;
    background-color: rgba(255, 255, 255, 0.4) !important;
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #364C84 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(54, 76, 132, 0.08) !important;
}

/* ─ st.info ─ */
.stAlert { border-radius:12px !important;border-left:4px solid #364C84 !important;background:#EEF3FC !important; }

/* ─ Inputs ─ */
label { color:#344054 !important;font-size:.83rem !important;font-weight:600 !important; }
hr    { border-color:#EEF1F8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  ARTIFACT STATUS
# ─────────────────────────────────────────────────────────────
model_ok  = os.path.exists(MODEL_PATH)
scaler_ok = os.path.exists(SCALER_PATH)
feat_ok   = os.path.exists(FEATURE_PATH)
data_ok   = os.path.exists(DATA_PATH)

# ─────────────────────────────────────────────────────────────
#  CACHED LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(FEATURE_PATH)

@st.cache_data(show_spinner="Memuat dataset...")
def load_dataset():
    return pd.read_csv(DATA_PATH)

def preprocess_input(raw: dict, feature_names: list) -> np.ndarray:
    # 1. Initialize all features to 0
    encoded = {col: 0 for col in feature_names}
    
    # 2. Fill in numerical features
    numerical_features = [
        'age', 'is_premium_user', 'total_visits', 'avg_session_time',
        'pages_per_session', 'email_open_rate', 'email_click_rate',
        'total_spent', 'avg_order_value', 'discount_used', 'support_tickets',
        'refund_requested', 'delivery_delay_days', 'satisfaction_score',
        'nps_score', 'marketing_spend_per_user', 'lifetime_value',
        'last_3_month_purchase_freq'
    ]
    for feat in numerical_features:
        if feat in raw:
            encoded[feat] = raw[feat]
            
    # 3. Fill in datetime-derived features with median values from the training data
    encoded['days_since_last_purchase'] = 397
    encoded['tenure_days'] = 266
    encoded['signup_year'] = 2023
    encoded['signup_month'] = 6
    
    # 4. Map subscription_type (Annual -> 0, Monthly -> 1)
    sub_type = raw.get('subscription_type')
    if sub_type == 'Annual':
        encoded['subscription_type'] = 0
    elif sub_type == 'Monthly':
        encoded['subscription_type'] = 1
        
    # 5. Map categorical features (One-hot encoding)
    # Gender
    gender = raw.get('gender')
    if gender == 'Male':
        encoded['gender_Male'] = 1
    elif gender == 'Other':
        encoded['gender_Other'] = 1
        
    # Country
    country = raw.get('country')
    country_key = f"country_{country}"
    if country_key in encoded:
        encoded[country_key] = 1
        
    # City
    city = raw.get('city')
    city_key = f"city_{city}"
    if city_key in encoded:
        encoded[city_key] = 1
        
    # Acquisition Channel
    acq = raw.get('acquisition_channel')
    acq_key = f"acquisition_channel_{acq}"
    if acq_key in encoded:
        encoded[acq_key] = 1
        
    # Device Type
    device = raw.get('device_type')
    device_key = f"device_type_{device}"
    if device_key in encoded:
        encoded[device_key] = 1
        
    # Payment Method
    payment = raw.get('payment_method')
    payment_key = f"payment_method_{payment}"
    if payment == "Credit Card":
        payment_key = "payment_method_Card"
    if payment_key in encoded:
        encoded[payment_key] = 1
        
    # 6. Convert to numpy array in the exact order of feature_names
    arr = [encoded[col] for col in feature_names]
    return np.array([arr])

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-icon">📊</div>
        <div>
            <div class="sb-name">ChurnSight</div>
            <div class="sb-sub">Data Science Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-lbl">Main Menu</div>', unsafe_allow_html=True)
    page = st.radio("nav",
                    ["🏠  Dashboard", "🔮  Prediksi Churn", "📈  Eksplorasi Data"],
                    label_visibility="collapsed")

    st.markdown('<div class="sb-lbl">Status Artifacts</div>', unsafe_allow_html=True)
    for ok, label in [(model_ok, "best_model.pkl"), (scaler_ok, "scaler.pkl"),
                      (feat_ok, "feature_names.pkl"), (data_ok, "Dataset CSV")]:
        cls  = "ok" if ok else "fail"
        icon = "<span>✔</span>" if ok else "<span>✘</span>"
        st.markdown(f'<div class="status-pill {cls}">{icon} {label}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-lbl">Info Mahasiswa</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card">
        <b>Desvita Maharani</b><br>
        NIM: A11.2023.15298<br>
        Program: Bengkel Koding Data Science<br>
        Dataset: Sales & Marketing Customer
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":

    st.markdown("""
    <div class="ph">
        <div class="ph-h1">Overview</div>
        <div class="ph-sub">Ringkasan analisis prediksi churn pelanggan — Sales &amp; Marketing Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Welcome Banner ───────────────────────
    st.markdown("""
    <div class="ccard" style="background: linear-gradient(135deg, #364C84 0%, #2D3F70 100%); border: none; color: #FFFFFF; padding: 1.8rem 2rem; margin-bottom: 1.5rem;">
        <h3 style="color: #FFFFFF !important; margin-top: 0; font-size: 1.25rem;">📊 Analisis Retensi &amp; Churn Pelanggan</h3>
        <p style="color: #E2ECF8; font-size: 0.88rem; line-height: 1.6; margin-bottom: 1.2rem; max-width: 800px;">
            Selamat datang di <b>ChurnSight</b>, platform analitik cerdas untuk memprediksi risiko churn pelanggan menggunakan model Machine Learning yang dioptimalkan. Berdasarkan dataset <i>Sales &amp; Marketing Customer</i> dengan 15.000 data pelanggan, sistem ini membantu mengidentifikasi pelanggan berisiko tinggi dan merekomendasikan tindakan retensi yang efektif.
        </p>
        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.15); padding: 6px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 500;">📁 15k Pelanggan</div>
            <div style="background: rgba(255,255,255,0.15); padding: 6px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 500;">🤖 Random Forest (Tuned)</div>
            <div style="background: rgba(255,255,255,0.15); padding: 6px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 500;">📏 F1-Score: 67.96%</div>
            <div style="background: rgba(255,255,255,0.15); padding: 6px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 500;">🎯 Target: Churn (Binary)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Penjelasan Fitur ───────────────────────
    with st.expander("📚 Penjelasan Fitur Dataset", expanded=False):
        st.markdown("""
        Berikut adalah penjelasan singkat mengenai fitur-fitur utama yang digunakan dalam analisis dan prediksi:
        
        * **👤 Data Demografis:**
            * `Gender`: Jenis kelamin pelanggan.
            * `Age`: Usia pelanggan.
            * `Country` & `City`: Lokasi geografis tempat tinggal pelanggan.
            * `Subscription Type`: Jenis paket layanan (Monthly / Annual).
            * `Is Premium User`: Status keanggotaan premium (0: Standar, 1: Premium).
        
        * **📱 Aktivitas Digital:**
            * `Total Visits`: Jumlah kunjungan pelanggan ke aplikasi/website.
            * `Avg Session Time`: Rata-rata waktu yang dihabiskan per kunjungan (menit).
            * `Pages per Session`: Rata-rata jumlah halaman yang diakses per kunjungan.
            * `Email Open Rate`: Rasio keterbukaan email marketing yang dikirim.
            * `Email Click Rate`: Rasio klik pada link dalam email marketing.
        
        * **💳 Data Transaksi & Pembayaran:**
            * `Total Spent`: Akumulasi nominal belanja pelanggan ($).
            * `Avg Order Value`: Rata-rata nilai per transaksi pembelian ($).
            * `Discount Used`: Apakah pelanggan pernah menggunakan diskon belanja.
            * `Support Tickets`: Jumlah tiket komplain/bantuan yang diajukan.
            * `Refund Requested`: Status pengajuan pengembalian dana.
            * `Delivery Delay Days`: Rata-rata hari keterlambatan pengiriman barang.
            * `Satisfaction Score`: Skor kepuasan yang diberikan pelanggan (skala 1–5).
            * `NPS Score`: Net Promoter Score pelanggan (skala 0–10).
        
        * **📈 Nilai Pelanggan & Marketing:**
            * `Lifetime Value (LTV)`: Estimasi total kontribusi finansial pelanggan selama berlangganan.
            * `Marketing Spend per User`: Total biaya pemasaran yang dialokasikan untuk pelanggan tersebut.
            * `Last 3 Month Purchase Freq`: Frekuensi transaksi belanja dalam 3 bulan terakhir.
        """)

    # ── Load data ────────────────────────────
    if not data_ok:
        st.error("❌ Dataset tidak ditemukan di folder `dataset/`.")
        st.stop()
    df = load_dataset()

    total     = len(df)
    churn_n   = int(df["churn"].sum())
    churn_pct = churn_n / total * 100
    avg_ltv   = df["lifetime_value"].mean()
    avg_spent = df["total_spent"].mean()

    # ── KPI cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kcard green">
            <div class="k-ico">🧑‍🤝‍🧑</div>
            <div class="k-lbl">Total Pelanggan</div>
            <div class="k-val">{total:,}</div>
            <div class="k-pill">Dataset Sales & Marketing</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kcard left-red">
            <div class="k-ico" style="background:#FFF1F3">⚠️</div>
            <div class="k-lbl">Churn Rate</div>
            <div class="k-val" style="color:#F04438">{churn_pct:.1f}%</div>
            <div class="k-sub k-dn">▼ {churn_n:,} pelanggan churn</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kcard left-blue">
            <div class="k-ico" style="background:#EEF3FC">💰</div>
            <div class="k-lbl">Avg Lifetime Value</div>
            <div class="k-val" style="color:#364C84">${avg_ltv:,.0f}</div>
            <div class="k-sub k-up">↑ Per pelanggan</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kcard left-light">
            <div class="k-ico" style="background:#F5F9E8">🛒</div>
            <div class="k-lbl">Avg Total Spent</div>
            <div class="k-val" style="color:#364C84">${avg_spent:,.0f}</div>
            <div class="k-sub k-up">↑ Per pelanggan</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model Performance Card ──
    with st.container(border=True):
        st.markdown("""
            <div class="cc-title">🏆 Model Terbaik &amp; Perbandingan Performa</div>
            <div class="cc-sub">Berdasarkan hasil tuning hyperparameter pada notebook</div>
        """, unsafe_allow_html=True)
        m1, m2 = st.columns([1, 1.2])
        with m1:
            st.markdown("""
            <div style="background: #F8F9FD; padding: 1.2rem; border-radius: 12px; border: 1px solid #EBF0FA; margin-bottom: 1rem;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #364C84; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;">Model Terbaik</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #101828; margin-bottom: 0.8rem;">[Tuned] Random Forest</div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                    <div style="background:#FFFFFF; padding: 8px; border-radius: 8px; border: 1px solid #EBF0FA;">
                        <div style="font-size: 0.65rem; color:#667085; font-weight:600;">F1-SCORE</div>
                        <div style="font-size: 1.1rem; font-weight:700; color:#364C84;">67.96%</div>
                    </div>
                    <div style="background:#FFFFFF; padding: 8px; border-radius: 8px; border: 1px solid #EBF0FA;">
                        <div style="font-size: 0.65rem; color:#667085; font-weight:600;">ACCURACY</div>
                        <div style="font-size: 1.1rem; font-weight:700; color:#364C84;">85.70%</div>
                    </div>
                    <div style="background:#FFFFFF; padding: 8px; border-radius: 8px; border: 1px solid #EBF0FA;">
                        <div style="font-size: 0.65rem; color:#667085; font-weight:600;">PRECISION</div>
                        <div style="font-size: 1.1rem; font-weight:700; color:#364C84;">51.70%</div>
                    </div>
                    <div style="background:#FFFFFF; padding: 8px; border-radius: 8px; border: 1px solid #EBF0FA;">
                        <div style="font-size: 0.65rem; color:#667085; font-weight:600;">RECALL</div>
                        <div style="font-size: 1.1rem; font-weight:700; color:#364C84;">99.13%</div>
                    </div>
                </div>
                <div style="font-size: 0.72rem; color: #667085; margin-top: 0.8rem; line-height: 1.4;">
                    *Model hasil tuning <b>RandomizedSearchCV</b> memiliki Recall 99.13% yang sangat tinggi, sangat optimal untuk mendeteksi hampir semua pelanggan yang berpotensi churn agar dapat diberi tindakan pencegahan.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            results_tuned_df = pd.DataFrame({
                'Logistic Regression (Tuned)': {'Accuracy': 0.7627, 'Precision': 0.3656, 'Recall': 0.7495, 'F1-Score': 0.4914},
                'Random Forest (Tuned)': {'Accuracy': 0.8570, 'Precision': 0.5170, 'Recall': 0.9913, 'F1-Score': 0.6796},
                'Voting Classifier (Tuned)': {'Accuracy': 0.8440, 'Precision': 0.4444, 'Recall': 0.0784, 'F1-Score': 0.1333}
            }).T
            results_tuned_df = results_tuned_df.sort_values('F1-Score', ascending=False)
            
            rc_backup = mpl.rcParams.copy()
            try:
                mpl.rcParams.update(mpl.rcParamsDefault)
                import seaborn as sns
                sns.set_theme(style="whitegrid")
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                x = np.arange(len(metrics))
                width = 0.25
                colors = ['#3498db', '#e74c3c', '#2ecc71']

                fig, ax = plt.subplots(figsize=(10, 5))
                for i, (name, row) in enumerate(results_tuned_df.iterrows()):
                    vals = [row[m] for m in metrics]
                    bars = ax.bar(x + i*width, vals, width, label=name,
                                  color=colors[i], edgecolor='black', alpha=0.85)
                    for bar, val in zip(bars, vals):
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.005,
                                f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')

                ax.set_xlabel('Metrik Evaluasi')
                ax.set_ylabel('Nilai')
                ax.set_title('Perbandingan Performa — Hyperparameter Tuning',
                             fontsize=12, fontweight='bold')
                ax.set_xticks(x + width)
                ax.set_xticklabels(metrics)
                ax.set_ylim(0, 1.15)
                ax.legend(loc='lower right', fontsize=8)
                ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            finally:
                mpl.rcParams.update(rc_backup)
                
        with st.expander("📊 Lihat Perbandingan Performa Lengkap (9 Model dari 3 Skenario)"):
            rc_backup = mpl.rcParams.copy()
            try:
                mpl.rcParams.update(mpl.rcParamsDefault)
                import seaborn as sns
                sns.set_theme(style="whitegrid")
                
                results_d = {
                    'Logistic Regression (Konvensional)': {'F1-Score': 0.2405},
                    'Random Forest (Ensemble Bagging)': {'F1-Score': 0.4457},
                    'Voting Classifier (LR + SVM + KNN)': {'F1-Score': 0.0952}
                }
                results_p = {
                    'Logistic Regression (Konvensional)': {'F1-Score': 0.2343},
                    'Random Forest (Ensemble Bagging)': {'F1-Score': 0.2857},
                    'Voting Classifier (LR + SVM + KNN)': {'F1-Score': 0.1333}
                }
                results_t = {
                    'Logistic Regression (Tuned)': {'F1-Score': 0.4914},
                    'Random Forest (Tuned)': {'F1-Score': 0.6796},
                    'Voting Classifier (Tuned)': {'F1-Score': 0.1333}
                }
                
                all_results = {}
                for name, res in results_d.items():
                    all_results[f'[Direct] {name}'] = res
                for name, res in results_p.items():
                    all_results[f'[Prep] {name}'] = res
                for name, res in results_t.items():
                    all_results[f'[Tuned] {name}'] = res

                all_df = pd.DataFrame(all_results).T
                all_df = all_df.sort_values('F1-Score', ascending=False)
                
                from matplotlib.patches import Patch
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # Plot 1 — F1 Score semua 9 model (horizontal bar)
                colors_map = {'[Direct]': '#e74c3c', '[Prep]': '#2ecc71', '[Tuned]': '#f39c12'}
                bar_colors = [colors_map[name.split(']')[0]+']'] for name in all_df.index]

                bars = axes[0].barh(
                    all_df.index[::-1],
                    all_df['F1-Score'][::-1],
                    color=bar_colors[::-1],
                    edgecolor='black', alpha=0.85
                )
                for bar, val in zip(bars, all_df['F1-Score'][::-1]):
                    axes[0].text(bar.get_width() + 0.003,
                                 bar.get_y() + bar.get_height()/2,
                                 f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

                axes[0].set_xlabel('F1-Score')
                axes[0].set_title('F1-Score — 9 Model', fontsize=12, fontweight='bold')
                axes[0].axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
                axes[0].set_xlim(0, 1.1)
                legend_elements = [
                    Patch(facecolor='#e74c3c', label='Direct Modeling'),
                    Patch(facecolor='#2ecc71', label='Preprocessing'),
                    Patch(facecolor='#f39c12', label='Hyperparameter Tuning')
                ]
                axes[0].legend(handles=legend_elements, loc='lower right')

                # Plot 2 — Grouped bar per skenario
                skenario_names = ['LR', 'RF', 'Voting']
                f1_s1 = [0.2405, 0.4457, 0.0952]
                f1_s2 = [0.2343, 0.2857, 0.1333]
                f1_s3 = [0.4914, 0.6796, 0.1333]

                x     = np.arange(len(skenario_names))
                width = 0.25

                b1 = axes[1].bar(x - width, f1_s1, width, label='Direct',
                                  color='#e74c3c', edgecolor='black', alpha=0.85)
                b2 = axes[1].bar(x,          f1_s2, width, label='Preprocessing',
                                  color='#2ecc71', edgecolor='black', alpha=0.85)
                b3 = axes[1].bar(x + width,  f1_s3, width, label='Tuned',
                                  color='#f39c12', edgecolor='black', alpha=0.85)

                for bars_g, vals in [(b1, f1_s1), (b2, f1_s2), (b3, f1_s3)]:
                    for bar, val in zip(bars_g, vals):
                        axes[1].text(bar.get_x() + bar.get_width()/2,
                                     bar.get_height() + 0.008,
                                     f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')

                axes[1].set_xlabel('Model')
                axes[1].set_ylabel('F1-Score')
                axes[1].set_title('Perbandingan F1-Score per Skenario',
                                   fontsize=12, fontweight='bold')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(skenario_names)
                axes[1].set_ylim(0, 1.0)
                axes[1].legend()
                axes[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8)

                plt.suptitle('Perbandingan Performa — 9 Model dari 3 Skenario',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            finally:
                mpl.rcParams.update(rc_backup)

    # ── Charts ───────────────────────────────
    # 1. Distribusi Churn (Double plot from Notebook - Full Width)
    with st.container(border=True):
        st.markdown("""
        <div class="cc-title">Distribusi Variabel Target — Churn</div>
        <div class="cc-sub">Diambil langsung dari hasil visualisasi notebook EDA (Bagian 1)</div>
        <pre style="font-family: monospace; font-size: 0.88rem; line-height: 1.4; color: #344054; background: #F8F9FD; padding: 12px; border-radius: 8px; border: 1px solid #EBF0FA; margin-top: 10px; margin-bottom: 15px;">=== DISTRIBUSI TARGET CHURN ===
       Jumlah  Persentase (%)
churn                        
0       12702           84.68
1        2298           15.32</pre>
        """, unsafe_allow_html=True)
        
        churn_counts = df['churn'].value_counts()
        churn_pct    = df['churn'].value_counts(normalize=True) * 100

        rc_backup = mpl.rcParams.copy()
        try:
            mpl.rcParams.update(mpl.rcParamsDefault)
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Bar chart
            axes[0].bar(['Tidak Churn (0)', 'Churn (1)'],
                        churn_counts.values,
                        color=['#2196F3', '#F44336'])
            for i, (val, pct) in enumerate(zip(churn_counts.values, churn_pct.values)):
                axes[0].text(i, val + 100, f'{val:,}\n({pct:.1f}%)',
                             ha='center', fontsize=11, fontweight='bold')
            axes[0].set_title('Distribusi Churn', fontsize=13, fontweight='bold')
            axes[0].set_ylabel('Jumlah Pelanggan')
            axes[0].set_ylim(0, max(churn_counts.values) * 1.15)

            # Pie chart
            axes[1].pie(churn_counts.values,
                        labels=['Tidak Churn (0)', 'Churn (1)'],
                        autopct='%1.1f%%',
                        colors=['#2196F3', '#F44336'],
                        startangle=90)
            axes[1].set_title('Proporsi Churn', fontsize=13, fontweight='bold')

            plt.suptitle('Distribusi Variabel Target - Churn', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        finally:
            mpl.rcParams.update(rc_backup)
            
        st.text("""⚠️  Catatan: Data IMBALANCED — rasio Tidak Churn:Churn ≈ 5.5:1
     Fokus evaluasi pada F1-Score dan Recall kelas Churn (1)""")

    # 2. Grid of other EDA charts (2 Columns)
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<div class="cc-title">Churn Rate per Acquisition Channel</div><div class="cc-sub">Saluran akuisisi dengan risiko churn tertinggi</div>', unsafe_allow_html=True)
            cg = df.groupby("acquisition_channel")["churn"].mean().reset_index()
            cg["churn"] = cg["churn"] * 100
            chart_acq = alt.Chart(cg).mark_bar(cornerRadiusEnd=6, height=18).encode(
                x=alt.X('churn:Q', title="Churn Rate (%)", scale=alt.Scale(domain=[0, 20]), axis=alt.Axis(labelColor='#344054')),
                y=alt.Y('acquisition_channel:N', sort='-x', title=None, axis=alt.Axis(labelColor='#344054')),
                color=alt.condition(
                    alt.datum.churn >= cg["churn"].mean(),
                    alt.value('#364C84'),
                    alt.value('#95B1EE')
                ),
                tooltip=[alt.Tooltip('acquisition_channel', title="Channel"), alt.Tooltip('churn:Q', title="Churn Rate (%)", format='.1f')]
            ).properties(
                height=220
            ).configure_view(
                strokeOpacity=0
            )
            st.altair_chart(chart_acq, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown('<div class="cc-title">Churn per Subscription Type</div><div class="cc-sub">Monthly vs Annual</div>', unsafe_allow_html=True)
            sc = df.groupby("subscription_type")["churn"].mean().reset_index()
            sc["churn"] = sc["churn"] * 100
            chart_sub = alt.Chart(sc).mark_bar(cornerRadiusEnd=6, width=32).encode(
                x=alt.X('subscription_type:N', title=None, axis=alt.Axis(labelAngle=0, labelColor='#344054')),
                y=alt.Y('churn:Q', title="Churn Rate (%)", scale=alt.Scale(domain=[0, 20]), axis=alt.Axis(labelColor='#344054')),
                color=alt.Color('subscription_type:N', scale=alt.Scale(range=['#364C84', '#95B1EE']), legend=None),
                tooltip=[alt.Tooltip('subscription_type', title="Tipe Langganan"), alt.Tooltip('churn:Q', title="Churn Rate (%)", format='.1f')]
            ).properties(
                height=220
            ).configure_view(
                strokeOpacity=0
            )
            st.altair_chart(chart_sub, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.markdown('<div class="cc-title">Distribusi Usia by Churn</div><div class="cc-sub">Sebaran usia pelanggan churn vs tidak churn</div>', unsafe_allow_html=True)
            import matplotlib as mpl
            rc_backup = mpl.rcParams.copy()
            try:
                mpl.rcParams.update(mpl.rcParamsDefault)
                import seaborn as sns
                sns.set_theme(style="whitegrid")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.kdeplot(data=df[df['churn'] == 0], x='age', fill=True, color='#2196F3', alpha=0.45, label='Tidak Churn (0)', ax=ax, linewidth=2)
                sns.kdeplot(data=df[df['churn'] == 1], x='age', fill=True, color='#F44336', alpha=0.45, label='Churn (1)', ax=ax, linewidth=2)
                
                ax.set_title('Distribusi Usia Berdasarkan Status Churn', fontsize=11, fontweight='bold', color='black')
                ax.set_xlabel('Usia', fontsize=9, color='black')
                ax.set_ylabel('Density', fontsize=9, color='black')
                ax.tick_params(colors='black', labelsize=8)
                ax.legend(fontsize=9, loc='upper right', frameon=True, facecolor='white', edgecolor='none')
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close(fig)
            finally:
                mpl.rcParams.update(rc_backup)

    with col4:
        with st.container(border=True):
            st.markdown("""
            <div class="cc-title">💡 Temuan Kunci Analisis Churn</div>
            <div class="cc-sub">Wawasan penting dari data eksplorasi</div>
            <div style="font-size: 0.88rem; line-height: 1.6; color: #475467; margin-top: 0.8rem;">
                <ul style="padding-left: 1.2rem; margin: 0;">
                    <li style="margin-bottom: 8px;"><b>Rasio Churn Dasar:</b> Tingkat churn rata-rata pelanggan adalah sebesar <b>15.3%</b> (2.298 pelanggan).</li>
                    <li style="margin-bottom: 8px;"><b>Saluran Akuisisi Berisiko:</b> Pelanggan dari channel <i>Organic</i> (15.9%) dan <i>Referral</i> (15.8%) menunjukkan persentase churn sedikit di atas rata-rata.</li>
                    <li style="margin-bottom: 8px;"><b>Tipe Langganan Bulanan:</b> Pengguna paket <i>Monthly</i> (15.5%) lebih rentan melakukan churn dibandingkan paket <i>Annual</i> (15.1%).</li>
                    <li style="margin-bottom: 8px;"><b>Demografi Usia:</b> Distribusi usia pelanggan churn terpusat secara merata di kisaran 30-50 tahun, serupa dengan pola non-churn.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ── Data table ───────────────────────────
    with st.container(border=True):
        st.markdown('<div class="cc-title">Recent Activities — Sample Data</div><div class="cc-sub">10 baris pertama dari dataset</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 2 — PREDIKSI CHURN
# ═════════════════════════════════════════════════════════════
elif page == "🔮  Prediksi Churn":

    st.markdown("""
    <div class="ph">
        <div class="ph-h1">🔮 Prediksi Churn Pelanggan</div>
        <div class="ph-sub">Masukkan data pelanggan untuk memprediksi kemungkinan churn menggunakan model terbaik</div>
    </div>
    """, unsafe_allow_html=True)

    if not all([model_ok, scaler_ok, feat_ok]):
        st.error("❌ Model belum tersedia. Jalankan notebook terlebih dahulu.")
        st.info("Pastikan `best_model.pkl`, `scaler.pkl`, dan `feature_names.pkl` ada di folder `model/`.")
        st.stop()

    model, scaler, feature_names = load_artifacts()

    with st.form("pred_form"):
        # ── Demografis ──
        st.markdown('<div style="font-size: 0.95rem; font-weight: 700; color: #364C84; margin-bottom: 0.8rem; border-bottom: 2px solid #EEF3FC; padding-bottom: 4px; display: flex; align-items: center; gap: 8px;"><span>👤</span> Data Demografis</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            gender  = st.selectbox("Gender", ["Male","Female","Other"])
            country = st.selectbox("Negara", ["USA","Germany","India","UK","Australia"])
        with d2:
            age  = st.slider("Usia", 18, 80, 30)
            city = st.selectbox("Kota", ["New York","Berlin","Mumbai","London","Sydney","Hamburg","Los Angeles"])
        with d3:
            subscription_type = st.selectbox("Tipe Langganan", ["Monthly","Annual"])
            is_premium_user   = st.selectbox("Premium User?", [0,1], format_func=lambda x: "Ya ✓" if x else "Tidak ✗")

        # ── Aktivitas ──
        st.markdown('<div style="font-size: 0.95rem; font-weight: 700; color: #364C84; margin-bottom: 0.8rem; border-bottom: 2px solid #EEF3FC; padding-bottom: 4px; display: flex; align-items: center; gap: 8px; margin-top: 1rem;"><span>📱</span> Aktivitas Digital</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            acquisition_channel = st.selectbox("Channel Akuisisi", ["Organic","Email","Facebook Ads","Referral","Google Ads"])
            device_type         = st.selectbox("Tipe Perangkat", ["Mobile","Desktop","Tablet"])
        with a2:
            total_visits     = st.slider("Total Kunjungan", 1, 50, 15)
            avg_session_time = st.slider("Avg Session Time (mnt)", 0.0, 30.0, 8.0, 0.1)
        with a3:
            pages_per_session = st.slider("Halaman per Sesi", 0.0, 15.0, 4.0, 0.1)
            email_open_rate   = st.slider("Email Open Rate", 0.0, 1.0, 0.5, 0.01)
            email_click_rate  = st.slider("Email Click Rate", 0.0, 0.5, 0.25, 0.01)

        # ── Transaksi ──
        st.markdown('<div style="font-size: 0.95rem; font-weight: 700; color: #364C84; margin-bottom: 0.8rem; border-bottom: 2px solid #EEF3FC; padding-bottom: 4px; display: flex; align-items: center; gap: 8px; margin-top: 1rem;"><span>💳</span> Data Transaksi</div>', unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        with t1:
            total_spent     = st.number_input("Total Spent ($)", 0.0, 20000.0, 500.0, 10.0)
            avg_order_value = st.number_input("Avg Order Value ($)", 0.0, 200.0, 60.0, 1.0)
            discount_used   = st.selectbox("Pernah Pakai Diskon?", [0,1], format_func=lambda x: "Ya" if x else "Tidak")
        with t2:
            payment_method  = st.selectbox("Metode Pembayaran", ["UPI","PayPal","BKash","Credit Card","Bank Transfer"])
            coupon_code     = st.selectbox("Kode Kupon", ["Tidak Ada","NEW20","REF10","SALE5"])
            support_tickets = st.slider("Jumlah Support Ticket", 0, 10, 2)
        with t3:
            refund_requested    = st.selectbox("Pernah Minta Refund?", [0,1], format_func=lambda x: "Ya" if x else "Tidak")
            delivery_delay_days = st.slider("Delay Pengiriman (hari)", 0, 15, 3)
            satisfaction_score  = st.slider("Satisfaction Score (1–5)", 1.0, 5.0, 3.5, 0.5)
            nps_score           = st.slider("NPS Score (0–10)", 0, 10, 5)

        # ── Nilai ──
        st.markdown('<div style="font-size: 0.95rem; font-weight: 700; color: #364C84; margin-bottom: 0.8rem; border-bottom: 2px solid #EEF3FC; padding-bottom: 4px; display: flex; align-items: center; gap: 8px; margin-top: 1rem;"><span>📈</span> Nilai &amp; Marketing</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        with v1:
            marketing_spend_per_user   = st.slider("Marketing Spend per User ($)", 5.0, 30.0, 17.0, 0.5)
            last_3_month_purchase_freq = st.slider("Frekuensi Beli 3 Bulan Terakhir", 0, 14, 7)
        with v2:
            lifetime_value = st.number_input("Lifetime Value ($)", 0.0, 5000.0, 1200.0, 10.0)

        submitted = st.form_submit_button("🔮  Prediksi Sekarang", use_container_width=True)

    # ── Hasil ─────────────────────────────────
    if submitted:
        raw = {
            "gender": gender, "age": age, "country": country, "city": city,
            "acquisition_channel": acquisition_channel, "device_type": device_type,
            "subscription_type": subscription_type, "is_premium_user": is_premium_user,
            "total_visits": total_visits, "avg_session_time": avg_session_time,
            "pages_per_session": pages_per_session, "email_open_rate": email_open_rate,
            "email_click_rate": email_click_rate, "total_spent": total_spent,
            "avg_order_value": avg_order_value, "discount_used": discount_used,
            "coupon_code": coupon_code if coupon_code != "Tidak Ada" else "NONE",
            "support_tickets": support_tickets, "refund_requested": refund_requested,
            "delivery_delay_days": delivery_delay_days, "payment_method": payment_method,
            "satisfaction_score": satisfaction_score, "nps_score": nps_score,
            "marketing_spend_per_user": marketing_spend_per_user,
            "lifetime_value": lifetime_value,
            "last_3_month_purchase_freq": last_3_month_purchase_freq,
        }
        with st.spinner("Memproses prediksi..."):
            X_scaled   = scaler.transform(preprocess_input(raw, feature_names))
            pred       = model.predict(X_scaled)[0]
            proba      = model.predict_proba(X_scaled)[0]
            churn_prob = proba[1]; safe_prob = proba[0]

        st.markdown('<div style="font-size: 1.1rem; font-weight: 700; color: #364C84; margin-top: 1.8rem; margin-bottom: 1.2rem; border-bottom: 2px solid #EEF3FC; padding-bottom: 6px;">📊 Hasil Analisis Prediksi</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            if pred == 1:
                st.markdown(f"""
                <div class="pb pb-red">
                    <div class="pb-lbl" style="color:#F04438">⚠️ Berisiko Churn</div>
                    <div class="pb-val" style="color:#E11D48">{churn_prob*100:.1f}%</div>
                    <div class="pb-desc">Probabilitas pelanggan akan churn</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pb pb-green">
                    <div class="pb-lbl" style="color:#364C84">✅ Pelanggan Aman</div>
                    <div class="pb-val" style="color:#364C84">{safe_prob*100:.1f}%</div>
                    <div class="pb-desc">Probabilitas pelanggan TIDAK churn</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption(f"Churn: {churn_prob*100:.1f}%");   st.progress(float(churn_prob))
            st.caption(f"Tidak Churn: {safe_prob*100:.1f}%"); st.progress(float(safe_prob))

        with r2:
            st.markdown('<div style="font-size: 0.95rem; font-weight: 700; color: #364C84; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 6px;">💡 Rekomendasi Retensi</div>', unsafe_allow_html=True)
            recoms = []
            if pred == 1:
                if satisfaction_score < 3:         recoms.append(("📞","Hubungi pelanggan untuk survey kepuasan"))
                if support_tickets >= 3:           recoms.append(("🛠️","Prioritaskan penyelesaian support ticket"))
                if discount_used == 0:             recoms.append(("🎁","Tawarkan voucher atau diskon eksklusif"))
                if last_3_month_purchase_freq < 3: recoms.append(("📧","Kirim email reaktivasi dengan penawaran menarik"))
                if nps_score < 5:                  recoms.append(("⭐","Program loyalitas untuk meningkatkan NPS"))
                if subscription_type == "Monthly": recoms.append(("🔄","Tawarkan upgrade ke Annual dengan harga promosi"))
                recoms.append(("👤","Assign customer success manager"))
            else:
                recoms = [
                    ("🌟","Pelanggan sehat — pertahankan kualitas layanan"),
                    ("🚀","Tawarkan program referral"),
                    ("💎","Pertimbangkan upgrade ke premium"),
                    ("📊","Pantau metrik secara berkala"),
                ]
            for icon, text in recoms:
                st.markdown(f'<div class="rc"><span>{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)

        with st.expander("📋 Ringkasan Input"):
            st.dataframe(pd.DataFrame([raw]), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
#  PAGE 3 — EKSPLORASI DATA
# ═════════════════════════════════════════════════════════════
elif page == "📈  Eksplorasi Data":

    st.markdown("""
    <div class="ph">
        <div class="ph-h1">📈 Eksplorasi Dataset</div>
        <div class="ph-sub">Analisis mendalam distribusi, korelasi, dan pola churn dalam data</div>
    </div>
    """, unsafe_allow_html=True)

    if not data_ok:
        st.error("❌ Dataset tidak ditemukan."); st.stop()

    df = load_dataset()
    import matplotlib.pyplot as plt
    import matplotlib, seaborn as sns
    matplotlib.rcParams.update({
        'figure.facecolor':'none',
        'axes.facecolor':'none',
        'text.color':'#344054',
        'axes.labelcolor':'#344054',
        'xtick.color':'#667085',
        'ytick.color':'#667085',
        'axes.edgecolor':'#EBF0FA',
        'grid.color':'#F8F9FD',
        'axes.spines.top':False,
        'axes.spines.right':False,
        'font.family':'sans-serif',
        'font.sans-serif':['Inter', 'DejaVu Sans', 'Arial']
    })

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribusi Fitur", "🔗 Korelasi", "🎯 Analisis Churn", "🏆 Performa Model & Fitur"])

    with tab1:
        # 1. Missing Value plot from Cell 16
        with st.container(border=True):
            st.markdown('<div class="cc-title">Visualisasi Missing Value per Kolom</div><div class="cc-sub">Diambil langsung dari Cell 16 di notebook</div>', unsafe_allow_html=True)
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            cols_missing = missing_pct[missing_pct > 0]
            
            if len(cols_missing) > 0:
                import matplotlib.pyplot as plt
                import matplotlib as mpl
                rc_backup = mpl.rcParams.copy()
                try:
                    mpl.rcParams.update(mpl.rcParamsDefault)
                    import seaborn as sns
                    sns.set_theme(style="whitegrid")
                    fig, ax = plt.subplots(figsize=(12, 5))
                    cols_missing.sort_values(ascending=False).plot(
                        kind='bar', color='steelblue', edgecolor='black', ax=ax
                    )
                    ax.set_title('Persentase Missing Value per Kolom', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Kolom')
                    ax.set_ylabel('Persentase (%)')
                    plt.xticks(rotation=45, ha='right')
                    for i, v in enumerate(cols_missing.sort_values(ascending=False)):
                        ax.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                finally:
                    mpl.rcParams.update(rc_backup)
            else:
                st.info("✅ Tidak ada missing value dalam dataset!")
        
        # 2. Outlier boxplot panel from Cell 21
        with st.container(border=True):
            st.markdown('<div class="cc-title">Analisis Outlier — Boxplot Fitur Numerik</div><div class="cc-sub">Diambil langsung dari Cell 21 di notebook</div>', unsafe_allow_html=True)
            import matplotlib as mpl
            rc_backup = mpl.rcParams.copy()
            try:
                mpl.rcParams.update(mpl.rcParamsDefault)
                import seaborn as sns
                sns.set_theme(style="whitegrid")
                
                num_cols = ['age', 'total_spent', 'avg_order_value', 'lifetime_value',
                            'avg_session_time', 'pages_per_session', 'total_visits',
                            'satisfaction_score', 'marketing_spend_per_user']

                fig, axes = plt.subplots(3, 3, figsize=(14, 10))
                axes = axes.flatten()

                for i, col in enumerate(num_cols):
                    axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                                    boxprops=dict(facecolor='#90CAF9'))
                    axes[i].set_title(col, fontsize=10, fontweight='bold')
                    axes[i].set_ylabel('Nilai')

                plt.suptitle('Analisis Outlier - Boxplot Fitur Numerik',
                             fontsize=13, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            finally:
                mpl.rcParams.update(rc_backup)
                
            st.text("""=== CEK ANOMALI age ===
Jumlah age negatif  : 3
Jumlah age > 80     : 30
Min age             : -4.0
Max age             : 95.0""")

        # 3. Interactive distribution selector
        with st.container(border=True):
            st.markdown('<div class="cc-title">Eksplorasi Distribusi Fitur Secara Interaktif</div>', unsafe_allow_html=True)
            num_cols_sel = df.select_dtypes(include=np.number).columns.tolist()
            selected = st.selectbox("Pilih fitur numerik:", num_cols_sel)
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown('<div class="cc-title">Histogram</div>', unsafe_allow_html=True)
                    rc_backup = mpl.rcParams.copy()
                    try:
                        mpl.rcParams.update(mpl.rcParamsDefault)
                        import seaborn as sns
                        sns.set_theme(style="whitegrid")
                        fig, ax = plt.subplots(figsize=(5,3.6))
                        ax.hist(df[selected].dropna(), bins=35, color="#364C84", alpha=0.82, edgecolor='none')
                        ax.set_title(f"Distribusi: {selected}", fontsize=10, fontweight='600')
                        ax.set_xlabel(selected,fontsize=9); ax.set_ylabel("Frekuensi",fontsize=9)
                        st.pyplot(fig)
                        plt.close(fig)
                    finally:
                        mpl.rcParams.update(rc_backup)
            with col2:
                with st.container(border=True):
                    st.markdown('<div class="cc-title">Boxplot by Churn</div>', unsafe_allow_html=True)
                    rc_backup = mpl.rcParams.copy()
                    try:
                        mpl.rcParams.update(mpl.rcParamsDefault)
                        import seaborn as sns
                        sns.set_theme(style="whitegrid")
                        fig, ax = plt.subplots(figsize=(5,3.6))
                        ax.boxplot([df[df["churn"]==0][selected].dropna(), df[df["churn"]==1][selected].dropna()],
                                   patch_artist=True,
                                   boxprops=dict(facecolor='#EEF3FC',color='#C5D5F5'),
                                   medianprops=dict(color='#364C84',linewidth=2.5),
                                   whiskerprops=dict(color='#D1D5DB'), capprops=dict(color='#D1D5DB'),
                                   flierprops=dict(marker='o',color='#F04438',alpha=0.3,markersize=3))
                        ax.set_xticklabels(["Tidak Churn","Churn"])
                        ax.set_title(f"{selected} per Kelompok",fontsize=10,fontweight='600')
                        st.pyplot(fig)
                        plt.close(fig)
                    finally:
                        mpl.rcParams.update(rc_backup)
            st.dataframe(df[selected].describe().to_frame().T, use_container_width=True)

    with tab2:
        with st.container(border=True):
            st.markdown('<div class="cc-title">Heatmap Korelasi Fitur Numerik</div><div class="cc-sub">Diambil langsung dari Cell 23 di notebook</div>', unsafe_allow_html=True)
            import matplotlib as mpl
            rc_backup = mpl.rcParams.copy()
            try:
                mpl.rcParams.update(mpl.rcParamsDefault)
                import seaborn as sns
                sns.set_theme(style="whitegrid")
                
                num_cols_corr = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'customer_id' in num_cols_corr:
                    num_cols_corr.remove('customer_id')
                
                corr_matrix = df[num_cols_corr].corr()
                
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(corr_matrix,
                            annot=True, fmt='.2f',
                            cmap='coolwarm', center=0,
                            linewidths=0.5,
                            annot_kws={'size': 8},
                            ax=ax)
                ax.set_title('Heatmap Korelasi Fitur Numerik', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            finally:
                mpl.rcParams.update(rc_backup)
        
        with st.container(border=True):
            st.markdown('<div class="cc-title">Top Korelasi Terhadap Churn</div><div class="cc-sub">Korelasi absolut 10 fitur teratas terhadap variabel target churn</div>', unsafe_allow_html=True)
            st.text("=== TOP KORELASI TERHADAP CHURN ===")
            num_cols_corr = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'customer_id' in num_cols_corr:
                num_cols_corr.remove('customer_id')
            corr_matrix = df[num_cols_corr].corr()
            churn_corr = corr_matrix['churn'].drop('churn').abs().sort_values(ascending=False)
            st.dataframe(churn_corr.head(10).to_frame('Korelasi Absolut'), use_container_width=True)

    with tab3:
        cat_feature = st.selectbox("Pilih fitur kategorikal:",
                                   ["acquisition_channel","device_type","subscription_type",
                                    "payment_method","country","gender"])
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown('<div class="cc-title">Churn Rate per Segmen</div>', unsafe_allow_html=True)
                cby  = df.groupby(cat_feature)["churn"].mean().sort_values() * 100
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5,3.6))
                cls = ["#C5D5F5" if v < cby.mean() else "#364C84" for v in cby.values]
                bars = ax.barh(cby.index,cby.values,color=cls,edgecolor='none',height=0.52)
                for bar, val in zip(bars,cby.values):
                    ax.text(bar.get_width()+0.2,bar.get_y()+bar.get_height()/2,
                            f'{val:.1f}%',va='center',color='#667085',fontsize=9)
                ax.set_xlabel("Churn Rate (%)")
                ax.spines['left'].set_visible(False)
                st.pyplot(fig); plt.close()
        with col2:
            with st.container(border=True):
                st.markdown('<div class="cc-title">Volume per Segmen</div>', unsafe_allow_html=True)
                cnt = df.groupby([cat_feature,"churn"]).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(5,3.6))
                x = np.arange(len(cnt)); w = 0.38
                ax.bar(x-w/2,cnt[0],w,label="Tidak Churn",color="#364C84",edgecolor='none')
                ax.bar(x+w/2,cnt[1],w,label="Churn",      color="#F04438",edgecolor='none')
                ax.set_xticks(x); ax.set_xticklabels(cnt.index,rotation=12,ha='right',fontsize=8)
                ax.legend(fontsize=8,frameon=True,framealpha=0.9)
                st.pyplot(fig); plt.close()

        pivot = df.groupby(cat_feature).agg(
            Total     =("churn","count"),
            Churn     =("churn","sum"),
            Churn_Rate=("churn",lambda x: f"{x.mean()*100:.1f}%"),
            Avg_LTV   =("lifetime_value",lambda x: f"${x.mean():,.0f}"),
            Avg_Spent =("total_spent",lambda x: f"${x.mean():,.0f}")
        ).reset_index()
        st.dataframe(pivot, use_container_width=True, hide_index=True)

    with tab4:
        with st.container(border=True):
            st.markdown('<div class="cc-title">Feature Importance — Tuned Random Forest</div><div class="cc-sub">Diambil langsung dari Cell 82 di notebook</div>', unsafe_allow_html=True)
            if all([model_ok, scaler_ok, feat_ok]):
                model_t, scaler_t, feat_t = load_artifacts()
                importances = model_t.feature_importances_
                indices = np.argsort(importances)[::-1]
                fi_df = pd.DataFrame({
                    'Fitur': [feat_t[i] for i in indices],
                    'Importance': importances[indices]
                })
                
                import matplotlib as mpl
                rc_backup = mpl.rcParams.copy()
                try:
                    mpl.rcParams.update(mpl.rcParamsDefault)
                    import seaborn as sns
                    sns.set_theme(style="whitegrid")
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    colors  = ['#e74c3c' if i < 5 else '#3498db' if i < 10 else '#95a5a6'
                               for i in range(20)]

                    bars = ax.barh(
                        fi_df['Fitur'].head(20)[::-1],
                        fi_df['Importance'].head(20)[::-1],
                        color=colors[::-1], edgecolor='black', alpha=0.85
                    )
                    for bar, val in zip(bars, fi_df['Importance'].head(20)[::-1]):
                        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                                f'{val:.4f}', va='center', fontsize=9)

                    ax.set_xlabel('Feature Importance Score')
                    ax.set_title('Top 20 Feature Importance — Random Forest',
                                 fontsize=13, fontweight='bold')
                    mean_val = fi_df['Importance'].mean()
                    ax.axvline(x=mean_val, color='red',
                               linestyle='--', linewidth=1.5,
                               label=f'Mean = {mean_val:.4f}')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                finally:
                    mpl.rcParams.update(rc_backup)
            else:
                st.warning("Model artifacts tidak ditemukan. Jalankan model terlebih dahulu.")

        with st.container(border=True):
            st.markdown('<div class="cc-title">Confusion Matrix — Direct Modeling</div><div class="cc-sub">Diambil langsung dari Cell 38 di notebook</div>', unsafe_allow_html=True)
            import matplotlib as mpl
            rc_backup = mpl.rcParams.copy()
            try:
                mpl.rcParams.update(mpl.rcParamsDefault)
                import seaborn as sns
                sns.set_theme(style="whitegrid")
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                short_names = ['Logistic\nRegression', 'Random\nForest', 'Voting\nClassifier']
                cms_direct = [
                    np.array([[2011, 50], [329, 60]]),
                    np.array([[1892, 169], [229, 160]]),
                    np.array([[2050, 11], [369, 20]])
                ]
                accuracies_direct = [0.8453, 0.8376, 0.8449]
                
                for ax, cm, acc, short in zip(axes, cms_direct, accuracies_direct, short_names):
                    sns.heatmap(
                        cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Pred: No Churn', 'Pred: Churn'],
                        yticklabels=['Act: No Churn', 'Act: Churn'],
                        ax=ax, linewidths=0.5
                    )
                    ax.set_title(f'{short}\nAccuracy: {acc:.4f}', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')

                plt.suptitle('Confusion Matrix — Direct Modeling', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            finally:
                mpl.rcParams.update(rc_backup)

        with st.container(border=True):
            st.markdown('<div class="cc-title">Confusion Matrix — Tuned Models</div><div class="cc-sub">Diambil langsung dari Cell 92 di notebook</div>', unsafe_allow_html=True)
            rc_backup = mpl.rcParams.copy()
            try:
                mpl.rcParams.update(mpl.rcParamsDefault)
                import seaborn as sns
                sns.set_theme(style="whitegrid")
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                short_names = ['Logistic\nRegression\n(Tuned)',
                               'Random\nForest\n(Tuned)',
                               'Voting\nClassifier\n(Tuned)']
                cms_tuned = [
                    np.array([[1942, 599], [113, 346]]),
                    np.array([[2115, 426], [2, 457]]),
                    np.array([[2449, 92], [375, 84]])
                ]
                accuracies_tuned = [0.7627, 0.8570, 0.8440]
                
                for ax, cm, acc, short in zip(axes, cms_tuned, accuracies_tuned, short_names):
                    sns.heatmap(
                        cm, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=['Pred: No Churn', 'Pred: Churn'],
                        yticklabels=['Act: No Churn', 'Act: Churn'],
                        ax=ax, linewidths=0.5
                    )
                    ax.set_title(f'{short}\nAccuracy: {acc:.4f}', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')

                plt.suptitle('Confusion Matrix — Hyperparameter Tuning', fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            finally:
                mpl.rcParams.update(rc_backup)
