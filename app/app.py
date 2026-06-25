import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
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
.stRadio > div             { gap:4px !important;flex-direction:column !important; }
.stRadio label             { border-radius:10px !important;padding:0.55rem 0.8rem !important;font-size:0.88rem !important;font-weight:500 !important;color:#475467 !important;width:100% !important;transition:all 0.2s ease !important; border: 1px solid transparent; }
.stRadio label:hover       { background:#EEF3FC !important;color:#364C84 !important; }

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
.ccard       { background:#FFFFFF;border-radius:16px;padding:1.4rem 1.6rem;border:1px solid #EBF0FA;box-shadow:0 4px 20px rgba(54,76,132,.03),0 1px 2px rgba(54,76,132,.04);margin-bottom:1rem; }
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

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] { background:transparent !important;border-bottom:1.5px solid #EBF0FA !important;gap:0 !important; }
.stTabs [data-baseweb="tab"]      { background:transparent !important;color:#667085 !important;font-size:.9rem !important;font-weight:500 !important;padding:.65rem 1.3rem !important; transition: all 0.2s; }
.stTabs [aria-selected="true"]    { color:#364C84 !important;font-weight:600 !important;border-bottom:2px solid #364C84 !important; }

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
    df = pd.DataFrame([raw])
    df["subscription_type"] = df["subscription_type"].map({"Monthly": 0, "Annual": 1})
    df["gender"]             = df["gender"].map({"Female": 0, "Male": 1, "Other": 2})
    ohe = ["country", "city", "acquisition_channel", "device_type", "payment_method", "coupon_code"]
    df  = pd.get_dummies(df, columns=ohe, drop_first=True)
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(int)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names].values

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
    import matplotlib.pyplot as plt
    import matplotlib
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

    st.markdown("""
    <div class="ccard">
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
        fig, ax = plt.subplots(figsize=(6, 3.2))
        model_names = ['Logistic\nRegression', 'Random\nForest', 'Voting\nClassifier']
        f1_scores = [49.14, 67.96, 13.33]
        colors = ['#C5D5F5', '#364C84', '#F04438']
        
        bars = ax.bar(model_names, f1_scores, color=colors, edgecolor='none', width=0.45, zorder=3)
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.2f}%', ha='center', color='#101828', fontsize=10, fontweight='600')
        
        ax.set_ylabel("F1-Score (%)", fontsize=9)
        ax.set_ylim(0, 85)
        ax.grid(axis='y', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        st.pyplot(fig, transparent=True)
        plt.close()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="ccard"><div class="cc-title">Distribusi Churn</div><div class="cc-sub">Proporsi pelanggan churn vs tidak churn</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        ax.pie([total - churn_n, churn_n], labels=["Tidak Churn","Churn"],
               colors=["#364C84","#F04438"], autopct="%1.1f%%", startangle=90,
               wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2.5),
               textprops=dict(color='#344054', fontsize=10))
        st.pyplot(fig, transparent=True); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="ccard"><div class="cc-title">Churn Rate per Acquisition Channel</div><div class="cc-sub">Saluran akuisisi dengan risiko churn tertinggi</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        cg  = df.groupby("acquisition_channel")["churn"].mean().sort_values() * 100
        cls = ["#C5D5F5" if v < cg.mean() else "#364C84" for v in cg.values]
        bars = ax.barh(cg.index, cg.values, color=cls, edgecolor='none', height=0.52)
        for bar, val in zip(bars, cg.values):
            ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='#667085', fontsize=9)
        ax.set_xlabel("Churn Rate (%)", fontsize=9)
        ax.grid(axis='x', alpha=0.35); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        st.pyplot(fig, transparent=True); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="ccard"><div class="cc-title">Churn per Subscription Type</div><div class="cc-sub">Monthly vs Annual</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        sc = df.groupby("subscription_type")["churn"].mean() * 100
        bars = ax.bar(sc.index, sc.values, color=["#364C84","#E7F1A8"], edgecolor='none', width=0.42, zorder=3)
        for bar, val in zip(bars, sc.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val:.1f}%', ha='center', color='#101828', fontsize=11, fontweight='600')
        ax.set_ylabel("Churn Rate (%)", fontsize=9)
        ax.grid(axis='y', alpha=0.35, zorder=0); ax.set_axisbelow(True)
        st.pyplot(fig, transparent=True); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="ccard"><div class="cc-title">Distribusi Usia by Churn</div><div class="cc-sub">Sebaran usia pelanggan churn vs tidak churn</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        ax.hist(df[df["churn"]==0]["age"].dropna(), bins=28, alpha=0.75, color="#364C84", label="Tidak Churn", edgecolor='none')
        ax.hist(df[df["churn"]==1]["age"].dropna(), bins=28, alpha=0.75, color="#F04438", label="Churn",       edgecolor='none')
        ax.set_xlabel("Usia", fontsize=9); ax.set_ylabel("Jumlah", fontsize=9)
        ax.legend(fontsize=8, frameon=True, framealpha=0.9)
        ax.grid(axis='y', alpha=0.35); ax.set_axisbelow(True)
        st.pyplot(fig, transparent=True); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Data table ───────────────────────────
    st.markdown('<div class="ccard"><div class="cc-title">Recent Activities — Sample Data</div><div class="cc-sub">10 baris pertama dari dataset</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


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

    tab1, tab2, tab3 = st.tabs(["📈 Distribusi Fitur", "🔗 Korelasi", "🎯 Analisis Churn"])

    with tab1:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected = st.selectbox("Pilih fitur numerik:", num_cols)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="ccard"><div class="cc-title">Histogram</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.6))
            ax.hist(df[selected].dropna(), bins=35, color="#364C84", alpha=0.82, edgecolor='none')
            ax.set_title(f"Distribusi: {selected}", fontsize=10, fontweight='600', color='#101828')
            ax.set_xlabel(selected,fontsize=9); ax.set_ylabel("Frekuensi",fontsize=9)
            ax.grid(axis='y',alpha=0.35); ax.set_axisbelow(True)
            st.pyplot(fig,transparent=True); plt.close()
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="ccard"><div class="cc-title">Boxplot by Churn</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.6))
            ax.boxplot([df[df["churn"]==0][selected].dropna(), df[df["churn"]==1][selected].dropna()],
                       patch_artist=True,
                       boxprops=dict(facecolor='#EEF3FC',color='#C5D5F5'),
                       medianprops=dict(color='#364C84',linewidth=2.5),
                       whiskerprops=dict(color='#D1D5DB'), capprops=dict(color='#D1D5DB'),
                       flierprops=dict(marker='o',color='#F04438',alpha=0.3,markersize=3))
            ax.set_xticklabels(["Tidak Churn","Churn"])
            ax.set_title(f"{selected} per Kelompok",fontsize=10,fontweight='600',color='#101828')
            ax.grid(axis='y',alpha=0.35); ax.set_axisbelow(True)
            st.pyplot(fig,transparent=True); plt.close()
            st.markdown("</div>", unsafe_allow_html=True)
        st.dataframe(df[selected].describe().to_frame().T, use_container_width=True)

    with tab2:
        num_df = df.select_dtypes(include=np.number).drop(columns=["customer_id"],errors="ignore")
        corr   = num_df.corr()
        st.markdown('<div class="ccard"><div class="cc-title">Heatmap Korelasi Fitur Numerik</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(18,14))
        sns.heatmap(corr,
                    annot=True, fmt=".2f", ax=ax, cmap="RdYlGn", center=0,
                    linewidths=0.5, annot_kws={"size":7}, cbar_kws={"shrink":0.75})
        ax.set_title("Heatmap Korelasi Fitur Numerik", fontsize=13, fontweight='600', color='#101828', pad=15)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0,  labelsize=9)
        plt.tight_layout()
        st.pyplot(fig, transparent=True, use_container_width=True); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ccard"><div class="cc-title">Korelasi dengan Target (Churn)</div>', unsafe_allow_html=True)
        corr_t = num_df.corr()["churn"].drop("churn").sort_values(key=abs,ascending=False)
        fig, ax = plt.subplots(figsize=(9,5))
        clr = ["#F04438" if v > 0 else "#364C84" for v in corr_t.values]
        ax.barh(corr_t.index[::-1],corr_t.values[::-1],color=clr[::-1],edgecolor='none',height=0.62)
        ax.axvline(0,color='#D1D5DB',linewidth=1)
        ax.set_title("Korelasi Fitur vs Churn",fontsize=10,fontweight='600',color='#101828')
        ax.grid(axis='x',alpha=0.35); ax.set_axisbelow(True)
        st.pyplot(fig,transparent=True); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        cat_feature = st.selectbox("Pilih fitur kategorikal:",
                                   ["acquisition_channel","device_type","subscription_type",
                                    "payment_method","country","gender"])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="ccard"><div class="cc-title">Churn Rate per Segmen</div>', unsafe_allow_html=True)
            cby  = df.groupby(cat_feature)["churn"].mean().sort_values() * 100
            fig, ax = plt.subplots(figsize=(5,3.6))
            cls = ["#C5D5F5" if v < cby.mean() else "#364C84" for v in cby.values]
            bars = ax.barh(cby.index,cby.values,color=cls,edgecolor='none',height=0.52)
            for bar, val in zip(bars,cby.values):
                ax.text(bar.get_width()+0.2,bar.get_y()+bar.get_height()/2,
                        f'{val:.1f}%',va='center',color='#667085',fontsize=9)
            ax.set_xlabel("Churn Rate (%)"); ax.grid(axis='x',alpha=0.35); ax.set_axisbelow(True)
            ax.spines['left'].set_visible(False)
            st.pyplot(fig,transparent=True); plt.close()
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="ccard"><div class="cc-title">Volume per Segmen</div>', unsafe_allow_html=True)
            cnt = df.groupby([cat_feature,"churn"]).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(5,3.6))
            x = np.arange(len(cnt)); w = 0.38
            ax.bar(x-w/2,cnt[0],w,label="Tidak Churn",color="#364C84",edgecolor='none')
            ax.bar(x+w/2,cnt[1],w,label="Churn",      color="#F04438",edgecolor='none')
            ax.set_xticks(x); ax.set_xticklabels(cnt.index,rotation=12,ha='right',fontsize=8)
            ax.legend(fontsize=8,frameon=True,framealpha=0.9)
            ax.grid(axis='y',alpha=0.35); ax.set_axisbelow(True)
            st.pyplot(fig,transparent=True); plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

        pivot = df.groupby(cat_feature).agg(
            Total     =("churn","count"),
            Churn     =("churn","sum"),
            Churn_Rate=("churn",lambda x: f"{x.mean()*100:.1f}%"),
            Avg_LTV   =("lifetime_value",lambda x: f"${x.mean():,.0f}"),
            Avg_Spent =("total_spent",lambda x: f"${x.mean():,.0f}")
        ).reset_index()
        st.dataframe(pivot, use_container_width=True, hide_index=True)
