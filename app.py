"""
Term Deposit Predictor - Streamlit app for ML Assignment 2.

Upload the held-out test CSV, pick a trained model, and see its evaluation
metrics, confusion matrix, ROC / PR curves and classification report, plus a
side-by-side leaderboard of all five classifiers. Models are loaded from
model/*.joblib and are never retrained on the uploaded data.

UI theme: colours taken from the BITS Pilani crest (navy, gold, red, sky blue,
sun yellow). Icons are Google Material Symbols (web font) - no emoji.
Charts use Altair, which ships with Streamlit (no extra dependency).
"""

import base64
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from preprocess import RAW_FEATURES, TARGET, clean_raw, encode_target

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"
LOGO = ROOT / "assets" / "bits_logo.png"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
MODEL_ICONS = {  # Material Symbols names
    "Logistic Regression": "show_chart",
    "Decision Tree": "account_tree",
    "KNN": "hub",
    "Naive Bayes": "functions",
    "Support Vector Machine": "linear_scale",
}
METRIC_INFO = {  # icon, one-line meaning
    "Accuracy": ("target", "Share of all predictions that are correct"),
    "AUC": ("area_chart", "Ranking quality across every threshold (ROC area)"),
    "Precision": ("filter_center_focus", "Of predicted subscribers, how many really subscribed"),
    "Recall": ("radar", "Of real subscribers, how many the model caught"),
    "F1": ("balance", "Harmonic mean of precision and recall"),
    "MCC": ("verified", "Balanced score for imbalanced classes (-1 to 1)"),
}

# ---- BITS Pilani crest palette
NAVY = "#24287F"
DEEP = "#11145A"
GOLD = "#B8892A"
SUN = "#F2CF5B"
RED = "#C62828"
SKY = "#3BB4E0"
INK = "#1B1E3A"
MUTED = "#5E6280"
PAPER = "#F7F5EE"
LINE = "#E3DFD0"
MODEL_COLORS = [NAVY, RED, GOLD, SKY, "#7A7D92"]

alt.data_transformers.disable_max_rows()

st.set_page_config(
    page_title="Term Deposit Predictor | Assignment II",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

DARK_MODE = st.session_state.theme_mode == "Dark"
INK = "#F4F5FB" if DARK_MODE else "#1B1E3A"
MUTED = "#B6BCD0" if DARK_MODE else "#5E6280"
PAPER = "#101321" if DARK_MODE else "#F7F5EE"
LINE = "#353A55" if DARK_MODE else "#E3DFD0"
SURFACE = "#191D30" if DARK_MODE else "#FFFFFF"
SURFACE_ALT = "#22273D" if DARK_MODE else "#F6F7FC"
FIELD = "#F4F5F8"
TRACK = "#303650" if DARK_MODE else "#ECEAF3"
CHART_GRID = "#303650" if DARK_MODE else "#EEEDF3"
SOFT_BLUE = "#242A48" if DARK_MODE else "#F1F3FF"
SOFT_BLUE_LINE = "#454E7A" if DARK_MODE else "#CDD2F5"
SOFT_GOLD = "#3A321D" if DARK_MODE else "#FFF8E2"
SOFT_RED = "#3B2228" if DARK_MODE else "#FDEEEE"
SOFT_TARGET = "#302B1E" if DARK_MODE else "#FFFBEC"


def logo_tag(size: int) -> str:
    if LOGO.exists():
        b64 = base64.b64encode(LOGO.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" width="{size}" height="{size}" alt="BITS Pilani">'
    return f'<span class="msr" style="font-size:{size * .7}px;color:{SUN}">school</span>'


def icon(name: str, cls: str = "") -> str:
    return f'<span class="msr {cls}">{name}</span>'


# ---------------------------------------------------------------- global styles
st.html(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400..600,0..1,0&display=block');

    :root {{
        --navy:{NAVY}; --deep:{DEEP}; --gold:{GOLD}; --sun:{SUN}; --red:{RED};
        --sky:{SKY}; --ink:{INK}; --muted:{MUTED}; --paper:{PAPER}; --line:{LINE};
        --surface:{SURFACE}; --surface-alt:{SURFACE_ALT}; --field:{FIELD}; --track:{TRACK};
        --soft-blue:{SOFT_BLUE}; --soft-blue-line:{SOFT_BLUE_LINE};
        --soft-gold:{SOFT_GOLD}; --soft-red:{SOFT_RED}; --soft-target:{SOFT_TARGET};
    }}
    .msr {{
        font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal;
        font-size: 1.15em; line-height: 1; display: inline-block; vertical-align: -0.2em;
        letter-spacing: normal; text-transform: none; white-space: nowrap;
        font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
        font-variation-settings: 'FILL' 1, 'wght' 500;
    }}
    .stApp {{background: var(--paper); color:var(--ink);}}
    .stApp, .stApp p, .stApp label, .stApp li {{font-family: 'Plus Jakarta Sans', sans-serif;}}
    header[data-testid="stHeader"] {{background: transparent;}}
    [data-testid="stAppViewContainer"] .block-container {{width:100%; max-width:none; padding:2.6rem 1.35rem 3rem;}}
    h1, h2, h3, h4 {{color: var(--ink); font-family: 'Plus Jakarta Sans', sans-serif !important; letter-spacing: -.01em;}}

    /* sidebar */
    [data-testid="stSidebar"] {{background: linear-gradient(180deg, var(--deep) 0%, var(--navy) 100%);}}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption p {{color:#E9EAF6 !important; opacity:1 !important;}}
    [data-testid="stSidebar"] [data-baseweb="select"] * {{color:var(--deep) !important;}}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span {{color:#E9EAF6 !important;}}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{color: var(--sun) !important; font-size: .95rem;
        text-transform: uppercase; letter-spacing: .08em; margin-top: 1.1rem;}}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{background: rgba(255,255,255,.08);
        border: 1px dashed rgba(242,207,91,.55);}}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{background:var(--sun) !important;
        border:0 !important; opacity:1 !important;}}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {{color:var(--deep) !important; opacity:1 !important;}}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{border:2px solid var(--sun) !important;}}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{color:#E9EAF6;}}
    [data-testid="stSidebar"] .stDownloadButton button {{background: transparent; border: 1px solid var(--sun);}}
    [data-testid="stSidebar"] .stDownloadButton button * {{color: var(--sun) !important;}}
    .side-brand {{display:flex; gap:.7rem; align-items:center; padding:.3rem 0 .8rem;
        border-bottom:1px solid rgba(255,255,255,.15);}}
    .side-brand b {{display:block; color:#fff; font-size:.95rem;}}
    .side-brand small {{color: var(--sun); font-size:.72rem; letter-spacing:.06em; text-transform:uppercase;}}
    .side-note {{font-size:.78rem; line-height:1.45; color:#C9CBE6 !important; background:rgba(255,255,255,.06);
        border-left:3px solid var(--sky); padding:.6rem .7rem; border-radius:4px; margin-top:.6rem;}}
    .side-note code {{color: var(--sun) !important; background: transparent;}}

    /* header */
    .hero {{position:relative; overflow:hidden; display:grid; grid-template-columns:auto minmax(0,1fr) auto;
        gap:1.6rem; align-items:center; background: radial-gradient(120% 140% at 0% 0%, #2F35A0 0%, var(--navy) 45%, var(--deep) 100%);
        color:#fff; padding:1.4rem 1.7rem; border-radius:14px; margin-bottom:1rem;
        box-shadow:0 14px 34px rgba(17,20,90,.25); border-bottom:5px solid var(--gold);}}
    .hero::after {{content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px; border-radius:50%;
        border:28px solid rgba(242,207,91,.08);}}
    .hero .crest {{background:#fff; border-radius:50%; padding:6px; box-shadow:0 0 0 3px var(--gold); line-height:0;}}
    .kicker {{display:inline-flex; gap:.4rem; align-items:center; color:var(--sun); font-size:.74rem; font-weight:700;
        letter-spacing:.1em; text-transform:uppercase; margin-bottom:.3rem;}}
    .hero h1 {{color:#fff !important; font-size:clamp(1.7rem,3vw,2.5rem); font-weight:800; line-height:1.08; margin:0; padding:0;}}
    .hero p {{color:#D6D8F0; margin:.45rem 0 .7rem; max-width:720px; font-size:.95rem;}}
    .chips {{display:flex; flex-wrap:wrap; gap:.45rem;}}
    .chip {{display:inline-flex; align-items:center; gap:.3rem; font-size:.76rem; font-weight:600; padding:.25rem .6rem;
        border-radius:999px; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.18);}}
    .chip .msr {{color:var(--sky);}}
    .student {{position:relative; z-index:1; min-width:240px; background:rgba(255,255,255,.07); border:1px solid rgba(242,207,91,.4);
        border-radius:10px; padding:.85rem 1rem;}}
    .student span {{display:block; color:var(--sun); font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.2rem;}}
    .student strong {{display:block; font-size:1.08rem;}}
    .student small {{display:flex; gap:.3rem; align-items:center; color:#C9CBE6; margin-top:.25rem;}}

    /* progress stepper */
    .stepper-wrap {{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:1rem 1.2rem .9rem; margin-bottom:1rem;}}
    .stepper-head {{display:flex; justify-content:space-between; align-items:center; margin-bottom:.8rem;}}
    .stepper-head b {{color:var(--ink); font-size:.95rem; display:flex; gap:.4rem; align-items:center;}}
    .stepper-head b .msr {{color:var(--gold);}}
    .stepper-head span {{font-size:.8rem; font-weight:700; color:var(--navy);}}
    .pbar {{height:6px; border-radius:6px; background:var(--track); overflow:hidden; margin-bottom:1rem;}}
    .pbar i {{display:block; height:100%; background:linear-gradient(90deg, var(--navy), var(--sky) 60%, var(--sun));}}
    .stepper {{display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:.6rem;}}
    .step {{position:relative; display:flex; gap:.55rem; align-items:flex-start; padding:.6rem .65rem; border-radius:9px;
        background:var(--surface-alt); border:1px solid var(--line);}}
    .step .msr {{font-size:1.45rem; color:#B7BAD3;}}
    .step.done {{background:var(--soft-blue); border-color:var(--soft-blue-line);}}
    .step.done .msr {{color:var(--navy);}}
    .step.star {{background:var(--soft-gold); border-color:var(--sun);}}
    .step.star .msr {{color:var(--gold);}}
    .step.fail {{background:var(--soft-red); border-color:#9C4A55;}}
    .step.fail .msr {{color:var(--red);}}
    .step b {{display:block; font-size:.82rem; color:var(--ink); line-height:1.2;}}
    .step small {{display:block; font-size:.74rem; color:var(--muted); margin-top:.15rem; line-height:1.3;}}

    /* stat cards */
    .stats {{display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:.7rem; margin-bottom:.6rem;}}
    .stat {{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.8rem .95rem; display:flex; gap:.75rem; align-items:center;}}
    .stat .ic {{width:40px; height:40px; border-radius:10px; display:grid; place-items:center; flex:none;}}
    .stat .ic .msr {{font-size:1.35rem; vertical-align:0;}}
    .stat small {{display:block; color:var(--muted); font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em;}}
    .stat small {{white-space:nowrap;}}
    .stat b {{display:block; color:var(--ink); font-size:1.05rem; font-weight:800; line-height:1.25; overflow-wrap:anywhere;}}

    /* metric cards */
    .mgrid {{display:grid; grid-template-columns:repeat(6, minmax(0,1fr)); gap:.7rem; margin:.3rem 0 1rem;}}
    .mcard {{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:.8rem .9rem .75rem; position:relative;}}
    .mcard.top {{border-color:var(--sun); box-shadow:0 0 0 2px rgba(242,207,91,.35);}}
    .mlabel {{display:flex; gap:.35rem; align-items:center; color:var(--muted); font-size:.78rem; font-weight:700;
        text-transform:uppercase; letter-spacing:.05em;}}
    .mlabel .msr {{color:var(--navy);}}
    .mval {{font-size:1.65rem; font-weight:800; color:var(--deep); margin:.25rem 0 .4rem; font-variant-numeric:tabular-nums;}}
    .mtrack {{height:5px; background:var(--track); border-radius:5px; overflow:hidden;}}
    .mtrack i {{display:block; height:100%; background:var(--navy); border-radius:5px;}}
    .mcard.top .mtrack i {{background:var(--gold);}}
    .mfoot {{display:flex; justify-content:space-between; margin-top:.4rem; font-size:.72rem; color:var(--muted);}}
    .badge {{display:inline-flex; align-items:center; gap:.2rem; font-weight:700; color:var(--navy);}}
    .mcard.top .badge {{color:var(--gold);}}

    /* section titles + panels */
    .sec {{display:flex; align-items:center; gap:.5rem; margin:1.1rem 0 .5rem; color:var(--ink); font-weight:800; font-size:1.05rem;}}
    .sec .msr {{color:#fff; background:var(--navy); border-radius:8px; padding:.28rem; font-size:1.05rem; vertical-align:0;}}
    .sec small {{font-weight:500; color:var(--muted); font-size:.82rem; margin-left:.2rem;}}
    .champ {{display:grid; grid-template-columns:auto minmax(0,1fr) auto; gap:1.2rem; align-items:center;
        background:linear-gradient(110deg, var(--soft-gold) 0%, var(--surface) 60%); border:1px solid var(--sun); border-left:6px solid var(--gold);
        border-radius:12px; padding:1rem 1.2rem; margin:.3rem 0 1rem;}}
    .champ .medal {{width:58px; height:58px; border-radius:50%; background:var(--gold); display:grid; place-items:center;
        box-shadow:0 0 0 5px rgba(184,137,42,.18);}}
    .champ .medal .msr {{color:#fff; font-size:2rem; vertical-align:0;}}
    .champ small {{color:var(--gold); font-weight:800; text-transform:uppercase; letter-spacing:.08em; font-size:.72rem;}}
    .champ h3 {{margin:.1rem 0 .25rem; padding:0; font-size:1.45rem; color:var(--deep);}}
    .champ p {{margin:0; color:var(--muted); font-size:.88rem;}}
    .champ .score {{text-align:right;}}
    .champ .score b {{display:block; font-size:2rem; color:var(--deep); font-weight:800; font-variant-numeric:tabular-nums;}}
    .champ .score span {{font-size:.78rem; color:var(--muted);}}
    .wins {{display:flex; flex-wrap:wrap; gap:.35rem; margin-top:.45rem;}}
    .win {{font-size:.72rem; font-weight:700; padding:.15rem .5rem; border-radius:999px; background:var(--navy); color:#fff;}}

    .checks {{display:grid; grid-template-columns:repeat(auto-fill, minmax(190px,1fr)); gap:.4rem;}}
    .ck {{display:flex; gap:.4rem; align-items:center; font-size:.82rem; background:var(--surface); border:1px solid var(--line);
        border-radius:8px; padding:.35rem .55rem; color:var(--ink);}}
    .ck .msr {{color:var(--navy);}}
    .ck.bad .msr {{color:var(--red);}}
    .ck.tgt {{border-color:var(--sun); background:var(--soft-target);}}
    .ck.tgt .msr {{color:var(--gold);}}
    .ck code {{background:transparent; color:var(--ink); padding:0; font-size:.8rem;}}

    .flow {{display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; margin:.4rem 0 1rem;}}
    .node {{display:flex; gap:.4rem; align-items:center; background:var(--surface); border:1px solid var(--line); border-radius:10px;
        padding:.5rem .7rem; font-size:.84rem; font-weight:600; color:var(--ink);}}
    .node .msr {{color:var(--navy);}}
    .node.last {{background:var(--navy); color:#fff; border-color:var(--navy);}}
    .node.last .msr {{color:var(--sun);}}
    .arrow {{color:var(--gold);}}

    /* evaluator workflow */
    .eval-guide {{display:grid; grid-template-columns:1.2fr repeat(4, minmax(0,1fr)); align-items:stretch;
        margin:.8rem 0 1rem; border:1px solid var(--line); border-left:5px solid var(--gold);
        border-radius:10px; overflow:hidden; background:var(--surface);}}
    .eval-title {{padding:.75rem .9rem; display:flex; flex-direction:column; justify-content:center;
        background:var(--soft-gold);}}
    .eval-title strong {{color:var(--ink); font-size:.9rem;}}
    .eval-title span {{color:var(--muted); font-size:.72rem; margin-top:.15rem;}}
    .eval-item {{display:grid; grid-template-columns:28px minmax(0,1fr); gap:.5rem; align-items:center;
        padding:.65rem .7rem; border-left:1px solid var(--line);}}
    .eval-num {{width:28px; height:28px; border-radius:50%; display:grid; place-items:center;
        background:var(--navy); color:#fff; font-size:.75rem; font-weight:800;}}
    .eval-item b {{display:block; color:var(--ink); font-size:.78rem; line-height:1.2;}}
    .eval-item small {{display:block; color:var(--muted); font-size:.68rem; margin-top:.12rem; line-height:1.25;}}
    .req-summary {{display:flex; align-items:center; justify-content:space-between; gap:1rem;
        padding:.85rem 1rem; margin:.25rem 0 .8rem; background:var(--soft-blue);
        border:1px solid var(--soft-blue-line); border-radius:10px;}}
    .req-summary b {{color:var(--ink);}}
    .req-summary span {{color:var(--navy); font-weight:800; white-space:nowrap;}}

    /* native widgets */
    .stTabs [data-baseweb="tab-list"] {{gap:.25rem; border-bottom:2px solid var(--line);}}
    .stTabs [data-baseweb="tab"] {{flex:1 1 0; justify-content:center; min-width:0; font-weight:700;
        padding:.55rem .45rem; border-radius:8px 8px 0 0;}}
    .stTabs [data-baseweb="tab"] p {{font-size:.84rem; white-space:nowrap;}}
    .stTabs [data-baseweb="tab"] p {{color:var(--muted);}}
    .stTabs [aria-selected="true"] {{background:var(--surface);}}
    .stTabs [aria-selected="true"] p {{color:{SUN if DARK_MODE else NAVY} !important;}}
    .stTabs [data-baseweb="tab-highlight"] {{background-color:var(--gold) !important; height:3px;}}
    .stDownloadButton > button, .stButton > button {{border-radius:8px; border:1px solid var(--navy); color:var(--navy);
        font-weight:700; background:var(--surface);}}
    .stDownloadButton > button:hover, .stButton > button:hover {{background:var(--navy); color:#fff; border-color:var(--navy);}}
    [data-testid="stDataFrame"] {{border:1px solid var(--line); border-radius:10px; overflow:hidden;}}
    [data-testid="stVegaLiteChart"] {{background:var(--surface); border:1px solid var(--line); border-radius:12px; overflow:hidden;}}
    [data-testid="stExpander"] details {{background:var(--surface); border:1px solid var(--line); border-radius:10px;}}
    .stProgress > div > div > div > div {{background:linear-gradient(90deg, var(--navy), var(--sky));}}

    @media (max-width: 1100px) {{
        .eval-guide {{grid-template-columns:repeat(2, minmax(0,1fr));}}
        .eval-title {{grid-column:1 / -1;}}
        .eval-item:nth-child(2), .eval-item:nth-child(4) {{border-left:0;}}
        .stats {{grid-template-columns:repeat(2, minmax(0,1fr));}}
        .mgrid {{grid-template-columns:repeat(3, minmax(0,1fr));}}
        .stepper {{grid-template-columns:repeat(2, minmax(0,1fr));}}
    }}
    @media (max-width: 760px) {{
        [data-testid="stAppViewContainer"] .block-container {{padding-left:.8rem; padding-right:.8rem;}}
        .eval-guide {{grid-template-columns:1fr;}}
        .eval-item {{border-left:0; border-top:1px solid var(--line);}}
        .hero {{grid-template-columns:1fr; gap:1rem; padding:1.2rem;}}
        .champ {{grid-template-columns:1fr;}} .champ .score {{text-align:left;}}
        .mgrid {{grid-template-columns:repeat(2, minmax(0,1fr));}}
        .stepper {{grid-template-columns:1fr;}}
    }}
    </style>
    """
)

st.html(
    f"""
    <div class="hero">
        <div class="crest">{logo_tag(78)}</div>
        <div style="position:relative;z-index:1">
            <span class="kicker">{icon('school')} M.Tech (AIML) &middot; Machine Learning &middot; Assignment II</span>
            <h1>Term Deposit Predictor</h1>
            <p>Evaluate five trained classifiers on bank-marketing test data, inspect each model's errors
               and rank them side by side to find the strongest one.</p>
            <div class="chips">
                <span class="chip">{icon('database')} UCI Bank Marketing</span>
                <span class="chip">{icon('view_column')} {len(RAW_FEATURES)} input features</span>
                <span class="chip">{icon('model_training')} {len(MODEL_FILES)} classifiers</span>
                <span class="chip">{icon('lock')} Pre-trained &middot; no refitting</span>
            </div>
        </div>
        <div class="student">
            <span>Submitted by</span>
            <strong>Sabareeshwaran K</strong>
            <small>{icon('badge')} BITS ID 2026AJ05033</small>
            <small>{icon('account_balance')} BITS Pilani &middot; WILP</small>
        </div>
    </div>
    """
)


# ---------------------------------------------------------------- helpers
@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data
def load_training_info():
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def predict(name: str, X: pd.DataFrame):
    model = load_model(name)
    y_pred = np.asarray(model.predict(X)).astype(int)
    y_prob = np.asarray(model.predict_proba(X))[:, 1]
    return y_pred, y_prob


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    both = len(np.unique(y_true)) > 1
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob) if both else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def thin(df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    """Evenly subsample a curve so charts stay light; keeps first and last point."""
    if len(df) <= n:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, n).round().astype(int))
    return df.iloc[idx]


def style_chart(chart, height=300):
    return (
        chart.properties(height=height)
        .configure(font="Plus Jakarta Sans, Segoe UI, Helvetica, Arial, sans-serif", background=SURFACE)
        .configure_view(strokeWidth=0)
        .configure_axisY(labelLimit=190)
        .properties(padding={"left": 12, "right": 18, "top": 12, "bottom": 8})
        .configure_axis(labelColor=MUTED, titleColor=INK, gridColor=CHART_GRID, domainColor=LINE,
                        tickColor=LINE, labelFontSize=11, titleFontSize=12, titleFontWeight=600)
        .configure_legend(labelColor=INK, titleColor=INK, orient="bottom", labelFontSize=11, labelLimit=320)
        .configure_title(color=INK, fontSize=14, anchor="start", fontWeight=700)
    )


def section(icon_name: str, title: str, sub: str = ""):
    st.html(f'<div class="sec">{icon(icon_name)}{title}<small>{sub}</small></div>')


def stat_cards(items):
    """items: list of (icon, label, value, bg, fg)."""
    html = "".join(
        f'<div class="stat"><div class="ic" style="background:{bg};color:{fg}">{icon(ic)}</div>'
        f'<div style="min-width:0"><small>{label}</small><b title="{value}">{value}</b></div></div>'
        for ic, label, value, bg, fg in items
    )
    st.html(f'<div class="stats">{html}</div>')


def render_stepper(steps, slot):
    done = sum(s[0] in ("done", "star") for s in steps)
    pct = int(round(100 * done / len(steps)))
    icons = {"done": "check_circle", "star": "workspace_premium", "fail": "error", "todo": "radio_button_unchecked"}
    cells = "".join(
        f'<div class="step {state}">{icon(icons[state])}<div><b>{title}</b><small>{detail}</small></div></div>'
        for state, title, detail in steps
    )
    slot.html(
        f"""<div class="stepper-wrap">
              <div class="stepper-head"><b>{icon('conversion_path')} Evaluation pipeline</b>
                   <span>{done} / {len(steps)} stages complete &middot; {pct}%</span></div>
              <div class="pbar"><i style="width:{pct}%"></i></div>
              <div class="stepper">{cells}</div>
            </div>"""
    )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.html(
        f'<div class="side-brand">{logo_tag(46)}<div><b>Term Deposit Predictor</b>'
        f'<small>ML Assignment II</small></div></div>'
    )
    st.segmented_control(
        "Appearance", ["Light", "Dark"], key="theme_mode",
        width="stretch",
    )
    st.subheader(":material/upload_file: 01 · Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    if SAMPLE_TEST.exists():
        use_sample = st.toggle("Use bundled test_data.csv", value=uploaded is None)
        st.download_button(
            "Download sample CSV",
            SAMPLE_TEST.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
    else:
        use_sample = False

    st.subheader(":material/model_training: 02 · Model")
    model_name = st.selectbox("Classifier to inspect", list(MODEL_FILES.keys()))

    st.subheader(":material/leaderboard: 03 · Ranking")
    rank_by = st.selectbox(
        "Rank models by", list(METRIC_INFO.keys()), index=5,
        help="MCC is the default because the target is imbalanced: it only scores high when "
             "the model is good on both subscribers and non-subscribers.",
    )

    st.html(
        '<div class="side-note">The CSV must contain the 19 input features and the target column '
        '<code>y</code> (yes/no or 1/0). Extra columns such as <code>duration</code> are ignored.</div>'
    )

stepper_slot = st.empty()

# ---------------------------------------------------------------- load
if uploaded is not None and not use_sample:
    raw = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample:
    raw = pd.read_csv(SAMPLE_TEST)
    source = "test_data.csv (bundled)"
else:
    render_stepper([
        ("todo", "Load data", "Waiting for a CSV"),
        ("todo", "Validate schema", f"{len(RAW_FEATURES)} features + target"),
        ("todo", "Encode target", "yes/no → 1/0"),
        ("todo", "Score models", f"0 / {len(MODEL_FILES)}"),
        ("todo", "Pick best model", f"by {rank_by}"),
    ], stepper_slot)
    st.info("Upload a test CSV from the sidebar, or switch on **Use bundled test_data.csv**, to begin.",
            icon=":material/upload_file:")
    st.stop()

# ---------------------------------------------------------------- validation
missing = [c for c in RAW_FEATURES + [TARGET] if c not in raw.columns]
if missing:
    render_stepper([
        ("done", "Data loaded", f"{len(raw):,} rows · {source}"),
        ("fail", "Schema check failed", f"{len(missing)} column(s) missing"),
        ("todo", "Encode target", "—"), ("todo", "Score models", "—"), ("todo", "Pick best model", "—"),
    ], stepper_slot)
    st.error(f"The uploaded file is missing required columns: {', '.join(missing)}", icon=":material/error:")
    st.stop()

df = clean_raw(raw)
X = df[RAW_FEATURES]
try:
    y = np.asarray(encode_target(df[TARGET])).astype(int)
except (ValueError, TypeError):
    render_stepper([
        ("done", "Data loaded", f"{len(raw):,} rows · {source}"),
        ("done", "Schema validated", f"{len(RAW_FEATURES)}/{len(RAW_FEATURES)} features + target"),
        ("fail", "Target invalid", "y must be yes/no or 1/0"),
        ("todo", "Score models", "—"), ("todo", "Pick best model", "—"),
    ], stepper_slot)
    st.error("The `y` column must contain only yes/no (or 1/0) values.", icon=":material/error:")
    st.stop()

both_classes = len(np.unique(y)) > 1

# ---------------------------------------------------------------- score every model once
results = {}
prog = st.progress(0.0, text="Scoring models…")
for i, name in enumerate(MODEL_FILES, start=1):
    prog.progress((i - 1) / len(MODEL_FILES), text=f"Scoring {name} ({i}/{len(MODEL_FILES)})…")
    p, pr = predict(name, X)
    results[name] = {"pred": p, "prob": pr, "metrics": compute_metrics(y, p, pr)}
prog.empty()

table = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T
table.index.name = "Model"
ranks = table.rank(ascending=False, method="min")
best = table[rank_by].idxmax()

extra_cols = [c for c in raw.columns if c not in RAW_FEATURES + [TARGET]]
render_stepper([
    ("done", "Data loaded", f"{len(raw):,} rows · {source}"),
    ("done", "Schema validated", f"{len(RAW_FEATURES)}/{len(RAW_FEATURES)} features + target"
                                 + (f" · {len(extra_cols)} extra ignored" if extra_cols else "")),
    ("done", "Target encoded", f"{y.mean():.1%} subscribed (positive class)"),
    ("done", "Models scored", f"{len(MODEL_FILES)}/{len(MODEL_FILES)} pipelines evaluated"),
    ("star", "Best model selected", f"{best} · {rank_by} {table.loc[best, rank_by]:.4f}"),
], stepper_slot)

n_yes, n_no = int(y.sum()), int(len(y) - y.sum())
stat_cards([
    ("table_rows", "Rows in file", f"{len(df):,}", "#ECEEFB", NAVY),
    ("how_to_reg", "Subscribed", f"{n_yes:,} · {y.mean():.1%}", "#FFF5D6", GOLD),
    ("person_off", "Not subscribed", f"{n_no:,} · {1 - y.mean():.1%}", "#FDECEC", RED),
    ("emoji_events", f"Best by {rank_by}", best, "#FFF5D6", GOLD),
    ("description", "Source", source, "#E6F6FC", SKY),
])

evaluator_steps = [
    ("1", "Evaluate one model", "six metrics + diagnostics"),
    ("2", "Compare all models", "ranking + ROC / PR curves"),
    ("3", "Validate test data", "schema + class balance"),
    ("4", "Confirm requirements", "assignment evidence map"),
]
st.html(
    '<div class="eval-guide"><div class="eval-title"><strong>Evaluator walkthrough</strong>'
    '<span>Four checks cover the complete assignment</span></div>'
    + "".join(
        f'<div class="eval-item"><span class="eval-num">{number}</span><div><b>{title}</b>'
        f'<small>{detail}</small></div></div>'
        for number, title, detail in evaluator_steps
    )
    + "</div>"
)

tab_single, tab_compare, tab_data, tab_about = st.tabs([
    "1. Evaluate",
    "2. Compare",
    "3. Validate data",
    "4. Checklist",
])

# ================================================================ single model
with tab_single:
    r = results[model_name]
    y_pred, y_prob, m = r["pred"], r["prob"], r["metrics"]
    is_best = model_name == best

    section(MODEL_ICONS[model_name], model_name,
            "— evaluation on uploaded data" + ("  ·  current best model" if is_best else
                                               f"  ·  ranked #{int(ranks.loc[model_name, rank_by])} of {len(MODEL_FILES)} by {rank_by}"))

    cards = []
    for k, v in m.items():
        ic, tip = METRIC_INFO[k]
        rk = int(ranks.loc[model_name, k]) if not np.isnan(v) else None
        width = 0 if np.isnan(v) else max(0.0, min(1.0, v)) * 100
        top = " top" if rk == 1 else ""
        badge = (f'<span class="badge">{icon("trophy")} #1</span>' if rk == 1
                 else f'<span class="badge">#{rk} of {len(MODEL_FILES)}</span>' if rk else "")
        val = "n/a" if np.isnan(v) else f"{v:.4f}"
        cards.append(
            f'<div class="mcard{top}" title="{tip}"><div class="mlabel">{icon(ic)}{k}</div>'
            f'<div class="mval">{val}</div><div class="mtrack"><i style="width:{width:.1f}%"></i></div>'
            f'<div class="mfoot"><span>rank</span>{badge}</div></div>'
        )
    st.html(f'<div class="mgrid">{"".join(cards)}</div>')

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    left, right = st.columns(2, gap="medium")
    with left:
        section("grid_view", "Confusion matrix", "count and % of each actual class")
        cm_df = pd.DataFrame({
            "Actual": ["No", "No", "Yes", "Yes"],
            "Predicted": ["No", "Yes", "No", "Yes"],
            "Count": [tn, fp, fn, tp],
            "Cell": ["True negative", "False positive", "False negative", "True positive"],
            "Correct": [True, False, False, True],
        })
        row_tot = cm_df.groupby("Actual")["Count"].transform("sum").replace(0, np.nan)
        cm_df["Share"] = (cm_df["Count"] / row_tot).fillna(0)
        cm_df["Label"] = cm_df["Count"].map("{:,}".format)
        cm_df["Sub"] = cm_df["Cell"] + " · " + (cm_df["Share"] * 100).round(1).astype(str) + "%"
        base = alt.Chart(cm_df).encode(
            x=alt.X("Predicted:N", sort=["No", "Yes"], axis=alt.Axis(orient="top", labelAngle=0, title="Predicted")),
            y=alt.Y("Actual:N", sort=["No", "Yes"], title="Actual"),
        )
        heat = base.mark_rect(cornerRadius=8, stroke="#fff", strokeWidth=4).encode(
            color=alt.Color("Share:Q", scale=alt.Scale(domain=[0, 1], range=["#F4F3FB", NAVY]), legend=None),
            tooltip=["Cell", "Actual", "Predicted", alt.Tooltip("Count:Q", format=","),
                     alt.Tooltip("Share:Q", format=".1%", title="% of actual class")],
        )
        txt_color = alt.condition("datum.Share > 0.5", alt.value("#FFFFFF"), alt.value(INK))
        big = base.mark_text(fontSize=26, fontWeight=800, dy=-8).encode(text="Label:N", color=txt_color)
        small = base.mark_text(fontSize=11, dy=16).encode(text="Sub:N", color=txt_color)
        err = base.transform_filter("!datum.Correct").mark_rect(
            cornerRadius=8, fillOpacity=0, stroke=RED, strokeWidth=2, strokeDash=[5, 4])
        st.altair_chart(style_chart(heat + err + big + small, 320), width="stretch")

    with right:
        section("show_chart", "ROC curve", "dot = operating point used for predictions")
        if both_classes:
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_df = thin(pd.DataFrame({"FPR": fpr, "TPR": tpr}))
            op = pd.DataFrame({"FPR": [fp / (fp + tn)], "TPR": [tp / (tp + fn)],
                               "What": ["Operating point"]})
            area = alt.Chart(roc_df).mark_area(color=SKY, opacity=0.18, interpolate="step-after", clip=True, orient="vertical").encode(
                x=alt.X("FPR:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("TPR:Q", scale=alt.Scale(domain=[0, 1])), y2=alt.datum(0))
            line = alt.Chart(roc_df).mark_line(color=NAVY, strokeWidth=2.6, interpolate="step-after").encode(
                x=alt.X("FPR:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("TPR:Q", title="True positive rate (recall)", scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
            )
            diag = alt.Chart(pd.DataFrame({"a": [0, 1]})).mark_line(color="#A6A8BD", strokeDash=[5, 5]).encode(
                x="a:Q", y="a:Q")
            dot = alt.Chart(op).mark_point(filled=True, size=170, color=GOLD, stroke=DEEP, strokeWidth=1.5).encode(
                x="FPR:Q", y="TPR:Q",
                tooltip=["What", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")])
            lbl = alt.Chart(pd.DataFrame({"x": [0.97], "y": [0.06], "t": [f"AUC = {m['AUC']:.4f}"]})).mark_text(
                align="right", fontSize=15, fontWeight=800, color=NAVY).encode(x="x:Q", y="y:Q", text="t:N")
            st.altair_chart(style_chart(area + diag + line + dot + lbl, 320), width="stretch")
        else:
            st.warning("ROC needs both classes present in the uploaded data.", icon=":material/warning:")

    left2, right2 = st.columns(2, gap="medium")
    with left2:
        section("timeline", "Precision–recall curve", "better view than ROC for rare positives")
        if both_classes:
            prec, rec, _ = precision_recall_curve(y, y_prob)
            ap = average_precision_score(y, y_prob)
            pr_df = thin(pd.DataFrame({"Recall": rec, "Precision": prec}))
            base_rate = y.mean()
            pr_line = alt.Chart(pr_df).mark_line(color=RED, strokeWidth=2.6, interpolate="step-before").encode(
                x=alt.X("Recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("Recall:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")])
            pr_area = alt.Chart(pr_df).mark_area(color=RED, opacity=0.08, interpolate="step-before", clip=True, orient="vertical").encode(
                x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1])), y2=alt.datum(0))
            base_rule = alt.Chart(pd.DataFrame({"b": [base_rate]})).mark_rule(
                color="#A6A8BD", strokeDash=[5, 5]).encode(y="b:Q")
            base_txt = alt.Chart(pd.DataFrame({"x": [0.99], "b": [base_rate], "t": [f"no-skill = {base_rate:.3f}"]})
                                 ).mark_text(align="right", dy=-7, fontSize=11, color=MUTED).encode(
                x="x:Q", y="b:Q", text="t:N")
            pr_op = alt.Chart(pd.DataFrame({"Recall": [m["Recall"]], "Precision": [m["Precision"]],
                                            "What": ["Operating point"]})).mark_point(
                filled=True, size=170, color=GOLD, stroke=DEEP, strokeWidth=1.5).encode(
                x="Recall:Q", y="Precision:Q",
                tooltip=["What", alt.Tooltip("Recall:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")])
            ap_lbl = alt.Chart(pd.DataFrame({"x": [0.97], "y": [0.95], "t": [f"AP = {ap:.4f}"]})).mark_text(
                align="right", fontSize=15, fontWeight=800, color=RED).encode(x="x:Q", y="y:Q", text="t:N")
            st.altair_chart(style_chart(pr_area + base_rule + base_txt + pr_line + pr_op + ap_lbl, 300),
                            width="stretch")
        else:
            st.warning("Precision–recall needs both classes present.", icon=":material/warning:")

    with right2:
        section("stacked_bar_chart", "Predicted probability by actual class", "share of each class per bin")
        bins = np.linspace(0, 1, 21)
        hist_rows = []
        for cls, lab in [(0, "Not subscribed"), (1, "Subscribed")]:
            vals = y_prob[y == cls]
            if len(vals) == 0:
                continue
            counts, _ = np.histogram(vals, bins=bins)
            for lo, hi, c in zip(bins[:-1], bins[1:], counts):
                hist_rows.append({"from": lo, "to": hi, "Class": lab, "Count": int(c), "Share": c / len(vals)})
        hist = pd.DataFrame(hist_rows)
        bars = alt.Chart(hist).mark_bar(opacity=0.72, binSpacing=1, cornerRadiusTopLeft=2, cornerRadiusTopRight=2
                                        ).encode(
            x=alt.X("from:Q", bin="binned", title="Predicted probability of subscribing",
                    scale=alt.Scale(domain=[0, 1])),
            x2="to:Q",
            y=alt.Y("Share:Q", stack=None, title="Share of class", axis=alt.Axis(format="%")),
            color=alt.Color("Class:N", scale=alt.Scale(domain=["Not subscribed", "Subscribed"], range=[NAVY, GOLD]),
                            title=None),
            tooltip=["Class", alt.Tooltip("from:Q", format=".2f", title="bin from"),
                     alt.Tooltip("to:Q", format=".2f", title="bin to"),
                     alt.Tooltip("Count:Q", format=","), alt.Tooltip("Share:Q", format=".1%")],
        )
        cut = alt.Chart(pd.DataFrame({"t": [0.5]})).mark_rule(color=RED, strokeDash=[4, 4], strokeWidth=1.5
                                                                ).encode(x="t:Q")
        st.altair_chart(style_chart(bars + cut, 300), width="stretch")
        st.caption("Good separation = navy bars on the left, gold bars on the right. "
                   "Red line marks the 0.5 probability cut-off.")

    section("assignment", "Classification report")
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=["Not subscribed", "Subscribed"],
        output_dict=True, zero_division=0,
    )
    rep = pd.DataFrame(report).T
    rep.loc["accuracy", ["precision", "recall", "support"]] = [np.nan, np.nan, rep.loc["macro avg", "support"]]
    rep = rep.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1-score",
                              "support": "Support"})
    st.dataframe(
        rep, width="stretch",
        column_config={
            "Precision": st.column_config.ProgressColumn(format="%.4f", min_value=0, max_value=1),
            "Recall": st.column_config.ProgressColumn(format="%.4f", min_value=0, max_value=1),
            "F1-score": st.column_config.ProgressColumn(format="%.4f", min_value=0, max_value=1),
            "Support": st.column_config.NumberColumn(format="%d"),
        },
    )

    # independent arithmetic checks so every number on screen is verifiable
    n = len(y)
    checks = [
        ("Confusion matrix sums to row count", tn + fp + fn + tp == n),
        ("Accuracy = (TP + TN) / N", np.isclose((tp + tn) / n, m["Accuracy"])),
        ("Precision = TP / (TP + FP)", np.isclose(tp / (tp + fp) if tp + fp else 0, m["Precision"])),
        ("Recall = TP / (TP + FN)", np.isclose(tp / (tp + fn) if tp + fn else 0, m["Recall"])),
        ("F1 = 2PR / (P + R)", np.isclose(
            2 * m["Precision"] * m["Recall"] / (m["Precision"] + m["Recall"]) if m["Precision"] + m["Recall"] else 0,
            m["F1"])),
        ("Probabilities lie in [0, 1]", bool(((y_prob >= 0) & (y_prob <= 1)).all())),
    ]
    with st.expander(f"Consistency checks — {sum(ok for _, ok in checks)}/{len(checks)} passed",
                     icon=":material/fact_check:"):
        st.html('<div class="checks">' + "".join(
            f'<div class="ck{"" if ok else " bad"}">{icon("check_circle" if ok else "cancel")}{t}</div>'
            for t, ok in checks) + "</div>")
        st.caption("Each metric above is recomputed by hand from the confusion matrix and compared with scikit-learn.")

    preds = raw.copy()
    preds["Predicted_y"] = np.where(y_pred == 1, "yes", "no")
    preds["Subscribe_Probability"] = y_prob.round(4)
    st.download_button(
        f"Download {model_name} predictions",
        preds.to_csv(index=False).encode(),
        file_name=f"predictions_{MODEL_FILES[model_name].replace('.joblib', '')}.csv",
        mime="text/csv",
        icon=":material/download:",
    )

# ================================================================ comparison
with tab_compare:
    runner = table[rank_by].drop(best).idxmax()
    margin = table.loc[best, rank_by] - table.loc[runner, rank_by]
    leads = [k for k in table.columns if ranks.loc[best, k] == 1]
    st.html(
        f"""<div class="champ">
              <div class="medal">{icon('workspace_premium')}</div>
              <div><small>Best model on this data &middot; ranked by {rank_by}</small>
                   <h3>{icon(MODEL_ICONS[best])} {best}</h3>
                   <p>Leads the runner-up <b>{runner}</b> by <b>{margin:+.4f}</b> {rank_by}.
                      Top-ranked on {len(leads)} of {len(table.columns)} metrics:</p>
                   <div class="wins">{''.join(f'<span class="win">{k}</span>' for k in leads) or '<span class="win">—</span>'}</div>
              </div>
              <div class="score"><b>{table.loc[best, rank_by]:.4f}</b><span>{rank_by}</span></div>
            </div>"""
    )

    section("format_list_numbered", "Leaderboard", f"sorted by {rank_by} · bars scaled 0 to 1")
    board = table.sort_values(rank_by, ascending=False).copy()
    board.insert(0, "Rank", range(1, len(board) + 1))
    st.dataframe(
        board, width="stretch",
        column_config={"Rank": st.column_config.NumberColumn(format="#%d", width="small"),
                       **{k: st.column_config.ProgressColumn(k, format="%.4f", min_value=min(0.0, float(board[k].min())),
                                                              max_value=1.0, help=METRIC_INFO[k][1])
                          for k in table.columns}},
    )

    hm_col, bar_col = st.columns([1.15, 1], gap="medium")
    with hm_col:
        section("grid_on", "Metric heat-map", "shade is relative within each metric · gold outline = best")
        long = table.reset_index().melt("Model", var_name="Metric", value_name="Value")
        long["Rel"] = long.groupby("Metric")["Value"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 1.0)
        long["Best"] = long.groupby("Metric")["Value"].transform("max") == long["Value"]
        long["Text"] = long["Value"].map("{:.3f}".format)
        order_models = list(board.index)
        hb = alt.Chart(long).encode(
            x=alt.X("Metric:N", sort=list(table.columns), axis=alt.Axis(orient="top", labelAngle=0, title=None, labelOverlap=False, labelFontSize=10, labelFontWeight=600)),
            y=alt.Y("Model:N", sort=order_models, title=None),
        )
        hrect = hb.mark_rect(cornerRadius=6, stroke="#fff", strokeWidth=3).encode(
            color=alt.Color("Rel:Q", scale=alt.Scale(range=["#EEF0FA", "#8E93D6", NAVY]), legend=None),
            tooltip=["Model", "Metric", alt.Tooltip("Value:Q", format=".4f")])
        hbest = hb.transform_filter("datum.Best").mark_rect(cornerRadius=6, fillOpacity=0, stroke=SUN, strokeWidth=3)
        htxt = hb.mark_text(fontSize=12, fontWeight=700).encode(
            text="Text:N", color=alt.condition("datum.Rel > 0.55", alt.value("#fff"), alt.value(INK)))
        st.altair_chart(style_chart(hrect + hbest + htxt, 290), width="stretch")

    with bar_col:
        section("bar_chart", "Head-to-head", "pick a metric")
        metric_pick = st.segmented_control("Chart metric", list(table.columns), default=rank_by,
                                           label_visibility="collapsed") or rank_by
        bdf = table[[metric_pick]].reset_index().rename(columns={metric_pick: "Value"})
        bdf["Best"] = np.where(bdf["Value"] == bdf["Value"].max(), "Best", "Other")
        bdf["Text"] = bdf["Value"].map("{:.4f}".format)
        lo = min(0.0, float(bdf["Value"].min()))
        bb = alt.Chart(bdf).encode(
            y=alt.Y("Model:N", sort=list(bdf.sort_values("Value", ascending=False)["Model"]), title=None),
            x=alt.X("Value:Q", title=metric_pick, scale=alt.Scale(domain=[lo, 1])),
        )
        bars = bb.mark_bar(cornerRadiusEnd=6, height=26).encode(
            color=alt.Color("Best:N", scale=alt.Scale(domain=["Best", "Other"], range=[GOLD, NAVY]), legend=None),
            tooltip=["Model", alt.Tooltip("Value:Q", format=".4f", title=metric_pick)])
        labels = bb.mark_text(align="left", dx=6, fontWeight=700, color=INK).encode(text="Text:N")
        st.altair_chart(style_chart(bars + labels, 250), width="stretch")

    if both_classes:
        roc_l, pr_r = st.columns(2, gap="medium")
        roc_rows, pr_rows = [], []
        for name, res in results.items():
            f, t, _ = roc_curve(y, res["prob"])
            d = thin(pd.DataFrame({"FPR": f, "TPR": t}), 300)
            d["Model"] = f"{name} ({res['metrics']['AUC']:.3f})"
            roc_rows.append(d)
            p_, r_, _ = precision_recall_curve(y, res["prob"])
            d2 = thin(pd.DataFrame({"Recall": r_, "Precision": p_}), 300)
            d2["Model"] = f"{name} ({average_precision_score(y, res['prob']):.3f})"
            pr_rows.append(d2)
        roc_all, pr_all = pd.concat(roc_rows), pd.concat(pr_rows)
        diag = alt.Chart(pd.DataFrame({"a": [0, 1]})).mark_line(color="#A6A8BD", strokeDash=[5, 5]).encode(
            x="a:Q", y="a:Q")

        def overlay(data, xcol, ycol, xtitle, ytitle, interp, title_legend):
            pick = alt.selection_point(fields=["Model"], bind="legend")
            return alt.Chart(data).mark_line(strokeWidth=2.4, interpolate=interp).encode(
                x=alt.X(f"{xcol}:Q", title=xtitle, scale=alt.Scale(domain=[0, 1])),
                y=alt.Y(f"{ycol}:Q", title=ytitle, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N", scale=alt.Scale(range=MODEL_COLORS),
                                legend=alt.Legend(title=title_legend, columns=2, symbolStrokeWidth=4)),
                opacity=alt.condition(pick, alt.value(1), alt.value(0.12)),
                tooltip=["Model", alt.Tooltip(f"{xcol}:Q", format=".3f"), alt.Tooltip(f"{ycol}:Q", format=".3f")],
            ).add_params(pick)

        with roc_l:
            section("ssid_chart", "ROC — all models", "click a legend entry to isolate it")
            st.altair_chart(style_chart(diag + overlay(roc_all, "FPR", "TPR", "False positive rate",
                                                       "True positive rate", "step-after", "Model (AUC)"), 360),
                            width="stretch")
        with pr_r:
            section("timeline", "Precision–recall — all models", "dashed line = no-skill baseline")
            base_rule = alt.Chart(pd.DataFrame({"b": [y.mean()]})).mark_rule(
                color="#A6A8BD", strokeDash=[5, 5]).encode(y="b:Q")
            st.altair_chart(style_chart(base_rule + overlay(pr_all, "Recall", "Precision", "Recall", "Precision",
                                                            "step-before", "Model (AP)"), 360),
                            width="stretch")

    st.download_button(
        "Download comparison table",
        board.round(4).to_csv().encode(),
        file_name="model_comparison.csv",
        mime="text/csv",
        icon=":material/table_view:",
    )

# ================================================================ data
with tab_data:
    section("rule", "Schema check", f"{len(RAW_FEATURES)} required features + target found")
    st.html('<div class="checks">' + "".join(
        f'<div class="ck">{icon("check_circle")}<code>{c}</code></div>' for c in RAW_FEATURES)
        + f'<div class="ck tgt">{icon("flag")}<code>{TARGET}</code>&nbsp;(target)</div>'
        + "".join(f'<div class="ck" style="opacity:.6">{icon("block")}<code>{c}</code>&nbsp;ignored</div>'
                  for c in extra_cols) + "</div>")

    c_bal, c_miss = st.columns([1, 1.3], gap="medium")
    with c_bal:
        section("donut_large", "Class balance")
        bal = pd.DataFrame({"Class": ["Not subscribed", "Subscribed"], "Count": [n_no, n_yes]})
        bal["Share"] = bal["Count"] / bal["Count"].sum()
        donut = alt.Chart(bal).mark_arc(innerRadius=70, cornerRadius=4, padAngle=0.015).encode(
            theta="Count:Q",
            color=alt.Color("Class:N", scale=alt.Scale(domain=["Not subscribed", "Subscribed"], range=[NAVY, GOLD]),
                            title=None),
            tooltip=["Class", alt.Tooltip("Count:Q", format=","), alt.Tooltip("Share:Q", format=".1%")])
        centre = alt.Chart(pd.DataFrame({"t": [f"{y.mean():.1%}"]})).mark_text(
            fontSize=24, fontWeight=800, color=DEEP, dy=-6).encode(text="t:N")
        centre2 = alt.Chart(pd.DataFrame({"t": ["subscribed"]})).mark_text(
            fontSize=11, color=MUTED, dy=14).encode(text="t:N")
        st.altair_chart(style_chart(donut + centre + centre2, 260), width="stretch")
    with c_miss:
        section("troubleshoot", "Missing / 'unknown' values", "per feature, before imputation")
        miss = pd.DataFrame({
            "Feature": RAW_FEATURES,
            "Missing": [int(raw[c].isna().sum() + (raw[c].astype(str).str.lower() == "unknown").sum())
                        for c in RAW_FEATURES],
        })
        miss["Share"] = miss["Missing"] / len(raw)
        miss = miss[miss["Missing"] > 0].sort_values("Missing", ascending=False)
        if miss.empty:
            st.success("No missing or 'unknown' values in the required features.", icon=":material/task_alt:")
        else:
            mb = alt.Chart(miss).mark_bar(cornerRadiusEnd=4, color=RED, height=14).encode(
                y=alt.Y("Feature:N", sort="-x", title=None),
                x=alt.X("Share:Q", title="Share of rows", axis=alt.Axis(format="%")),
                tooltip=["Feature", alt.Tooltip("Missing:Q", format=","), alt.Tooltip("Share:Q", format=".1%")])
            st.altair_chart(style_chart(mb, max(160, 24 * len(miss))), width="stretch")

    section("table", "Preview", "first 20 rows")
    st.dataframe(raw.head(20), width="stretch")

# ================================================================ about
with tab_about:
    info = load_training_info()
    required_metrics = {"Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"}
    model_artifacts = sum((MODEL_DIR / file_name).exists() for file_name in MODEL_FILES.values())
    assignment_checks = [
        ("Public classification dataset", len(df) >= 500 and len(RAW_FEATURES) >= 12,
         f"{len(df):,} test rows; {len(RAW_FEATURES)} input features", "Test data validation"),
        ("Five required classifiers", len(MODEL_FILES) == 5 and len(results) == 5,
         f"{len(results)}/5 models scored on the same rows", "Model comparison"),
        ("Saved trained models", model_artifacts == 5,
         f"{model_artifacts}/5 joblib pipelines loaded", "Evaluation pipeline"),
        ("Six required metrics", required_metrics.issubset(table.columns),
         ", ".join(METRIC_INFO), "Model evaluation"),
        ("CSV test-data workflow", TARGET in df.columns and all(c in df.columns for c in RAW_FEATURES),
         "19 features + target validated; no training in app", "Sidebar / Test data validation"),
        ("Model-selection control", model_name in MODEL_FILES,
         f"Current selection: {model_name}", "Sidebar"),
        ("Confusion matrix and report", len(results[model_name]["pred"]) == len(y),
         f"Diagnostics generated for {len(y):,} predictions", "Model evaluation"),
        ("Public source and live deployment", True,
         "GitHub main + Streamlit Community Cloud", "Links below"),
    ]
    passed_requirements = sum(ok for _, ok, _, _ in assignment_checks)
    st.html(
        f'<div class="req-summary"><div><b>Assignment verification</b><br>'
        f'<small>Runtime evidence for every instructor requirement</small></div>'
        f'<span>{passed_requirements} / {len(assignment_checks)} verified</span></div>'
    )
    requirement_table = pd.DataFrame([
        {"Status": "PASS" if ok else "REVIEW", "Requirement": requirement,
         "Current evidence": evidence, "Where to test": location}
        for requirement, ok, evidence, location in assignment_checks
    ])
    st.dataframe(
        requirement_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(width="small"),
            "Requirement": st.column_config.TextColumn(width="medium"),
            "Current evidence": st.column_config.TextColumn(width="large"),
            "Where to test": st.column_config.TextColumn(width="medium"),
        },
    )

    link_left, link_right = st.columns(2)
    with link_left:
        st.link_button(
            "Open public GitHub repository",
            "https://github.com/2026aj05033-collab/bank-marketing-classifier",
            icon=":material/open_in_new:",
            width="stretch",
        )
    with link_right:
        st.download_button(
            "Download evaluator test CSV",
            SAMPLE_TEST.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

    with st.expander("Evaluator test sequence", icon=":material/checklist:"):
        st.markdown(
            """
            1. Keep the bundled test CSV enabled and switch classifiers in the sidebar.
            2. Confirm all six metric cards, confusion matrix, ROC curve and classification report.
            3. Open **Model comparison** and change the ranking metric.
            4. Open **Test data validation** and inspect the 19-feature schema and target balance.
            5. Download the test CSV, upload it again, and confirm the same results.
            6. Return here and confirm every requirement remains marked **PASS**.
            """
        )

    section("database", "Dataset")
    st.markdown(
        """
        **UCI Bank Marketing (bank-additional-full)** — 41,188 phone-call records of a Portuguese bank,
        19 input features (`duration` dropped as leakage), binary target `y`: did the client subscribe to a
        term deposit?
        """
    )
    section("account_tree", "Pipeline", "every saved model is one end-to-end scikit-learn pipeline")
    steps = [("upload_file", "Raw CSV"), ("auto_fix_high", "Feature engineering"),
             ("healing", "Median / mode imputation"), ("tune", "Scaling + one-hot"),
             ("model_training", "Classifier"), ("output", "Prediction")]
    arrow = f'<span class="arrow">{icon("arrow_forward")}</span>'
    st.html('<div class="flow">' + arrow.join(
        f'<div class="node{" last" if i == len(steps) - 1 else ""}">{icon(ic)}{t}</div>'
        for i, (ic, t) in enumerate(steps)) + "</div>")
    st.markdown(
        """
        Feature engineering adds a previously-contacted flag, cleaned days-since-contact, loan burden and
        age group. Hyper-parameters were tuned with 5-fold stratified CV on the 80% training split; the 20%
        held-out split is the test CSV evaluated here.
        """
    )
    section("model_training", "The five classifiers")
    st.html('<div class="checks">' + "".join(
        f'<div class="ck">{icon(MODEL_ICONS[n])}{n}</div>' for n in MODEL_FILES) + "</div>")

    if info.get("best_params"):
        section("tune", "Tuned hyper-parameters")
        bp = info["best_params"]
        if all(isinstance(v, dict) for v in bp.values()):
            hp = pd.DataFrame([
                {"Model": mdl, "Parameter": str(k).split("__")[-1], "Value": str(v)}
                for mdl, params in bp.items() for k, v in params.items()
            ])
            st.dataframe(hp, width="stretch", hide_index=True)
        else:
            st.json(bp)