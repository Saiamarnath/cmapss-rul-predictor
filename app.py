"""
app.py — ADA Predictive Aerospace Intelligence | Streamlit UI
Run:  streamlit run app.py
"""

import os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch

import pipeline
from pipeline import (
    COLUMN_NAMES, load_model,
    apply_condition_cluster, apply_condition_scalers,
    PipelineConfig
)

# ─────────────────────────────────────────────
#  CRITICAL FIX: Pickle Namespace Resolution
# ─────────────────────────────────────────────
# Kaggle saved the artifacts under the 'main' and '__main__' namespaces. 
# We explicitly inject PipelineConfig into the system memory so the unpickler finds it.
import types
if 'main' not in sys.modules:
    sys.modules['main'] = types.ModuleType('main')
sys.modules['main'].PipelineConfig = PipelineConfig

if '__main__' in sys.modules:
    sys.modules['__main__'].PipelineConfig = PipelineConfig


# ─────────────────────────────────────────────
#  Page config & Theme
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ADA | RUL Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;600&display=swap');

[data-testid="stAppViewContainer"] { background: #080b10; }
[data-testid="stHeader"]           { background: transparent; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e6edf3; }

.ada-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 4.5rem;
    font-weight: 900;
    color: #58a6ff;
    text-align: center;
    margin-bottom: 0;
    letter-spacing: 4px;
    background: -webkit-linear-gradient(#58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.ada-subtitle {
    text-align: center;
    color: #8b949e;
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: -10px;
    margin-bottom: 2rem;
    letter-spacing: 1px;
}
[data-testid="stTabs"] { margin-top: -2rem; }
[data-baseweb="tab-list"] { justify-content: center; border-bottom: 1px solid #21262d; gap: 3rem; }
[data-baseweb="tab"] { color: #8b949e; font-weight: 600; font-size: 1.1rem; padding-bottom: 1rem; }
[aria-selected="true"][data-baseweb="tab"] { color: #58a6ff !important; border-bottom: 3px solid #58a6ff !important; }
[data-testid="metric-container"] { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.9rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #fff !important; font-size: 2.2rem !important; font-family: 'Orbitron', sans-serif; }
.stButton > button { background: #58a6ff; color: #080b10; border: none; border-radius: 8px; padding: 0.6rem 1.5rem; font-weight: 700; font-family: 'Inter', sans-serif; transition: all 0.2s; }
.stButton > button:hover { background: #79b8ff; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3); }
[data-testid="stFileUploader"] { background: #161b22; border: 1px dashed #58a6ff; border-radius: 12px; padding: 1.5rem; }
.footer { text-align: center; margin-top: 5rem; color: #484f58; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers & Artifact Loading
# ─────────────────────────────────────────────

DEVICE = torch.device("cpu") 
MODEL_DIR = "models"
PALETTE = {"blue": "#58a6ff", "green": "#3fb950", "orange": "#d29922", "red": "#f85149", "bg": "#080b10", "card": "#161b22", "border": "#21262d", "muted": "#8b949e"}

def load_artifacts(path: str):
    results = []
    for name in ['kmeans', 'scalers', 'features', 'config']:
        with open(f"{path}/{name}.pkl", 'rb') as f:
            results.append(pickle.load(f))
    return results

def plotly_defaults(fig):
    fig.update_layout(paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["card"], font=dict(color=PALETTE["muted"], size=12), margin=dict(l=40, r=20, t=40, b=40), xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]), yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]))
    return fig

def model_exists(fd: int) -> bool: return os.path.exists(os.path.join(MODEL_DIR, f"FD00{fd}", "model.pt"))
def rul_to_health(rul: float, threshold: int = 100) -> float: return float(np.clip(rul / threshold * 100, 0, 100))
def health_color(h: float) -> str: return PALETTE["green"] if h > 65 else PALETTE["orange"] if h > 35 else PALETTE["red"]

def generate_demo_engine_data():
    cycles = np.arange(1, 51)
    data = {'id': [1]*50, 'cycle': cycles, 'op1': np.random.normal(0, 0.001, 50), 'op2': np.random.normal(0, 0.0003, 50), 'op3': [100.0]*50}
    for i in range(1, 22):
        data[f's{i}'] = np.random.uniform(10, 500) + np.linspace(0, np.random.uniform(0.5, 5.0), 50) + np.random.normal(0, 0.5, 50)
    return pd.DataFrame(data)

# ─────────────────────────────────────────────
#  Main UI Layout
# ─────────────────────────────────────────────

st.markdown("<h1 class='ada-title'>ADA</h1>", unsafe_allow_html=True)
st.markdown("<p class='ada-subtitle'>Condition-Aware Turbofan RUL Prediction</p>", unsafe_allow_html=True)

tab_home, tab_eval, tab_infer = st.tabs(["🚀 Mission Control", "📊 Evaluation Intel", "🔮 Diagnostics"])

with tab_home:
    st.image("https://images.unsplash.com/photo-1517976487492-5750f3195933?auto=format&fit=crop&w=2000&q=80", use_container_width=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Predictive Maintenance, Evolved.\n**ADA** leverages a state-of-the-art **MS-TCT** architecture to predict the Remaining Useful Life (RUL) of turbofan jet engines.")
    with c2:
        st.info("**System Specs:**\n* **Window Size:** 50 Cycles\n* **Loss Function:** Huber\n* **Attention:** 4-Head Transformer\n* **Cluster:** KMeans 6-Regime")

    st.markdown("---")
    cols = st.columns(4)
    descs = {1: "Single condition · Low fault modes", 2: "6 conditions · Low fault modes", 3: "Single condition · High fault modes", 4: "6 conditions · High fault modes"}
    for i, fd in enumerate([1, 2, 3, 4]):
        with cols[i]:
            status = "🟢 ONLINE" if model_exists(fd) else "🔴 OFFLINE"
            border = PALETTE["green"] if model_exists(fd) else PALETTE["border"]
            st.markdown(f"<div style='background:{PALETTE['card']}; border:1px solid {border}; border-radius:12px; padding:1.5rem; text-align:center;'><div style='font-family: Orbitron; font-size:1.5rem; color:{PALETTE['blue']}'>FD00{fd}</div><div style='font-size:0.85rem; color:{PALETTE['muted']}; margin: 0.5rem 0;'>{descs[fd]}</div><div style='font-size:0.75rem; font-weight:bold; color:{status[:1]};'>{status}</div></div>", unsafe_allow_html=True)

with tab_eval:
    trained_fds = [fd for fd in [1, 2, 3, 4] if model_exists(fd)]
    if not trained_fds:
        st.error("No trained models found in the `/models` directory.")
    else:
        selected_eval_fd = st.selectbox("Select Model to Review", trained_fds, format_func=lambda x: f"FD00{x} (CMAPSS)")
        metrics_path = os.path.join(MODEL_DIR, f"FD00{selected_eval_fd}", "metrics.json")
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path) as f: saved_metrics = json.load(f)
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE (Cycles)", f"{saved_metrics['rmse']:.2f}")
            c2.metric("MAPE", f"{saved_metrics['mape']*100:.1f}%")
            c3.metric("Deployment Status", "Active", delta="Ready")
            if "loss_curve" in saved_metrics:
                fig = px.line(x=list(range(1, len(saved_metrics["loss_curve"])+1)), y=saved_metrics["loss_curve"], labels={"x": "Epoch", "y": "Loss"}, title=f"FD00{selected_eval_fd} — Convergence", color_discrete_sequence=[PALETTE["blue"]])
                st.plotly_chart(plotly_defaults(fig), use_container_width=True)

with tab_infer:
    if not trained_fds:
        st.warning("No models online. Please train and deploy models first.")
    else:
        col_l, col_r = st.columns([1, 2], gap="large")
        with col_l:
            fd = st.selectbox("Engage Model", trained_fds, format_func=lambda x: f"FD00{x} Predictor")
            input_method = st.radio("Select Data Source", ["Use Demo Example", "Upload Custom CSV"])
            uploaded = st.file_uploader("Upload Sensor Time-Series", type=["csv", "txt"]) if input_method == "Upload Custom CSV" else None
            predict_btn = st.button("▶ EXECUTE DIAGNOSTICS", use_container_width=True)

        with col_r:
            if predict_btn:
                fd_dir = os.path.join(MODEL_DIR, f"FD00{fd}")
                try: kmeans, scalers, features, cfg = load_artifacts(fd_dir)
                except Exception as e: st.error(f"Integrity Error: {e}"); st.stop()

                if input_method == "Use Demo Example":
                    raw = generate_demo_engine_data()
                else:
                    if uploaded is None: st.warning("Awaiting CSV Upload."); st.stop()
                    raw = pd.read_csv(uploaded, sep=r"\s+|,", header=None, engine="python").iloc[:, :26].copy()
                    if len(raw.columns) == 24:
                        raw.columns = ['cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]
                        raw.insert(0, 'id', 1)
                    elif len(raw.columns) == 26: raw.columns = COLUMN_NAMES
                    else: st.error("Format mismatch."); st.stop()

                if len(raw) < cfg.window_size: st.error(f"Require {cfg.window_size} consecutive cycles."); st.stop()

                with st.spinner("Aligning condition clusters..."):
                    raw = apply_condition_scalers(apply_condition_cluster(raw, kmeans), features, scalers)
                    model = load_model(os.path.join(fd_dir, "model.pt"), len(features), cfg, DEVICE)
                    model.eval()
                    window = torch.tensor(raw[features].values[-cfg.window_size:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    cond_t = torch.tensor([int(raw['condition'].values[-1])], dtype=torch.long).to(DEVICE)

                with torch.no_grad(): rul_pred = max(0.0, float(model(window, cond_t).item()))
                health, hcol = rul_to_health(rul_pred, cfg.rul_threshold), health_color(rul_to_health(rul_pred, cfg.rul_threshold))

                rc1, rc2 = st.columns(2)
                rc1.metric("Predicted RUL", f"{rul_pred:.1f} Cycles")
                rc2.metric("Overall Health", f"{health:.0f}%")

                fig = go.Figure(go.Indicator(mode="gauge+number", value=health, number={"suffix": "%", "font": {"color": hcol, "size": 48}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": hcol}, "bgcolor": PALETTE["card"], "bordercolor": PALETTE["border"]}))
                fig.update_layout(paper_bgcolor=PALETTE["bg"], font={"color": PALETTE["muted"]}, height=320, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='footer'>ADA Systems | Engineered by Sai Amarnath</div>", unsafe_allow_html=True)
