"""
app.py — ADA Predictive Aerospace Intelligence | Streamlit UI
Run:  streamlit run app.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch

from pipeline import (
    COLUMN_NAMES,
    load_artifacts, load_model,
    apply_condition_cluster, apply_condition_scalers,
)

# ─────────────────────────────────────────────
#  Page config & Theme
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ADA | RUL Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed", # Hide sidebar entirely
)

# Inject custom CSS for ADA Branding and complete UI overhaul
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;600&display=swap');

/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #080b10; }
[data-testid="stHeader"]           { background: transparent; }

/* ── Typography & ADA Branding ── */
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

/* ── Tabs acting as Navbar ── */
[data-testid="stTabs"] { margin-top: -2rem; }
[data-baseweb="tab-list"] { 
    justify-content: center; 
    border-bottom: 1px solid #21262d; 
    gap: 3rem;
}
[data-baseweb="tab"] { 
    color: #8b949e; 
    font-weight: 600; 
    font-size: 1.1rem;
    padding-bottom: 1rem;
}
[aria-selected="true"][data-baseweb="tab"] { 
    color: #58a6ff !important; 
    border-bottom: 3px solid #58a6ff !important; 
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.9rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #fff !important; font-size: 2.2rem !important; font-family: 'Orbitron', sans-serif; }

/* ── Buttons ── */
.stButton > button {
    background: #58a6ff;
    color: #080b10;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s;
}
.stButton > button:hover { background: #79b8ff; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3); }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #58a6ff;
    border-radius: 12px;
    padding: 1.5rem;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 5rem;
    color: #484f58;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers & Configuration
# ─────────────────────────────────────────────

DEVICE = torch.device("cpu") # Forced to CPU for safe deployment inference
MODEL_DIR = "models"

PALETTE = {
    "blue":   "#58a6ff",
    "green":  "#3fb950",
    "orange": "#d29922",
    "red":    "#f85149",
    "purple": "#bc8cff",
    "bg":     "#080b10",
    "card":   "#161b22",
    "border": "#21262d",
    "muted":  "#8b949e",
}

def plotly_defaults(fig):
    fig.update_layout(
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["card"],
        font=dict(color=PALETTE["muted"], size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    )
    return fig

def model_exists(fd: int) -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, f"FD00{fd}", "model.pt"))

def rul_to_health(rul: float, threshold: int = 100) -> float:
    return float(np.clip(rul / threshold * 100, 0, 100))

def health_color(h: float) -> str:
    if h > 65: return PALETTE["green"]
    if h > 35: return PALETTE["orange"]
    return PALETTE["red"]

def generate_demo_engine_data():
    """Generates a perfectly formatted, synthetic 50-cycle engine signal for testing."""
    cycles = np.arange(1, 51)
    data = {
        'id': [1]*50,
        'cycle': cycles,
        'op1': np.random.normal(0, 0.001, 50),
        'op2': np.random.normal(0, 0.0003, 50),
        'op3': [100.0]*50
    }
    # Create somewhat realistic degrading sensor data
    for i in range(1, 22):
        base_val = np.random.uniform(10, 500)
        degradation = np.linspace(0, np.random.uniform(0.5, 5.0), 50)
        noise = np.random.normal(0, 0.5, 50)
        data[f's{i}'] = base_val + degradation + noise
        
    return pd.DataFrame(data)

# ─────────────────────────────────────────────
#  Main UI Layout
# ─────────────────────────────────────────────

# Hero Header
st.markdown("<h1 class='ada-title'>ADA</h1>", unsafe_allow_html=True)
st.markdown("<p class='ada-subtitle'>Condition-Aware Turbofan RUL Prediction</p>", unsafe_allow_html=True)

# Navigation via Tabs (Acts as Navbar)
tab_home, tab_eval, tab_infer = st.tabs(["🚀 Mission Control (Home)", "📊 Evaluation Intel", "🔮 Diagnostics (Inference)"])

# ─────────────────────────────────────────────
#  PAGE: Home
# ─────────────────────────────────────────────
with tab_home:
    st.image("https://images.unsplash.com/photo-1517976487492-5750f3195933?auto=format&fit=crop&w=2000&q=80", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### Predictive Maintenance, Evolved.
        **ADA** leverages a state-of-the-art **MS-TCT (Multi-Scale CNN + TCN + Transformer)** architecture to predict the Remaining Useful Life (RUL) of turbofan jet engines. 
        
        By analyzing dense time-series sensor data, ADA understands exact operating regimes and degradations across complex mechanical failure modes. Instead of waiting for an engine to break down, ADA provides a real-time health gauge based on historical fault patterns.
        """)
        
    with c2:
        st.info("""
        **System Specs:**
        * **Window Size:** 50 Cycles
        * **Loss Function:** Huber (SmoothL1)
        * **Attention:** 4-Head Transformer
        * **Data Cluster:** KMeans 6-Regime
        """)

    st.markdown("---")
    st.markdown("### 🛰️ Available Core Models")
    cols = st.columns(4)
    descs = {
        1: "Single condition · Low fault modes",
        2: "6 conditions · Low fault modes",
        3: "Single condition · High fault modes",
        4: "6 conditions · High fault modes"
    }
    for i, fd in enumerate([1, 2, 3, 4]):
        with cols[i]:
            status = "🟢 ONLINE" if model_exists(fd) else "🔴 OFFLINE"
            border = PALETTE["green"] if model_exists(fd) else PALETTE["border"]
            st.markdown(f"""
            <div style='background:{PALETTE["card"]}; border:1px solid {border}; border-radius:12px; padding:1.5rem; text-align:center;'>
            <div style='font-family: Orbitron; font-size:1.5rem; color:{PALETTE["blue"]}'>FD00{fd}</div>
            <div style='font-size:0.85rem; color:{PALETTE["muted"]}; margin: 0.5rem 0;'>{descs[fd]}</div>
            <div style='font-size:0.75rem; font-weight:bold; color:{status[:1]};'>{status}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: Evaluation
# ─────────────────────────────────────────────
with tab_eval:
    st.markdown("### 📊 Model Telemetry")
    st.markdown("<span style='color:#8b949e'>View cached training metrics from the latest deployment push.</span>", unsafe_allow_html=True)
    
    trained_fds = [fd for fd in [1, 2, 3, 4] if model_exists(fd)]
    if not trained_fds:
        st.error("No trained models found in the `/models` directory.")
    else:
        selected_eval_fd = st.selectbox("Select Model to Review", trained_fds, format_func=lambda x: f"FD00{x} (CMAPSS)")
        metrics_path = os.path.join(MODEL_DIR, f"FD00{selected_eval_fd}", "metrics.json")
        
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path) as f:
                saved_metrics = json.load(f)
                
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE (Cycles)", f"{saved_metrics['rmse']:.2f}")
            c2.metric("MAPE", f"{saved_metrics['mape']*100:.1f}%")
            c3.metric("Deployment Status", "Active", delta="Ready")
            
            if "loss_curve" in saved_metrics:
                lc = saved_metrics["loss_curve"]
                fig = px.line(
                    x=list(range(1, len(lc)+1)), y=lc,
                    labels={"x": "Epoch", "y": "SmoothL1 Loss"},
                    title=f"FD00{selected_eval_fd} — Training Convergence Curve",
                    color_discrete_sequence=[PALETTE["blue"]],
                )
                st.plotly_chart(plotly_defaults(fig), use_container_width=True)
        else:
            st.warning("Metrics file not found. Ensure `metrics.json` is pushed to your GitHub repo.")

# ─────────────────────────────────────────────
#  PAGE: Inference
# ─────────────────────────────────────────────
with tab_infer:
    if not trained_fds:
        st.warning("No models online. Please train and deploy models first.")
    else:
        col_l, col_r = st.columns([1, 2], gap="large")

        with col_l:
            st.markdown("### Input Data")
            fd = st.selectbox("Engage Model", trained_fds, format_func=lambda x: f"FD00{x} Predictor")
            
            st.markdown("---")
            input_method = st.radio("Select Data Source", ["Use Demo Example", "Upload Custom CSV"])
            
            uploaded = None
            if input_method == "Upload Custom CSV":
                uploaded = st.file_uploader(
                    "Upload Sensor Time-Series",
                    type=["csv", "txt"],
                    help="Must contain 50+ rows. Format: id, cycle, op1-op3, s1-s21."
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("▶ EXECUTE DIAGNOSTICS", use_container_width=True)

        with col_r:
            if predict_btn:
                fd_dir = os.path.join(MODEL_DIR, f"FD00{fd}")
                
                try:
                    kmeans, scalers, features, cfg = load_artifacts(fd_dir)
                except Exception as e:
                    st.error(f"Integrity Error: Could not load artifacts: {e}")
                    st.stop()

                # Process Input Method
                if input_method == "Use Demo Example":
                    raw = generate_demo_engine_data()
                    st.success("Loaded synthetic 50-cycle engine footprint successfully.")
                else:
                    if uploaded is None:
                        st.warning("Awaiting Custom CSV Upload.")
                        st.stop()
                    
                    raw = pd.read_csv(uploaded, sep=r"\s+|,", header=None, engine="python")
                    raw = raw.iloc[:, :26].copy()
                    
                    if len(raw.columns) == 24:
                        raw.columns = ['cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]
                        raw.insert(0, 'id', 1)
                    elif len(raw.columns) == 26:
                        raw.columns = COLUMN_NAMES
                    else:
                        st.error(f"Format mismatch: Expected 24 or 26 columns, found {len(raw.columns)}.")
                        st.stop()

                if len(raw) < cfg.window_size:
                    st.error(f"Insufficient data limit: Require {cfg.window_size} consecutive cycles, got {len(raw)}.")
                    st.stop()

                # Preprocessing
                with st.spinner("Aligning condition clusters & normalizing sensors..."):
                    raw = apply_condition_cluster(raw, kmeans)
                    raw = apply_condition_scalers(raw, features, scalers)

                    model = load_model(os.path.join(fd_dir, "model.pt"), len(features), cfg, DEVICE)
                    model.eval()

                    window = torch.tensor(raw[features].values[-cfg.window_size:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    cond_v = int(raw['condition'].values[-1])
                    cond_t = torch.tensor([cond_v], dtype=torch.long).to(DEVICE)

                # Inference
                with torch.no_grad():
                    rul_pred = float(model(window, cond_t).item())

                rul_pred = max(0.0, rul_pred)
                health = rul_to_health(rul_pred, cfg.rul_threshold)
                hcol = health_color(health)

                st.markdown("### 📡 Diagnostics Output")
                rc1, rc2 = st.columns(2)
                rc1.metric("Predicted RUL", f"{rul_pred:.1f} Cycles")
                rc2.metric("Overall Health", f"{health:.0f}%")

                # Gauge Graphic
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=health,
                    number={"suffix": "%", "font": {"color": hcol, "size": 48, "family": "Orbitron"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": PALETTE["muted"], "tickwidth": 2},
                        "bar":  {"color": hcol, "thickness": 0.8},
                        "bgcolor": PALETTE["card"],
                        "bordercolor": PALETTE["border"],
                        "steps": [
                            {"range": [0, 35],  "color": "rgba(248, 81, 73, 0.1)"},
                            {"range": [35, 65], "color": "rgba(210, 153, 34, 0.1)"},
                            {"range": [65, 100],"color": "rgba(63, 185, 80, 0.1)"},
                        ],
                        "threshold": {"line": {"color": hcol, "width": 4}, "thickness": 1, "value": health},
                    },
                    title={"text": "SYSTEM INTEGRITY", "font": {"color": PALETTE["muted"], "size": 14}},
                ))
                fig.update_layout(
                    paper_bgcolor=PALETTE["bg"], font={"color": PALETTE["muted"]},
                    height=320, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            elif not predict_btn:
                st.markdown(
                    f"<div style='color:{PALETTE['muted']};padding:4rem 2rem;text-align:center;"
                    f"border:1px dashed {PALETTE['border']};border-radius:12px;margin-top:1rem;background:{PALETTE['card']}'>"
                    "<h4>Awaiting Sequence Data</h4><p>Select an input method and click Execute to run the MS-TCT architecture.</p></div>",
                    unsafe_allow_html=True,
                )

# Footer tag
st.markdown("<div class='footer'>ADA Systems | Engineered by Saiamarnath</div>", unsafe_allow_html=True)
