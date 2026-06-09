"""
app.py — CMAPSS RUL Predictor  |  Streamlit UI
Run:  streamlit run app.py
"""

import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch

from pipeline import (
    PipelineConfig, run_full_pipeline,
    load_artifacts, load_model,
    apply_condition_cluster, apply_condition_scalers,
    CMAPSSDataset, evaluate_model,
)
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
#  Page config & theme
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CMAPSS RUL Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #21262d; }
[data-testid="stHeader"]           { background: transparent; }

/* ── Typography ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; color: #e6edf3; }
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; color: #8b949e; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #e6edf3; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.9rem !important; font-weight: 700 !important; }

/* ── Buttons ── */
.stButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: background 0.15s;
}
.stButton > button:hover { background: #2ea043; }

/* ── Selectbox / sliders ── */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] > div { color: #e6edf3; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { border-bottom: 1px solid #21262d; }
[data-baseweb="tab"] { color: #8b949e; font-weight: 500; }
[aria-selected="true"][data-baseweb="tab"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div { background: #238636; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 6px; }

/* ── Divider ── */
hr { border-color: #21262d; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 1rem;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label { color: #8b949e !important; font-size: 0.8rem !important; }

/* ── Health gauge wrapper ── */
.gauge-wrapper { display: flex; justify-content: center; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"
DATA_PATH_KEY = "data_path"

PALETTE = {
    "blue":   "#58a6ff",
    "green":  "#3fb950",
    "orange": "#d29922",
    "red":    "#f85149",
    "purple": "#bc8cff",
    "bg":     "#0d1117",
    "card":   "#161b22",
    "border": "#21262d",
    "muted":  "#8b949e",
}

FD_DESCRIPTIONS = {
    1: "Single condition · Low fault modes",
    2: "6 conditions · Low fault modes",
    3: "Single condition · High fault modes",
    4: "6 conditions · High fault modes",
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
    """Convert RUL to 0–100 health score."""
    return float(np.clip(rul / threshold * 100, 0, 100))


def health_color(h: float) -> str:
    if h > 65:   return PALETTE["green"]
    if h > 35:   return PALETTE["orange"]
    return PALETTE["red"]


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ RUL Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔧 Train", "📊 Evaluate", "🔮 Inference"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Dataset path**")
    # Auto-detect Kaggle vs deployed environment
    _default_path = (
        "/kaggle/input/nasa-data"
        if os.path.exists("/kaggle/input/nasa-data")
        else st.session_state.get(DATA_PATH_KEY, "")
    )
    data_path = st.text_input(
        "CMAPSS folder (only needed for re-training)",
        value=_default_path,
        label_visibility="collapsed",
    )
    st.session_state[DATA_PATH_KEY] = data_path

    st.markdown("---")
    st.markdown(
        f"<span style='color:{PALETTE['muted']};font-size:0.75rem'>Device: **{DEVICE}**</span>",
        unsafe_allow_html=True,
    )
    trained = [fd for fd in [1, 2, 3, 4] if model_exists(fd)]
    if trained:
        st.markdown(
            f"<span style='color:{PALETTE['green']};font-size:0.75rem'>Trained: {', '.join(f'FD00{fd}' for fd in trained)}</span>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
#  PAGE: Overview
# ─────────────────────────────────────────────

if page == "🏠 Overview":

    st.title("CMAPSS Engine RUL Predictor")
    st.markdown(
        "<span style='color:#8b949e'>Predicting remaining useful life of turbofan jet engines "
        "using a condition-aware MS-TCT architecture.</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Architecture", "MS-TCT")
    with col2: st.metric("Datasets", "FD001 – FD004")
    with col3: st.metric("Window", "50 cycles")
    with col4: st.metric("Loss", "Huber (SmoothL1)")

    st.markdown("---")

    st.markdown("### Architecture")
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("""
**MS-TCT-Condition** combines five components in sequence:

| Block | Role |
|---|---|
| **MultiScale CNN** | Parallel Conv1D (k=3,5,7) captures short, mid, long patterns |
| **SE Block** | Channel-wise attention — re-weights feature maps |
| **Residual TCN** | Dilated temporal convolutions (d=2,4) for longer context |
| **Transformer** | 2-layer encoder with positional encoding |
| **Condition Embed** | KMeans operating-regime embedding concatenated at head |

The RUL target is clipped at 100 (piecewise-linear degradation assumption).
Per-condition StandardScalers normalise each operating regime independently.
        """)

    with c2:
        # Architecture flow diagram
        stages = ["Input (B, 50, F)", "MultiScale CNN", "SE Block", "Residual TCN",
                  "PositionalEncoding", "Transformer ×2", "Attention Pool", "Condition Embed", "RUL output"]
        colors = [PALETTE["muted"]] + [PALETTE["blue"]] * 6 + [PALETTE["purple"], PALETTE["green"]]

        fig = go.Figure()
        for i, (label, color) in enumerate(zip(stages, colors)):
            y = len(stages) - 1 - i
            fig.add_shape(type="rect",
                x0=0.1, x1=0.9, y0=y + 0.1, y1=y + 0.8,
                fillcolor=PALETTE["card"], line=dict(color=color, width=1.5))
            fig.add_annotation(x=0.5, y=y + 0.45, text=label,
                showarrow=False, font=dict(color=color, size=11), xref="x", yref="y")
            if i < len(stages) - 1:
                fig.add_annotation(
                    x=0.5, y=y + 0.05, ay=y - 0.15, ax=0.5,
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=2, arrowcolor=PALETTE["border"], arrowwidth=1.5,
                    showarrow=True, text="")

        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[-0.3, len(stages)]),
            height=420,
            paper_bgcolor=PALETTE["bg"],
            plot_bgcolor=PALETTE["bg"],
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Sub-datasets")
    cols = st.columns(4)
    for i, (fd, desc) in enumerate(FD_DESCRIPTIONS.items()):
        with cols[i]:
            status = "✅ Trained" if model_exists(fd) else "⬜ Untrained"
            color  = PALETTE["green"] if model_exists(fd) else PALETTE["muted"]
            st.markdown(f"""
<div style='background:{PALETTE["card"]};border:1px solid {PALETTE["border"]};
border-radius:8px;padding:1rem;'>
<div style='font-size:1.3rem;font-weight:700;color:{PALETTE["blue"]}'>FD00{fd}</div>
<div style='font-size:0.8rem;color:{PALETTE["muted"]};margin-top:0.3rem'>{desc}</div>
<div style='font-size:0.75rem;color:{color};margin-top:0.6rem'>{status}</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: Train
# ─────────────────────────────────────────────

elif page == "🔧 Train":

    st.title("Train")
    st.markdown("<span style='color:#8b949e'>Configure and run training on one or more sub-datasets.</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("### Config")
        fd_choices = st.multiselect("Sub-datasets", [1, 2, 3, 4],
                                    default=[1], format_func=lambda x: f"FD00{x}")
        epochs      = st.slider("Epochs",       10, 100, 35, 5)
        window_size = st.slider("Window size",  20, 100, 50, 5)
        batch_size  = st.select_slider("Batch size", [64, 128, 256], value=128)
        lr          = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
        n_clusters  = st.slider("Operating clusters (KMeans)", 3, 10, 6)
        rul_clip    = st.slider("RUL clip threshold", 50, 150, 100, 10)

        run_btn = st.button("▶  Start Training", use_container_width=True)

    with c2:
        st.markdown("### Training progress")

        if run_btn:
            if not fd_choices:
                st.warning("Select at least one sub-dataset.")
            elif not os.path.exists(data_path):
                st.error(f"Data path not found: `{data_path}`")
            else:
                cfg = PipelineConfig(
                    window_size=window_size,
                    rul_threshold=rul_clip,
                    n_clusters=n_clusters,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                )

                all_results = {}
                status_box  = st.empty()
                prog_bar    = st.progress(0)
                loss_placeholder = st.empty()

                for fd in fd_choices:
                    status_box.info(f"Training FD00{fd}…")
                    loss_history: list = []

                    chart_data = pd.DataFrame({"epoch": [], "loss": []})
                    loss_fig_placeholder = loss_placeholder.empty()

                    def cb(epoch, total, loss, fd=fd, lh=loss_history):
                        lh.append(loss)
                        prog_bar.progress(epoch / total)
                        df = pd.DataFrame({"Epoch": list(range(1, len(lh)+1)), "Loss": lh})
                        fig = px.line(df, x="Epoch", y="Loss",
                                      title=f"FD00{fd} — Training Loss",
                                      color_discrete_sequence=[PALETTE["blue"]])
                        loss_placeholder.plotly_chart(plotly_defaults(fig), use_container_width=True)

                    result = run_full_pipeline(
                        data_path=data_path,
                        fd=fd,
                        cfg=cfg,
                        device=DEVICE,
                        save_dir=MODEL_DIR,
                        progress_callback=cb,
                    )
                    all_results[fd] = result

                status_box.success("Training complete!")
                prog_bar.progress(1.0)

                st.markdown("### Results")
                res_cols = st.columns(len(fd_choices))
                for i, fd in enumerate(fd_choices):
                    r = all_results[fd]
                    with res_cols[i]:
                        st.metric(f"FD00{fd} RMSE", f"{r['rmse']:.2f}")
                        st.metric(f"FD00{fd} MAPE", f"{r['mape']*100:.1f}%")

                st.session_state["last_train_results"] = all_results
        else:
            st.markdown(
                f"<div style='color:{PALETTE['muted']};padding:2rem;text-align:center;"
                f"border:1px dashed {PALETTE['border']};border-radius:8px'>"
                "Configure settings on the left and press Start Training.</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
#  PAGE: Evaluate
# ─────────────────────────────────────────────

elif page == "📊 Evaluate":

    st.title("Evaluate")
    st.markdown("<span style='color:#8b949e'>Model performance from the last training run.</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    trained_fds = [fd for fd in [1, 2, 3, 4] if model_exists(fd)]
    if not trained_fds:
        st.warning("No trained models found. Go to **Train** first.")
        st.stop()

    fd = st.selectbox("Sub-dataset", trained_fds, format_func=lambda x: f"FD00{x}")
    fd_dir = os.path.join(MODEL_DIR, f"FD00{fd}")

    # ── Load saved metrics (available on Render without raw data) ──────────
    metrics_path = os.path.join(fd_dir, "metrics.json")
    saved_metrics = None
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            saved_metrics = json.load(f)

    # ── Summary metrics ────────────────────────────────────────────────────
    if saved_metrics:
        st.markdown("### Saved training results")
        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"{saved_metrics['rmse']:.2f} cycles")
        c2.metric("MAPE", f"{saved_metrics['mape']*100:.1f}%")

        if saved_metrics.get("loss_curve"):
            lc = saved_metrics["loss_curve"]
            fig = px.line(
                x=list(range(1, len(lc)+1)), y=lc,
                labels={"x": "Epoch", "y": "Loss"},
                title=f"FD00{fd} — Training Loss Curve",
                color_discrete_sequence=[PALETTE["blue"]],
            )
            st.plotly_chart(plotly_defaults(fig), use_container_width=True)
    else:
        st.info("No saved metrics found for this model. Run live evaluation below (requires raw data).")

    st.markdown("---")

    # ── Live re-evaluation (optional, needs raw CMAPSS data) ──────────────
    with st.expander("▸  Live re-evaluation on validation split (needs dataset path)"):
        if st.button("Run live evaluation", use_container_width=False):
            fd_dir_live = os.path.join(MODEL_DIR, f"FD00{fd}")
            try:
                kmeans, scalers, features, cfg = load_artifacts(fd_dir_live)
            except Exception as e:
                st.error(f"Could not load artifacts: {e}")
                st.stop()

            if not os.path.exists(data_path):
                st.error(f"Dataset not found at `{data_path}`. Set the correct path in the sidebar.")
                st.stop()

            from pipeline import load_cmapss, add_train_rul, apply_condition_cluster, apply_condition_scalers, CMAPSSDataset
            from sklearn.model_selection import train_test_split

            with st.spinner("Loading data…"):
                train_df, _, _ = load_cmapss(data_path, fd)
                train_df = add_train_rul(train_df, cfg.rul_threshold)
                train_df = apply_condition_cluster(train_df, kmeans)
                train_df = apply_condition_scalers(train_df, features, scalers)
                all_ids  = train_df['id'].unique()
                _, val_ids = train_test_split(all_ids, test_size=cfg.val_split,
                                              random_state=cfg.random_state)
                val_data    = train_df[train_df['id'].isin(val_ids)]
                val_dataset = CMAPSSDataset(val_data, features, cfg.window_size)
                val_loader  = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

            with st.spinner("Running inference…"):
                model = load_model(os.path.join(fd_dir_live, "model.pt"),
                                   len(features), cfg, DEVICE)
                rmse, mape, preds, truths = evaluate_model(model, val_loader, DEVICE)

            st.metric("Live RMSE", f"{rmse:.2f}")
            st.metric("Live MAPE", f"{mape*100:.1f}%")

            t1, t2, t3 = st.tabs(["Predicted vs Actual", "Residuals", "Error Distribution"])

            with t1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=truths, y=preds, mode='markers',
                    marker=dict(color=PALETTE["blue"], size=3, opacity=0.4), name="Predictions"))
                lim = max(truths.max(), preds.max()) + 5
                fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode='lines',
                    line=dict(color=PALETTE["green"], dash='dash', width=1.5), name="Perfect"))
                fig.update_layout(title=f"FD00{fd} — Predicted vs Actual RUL",
                    xaxis_title="True RUL (cycles)", yaxis_title="Predicted RUL (cycles)")
                st.plotly_chart(plotly_defaults(fig), use_container_width=True)

            with t2:
                residuals = preds - truths
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=truths, y=residuals, mode='markers',
                    marker=dict(color=PALETTE["purple"], size=3, opacity=0.4), name="Residual"))
                fig.add_hline(y=0, line=dict(color=PALETTE["muted"], dash='dash', width=1))
                fig.update_layout(title=f"FD00{fd} — Residuals",
                    xaxis_title="True RUL", yaxis_title="Prediction error")
                st.plotly_chart(plotly_defaults(fig), use_container_width=True)

            with t3:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=residuals, nbinsx=60,
                    marker_color=PALETTE["orange"], opacity=0.8, name="Error"))
                fig.update_layout(title="Prediction error distribution",
                    xaxis_title="Error (pred − true)", yaxis_title="Count")
                st.plotly_chart(plotly_defaults(fig), use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE: Inference
# ─────────────────────────────────────────────

elif page == "🔮 Inference":

    st.title("Inference")
    st.markdown("<span style='color:#8b949e'>Upload sensor readings and get a predicted RUL.</span>",
                unsafe_allow_html=True)
    st.markdown("---")

    trained_fds = [fd for fd in [1, 2, 3, 4] if model_exists(fd)]
    if not trained_fds:
        st.warning("No trained models found. Go to **Train** first.")
        st.stop()

    col_l, col_r = st.columns([1, 2])

    with col_l:
        fd = st.selectbox("Model (sub-dataset)", trained_fds, format_func=lambda x: f"FD00{x}")
        uploaded = st.file_uploader(
            "Upload CSV (sensor time-series for one engine, ≥50 rows)",
            type=["csv", "txt"],
        )
        st.markdown(
            f"<span style='color:{PALETTE['muted']};font-size:0.78rem'>"
            "Expected columns: <code>cycle, op1, op2, op3, s1 … s21</code><br>"
            "No header row — same format as CMAPSS test files."
            "</span>", unsafe_allow_html=True)

        predict_btn = st.button("Predict RUL", use_container_width=True)

    with col_r:
        if predict_btn and uploaded is not None:
            fd_dir = os.path.join(MODEL_DIR, f"FD00{fd}")
            try:
                kmeans, scalers, features, cfg = load_artifacts(fd_dir)
            except Exception as e:
                st.error(f"Could not load artifacts: {e}")
                st.stop()

            from pipeline import COLUMN_NAMES, apply_condition_cluster, apply_condition_scalers

            raw = pd.read_csv(uploaded, sep=r"\s+|,", header=None, engine="python")
            raw = raw.iloc[:, :26].copy()
            if len(raw.columns) == 24:
                raw.columns = ['cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]
                raw.insert(0, 'id', 1)
            elif len(raw.columns) == 26:
                raw.columns = COLUMN_NAMES
            else:
                st.error(f"Unexpected number of columns: {len(raw.columns)}. Expected 24 or 26.")
                st.stop()

            if len(raw) < cfg.window_size:
                st.error(f"Need at least {cfg.window_size} rows; got {len(raw)}.")
                st.stop()

            raw = apply_condition_cluster(raw, kmeans)
            raw = apply_condition_scalers(raw, features, scalers)

            # Predict on last window
            model = load_model(os.path.join(fd_dir, "model.pt"), len(features), cfg, DEVICE)
            model.eval()

            window  = torch.tensor(raw[features].values[-cfg.window_size:],
                                   dtype=torch.float32).unsqueeze(0).to(DEVICE)
            cond_v  = int(raw['condition'].values[-1])
            cond_t  = torch.tensor([cond_v], dtype=torch.long).to(DEVICE)

            with torch.no_grad():
                rul_pred = float(model(window, cond_t).item())

            rul_pred = max(0.0, rul_pred)
            health   = rul_to_health(rul_pred, cfg.rul_threshold)
            hcol     = health_color(health)

            st.markdown("### Result")
            rc1, rc2 = st.columns(2)
            rc1.metric("Predicted RUL", f"{rul_pred:.1f} cycles")
            rc2.metric("Health score", f"{health:.0f} / 100")

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health,
                number={"suffix": "%", "font": {"color": hcol, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": PALETTE["muted"]},
                    "bar":  {"color": hcol},
                    "bgcolor": PALETTE["card"],
                    "bordercolor": PALETTE["border"],
                    "steps": [
                        {"range": [0, 35],  "color": "#2d1414"},
                        {"range": [35, 65], "color": "#2d2414"},
                        {"range": [65, 100],"color": "#14231a"},
                    ],
                    "threshold": {"line": {"color": PALETTE["muted"], "width": 2},
                                  "thickness": 0.8, "value": health},
                },
                title={"text": "Engine Health", "font": {"color": PALETTE["muted"]}},
            ))
            fig.update_layout(
                paper_bgcolor=PALETTE["bg"], font={"color": PALETTE["muted"]},
                height=280, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Sensor time-series for the last window
            st.markdown("### Last 50 cycles — sensor overview")
            sensor_cols = [c for c in features if c.startswith('s')][:6]
            fig2 = make_subplots(rows=2, cols=3,
                subplot_titles=sensor_cols,
                shared_xaxes=False, vertical_spacing=0.15)
            for i, s in enumerate(sensor_cols):
                r, c_idx = divmod(i, 3)
                series = raw[s].values[-cfg.window_size:]
                fig2.add_trace(
                    go.Scatter(y=series, mode='lines',
                               line=dict(color=PALETTE["blue"], width=1.5),
                               showlegend=False),
                    row=r+1, col=c_idx+1)
            fig2.update_layout(height=340, paper_bgcolor=PALETTE["bg"],
                               plot_bgcolor=PALETTE["card"],
                               font=dict(color=PALETTE["muted"]),
                               margin=dict(l=30, r=10, t=40, b=20))
            fig2.update_xaxes(gridcolor=PALETTE["border"])
            fig2.update_yaxes(gridcolor=PALETTE["border"])
            st.plotly_chart(fig2, use_container_width=True)

        elif predict_btn and uploaded is None:
            st.warning("Please upload a CSV file first.")
        else:
            st.markdown(
                f"<div style='color:{PALETTE['muted']};padding:2.5rem;text-align:center;"
                f"border:1px dashed {PALETTE['border']};border-radius:8px;margin-top:1rem'>"
                "Upload a sensor CSV and click Predict RUL.</div>",
                unsafe_allow_html=True,
            )
