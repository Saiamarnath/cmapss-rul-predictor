# CMAPSS Engine RUL Predictor

Predicts **Remaining Useful Life (RUL)** of turbofan engines from the NASA CMAPSS dataset
using a condition-aware **MS-TCT** (Multi-Scale CNN + TCN + Transformer) model.

## Project structure

```
cmapss_rul/
├── app.py               # Streamlit UI (4 pages)
├── src/
│   └── pipeline.py      # Data loading, preprocessing, model, training, evaluation
├── models/              # Saved artifacts (created at training time)
│   └── FD00{1-4}/
│       ├── model.pt
│       ├── kmeans.pkl
│       ├── scalers.pkl
│       ├── features.pkl
│       └── config.pkl
├── requirements.txt
├── Procfile             # Render / Railway
└── .streamlit/
    └── config.toml
```

## Quick start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Point the app at your CMAPSS data folder
#    (must contain train_FD001.txt … RUL_FD004.txt)
#    Set it in the sidebar when the app opens, or export:
export CMAPSS_DATA=/path/to/nasa-cmapss-data

# 3. Run
streamlit run app.py
```

Open http://localhost:8501

## Pages

| Page | Purpose |
|---|---|
| **Overview** | Architecture diagram, dataset cards, trained model status |
| **Train** | Configure hyperparams, train one or more FD sub-datasets, live loss curve |
| **Evaluate** | Predicted vs actual scatter, residuals, error histogram |
| **Inference** | Upload a single-engine CSV → health gauge + sensor overview |

## Inference CSV format

Same as the CMAPSS test files — whitespace or comma separated, no header.
Columns: `id, cycle, op1, op2, op3, s1 … s21`  (26 columns)
or without the `id` column (25/24 columns — the app will handle it).
The file must have **at least 50 rows** (one window).

## Deploy to Render

1. Push this folder to a GitHub repo.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: _(auto-detected from Procfile)_
4. Add env var `PORT=8501` if not set automatically.

> **Note**: The app expects pre-trained model files in `models/`.
> Either commit them (small models ~10MB each) or add a startup script
> that downloads them from cloud storage (S3, GCS, etc.).

## Deploy to Railway

```bash
railway login
railway init
railway up
```

Railway auto-detects the Procfile.

## Key differences from the original notebook

| Notebook | This pipeline |
|---|---|
| In-place dtype cast warning | Fixed — features cast to float before scaling |
| No test-set evaluation | `add_test_rul()` implemented for held-out eval |
| Scalers/KMeans discarded | Saved to disk with every training run |
| No epoch-level loss logging | `progress_callback` streams loss per epoch |
| Hard-coded Kaggle path | Configurable via UI sidebar |

## Model architecture

```
Input (B, W, F)
  ↓  permute → (B, F, W)
MultiScaleCNN  [Conv1d k=3,5,7 → concat]  →  (B, 96, W)
SEBlock        [Squeeze-and-Excitation]
ResidualTCN    [Dilated conv d=2,4]
Conv1d reduce                               →  (B, 64, W)
  ↓  permute → (B, W, 64)
PositionalEncoding
TransformerEncoder  [2 layers, 4 heads]
AttentionPool       [weighted sum]          →  (B, 64)
Concat ConditionEmbedding(6→16)             →  (B, 80)
Linear                                      →  RUL scalar
```

Loss: **Huber (SmoothL1)** · Optimiser: **Adam** · Grad clip: 1.0
