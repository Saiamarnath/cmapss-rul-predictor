"""
pipeline.py
End-to-end pipeline for CMAPSS RUL prediction using MS-TCT-Condition model.
"""

import os
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

@dataclass
class PipelineConfig:
    window_size: int = 50
    rul_threshold: int = 100
    n_clusters: int = 6
    batch_size: int = 128
    epochs: int = 35
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    random_state: int = 42
    # transformer
    d_model: int = 64
    nhead: int = 4
    num_tf_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.2
    # condition embedding
    cond_emb_dim: int = 16


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────

COLUMN_NAMES = (
    ['id', 'cycle']
    + [f'op{i}' for i in range(1, 4)]
    + [f's{i}' for i in range(1, 22)]
)


def load_cmapss(data_path: str, fd: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one CMAPSS sub-dataset (FD001–FD004)."""
    train = pd.read_csv(f"{data_path}/train_FD00{fd}.txt", sep=r"\s+", header=None)
    test  = pd.read_csv(f"{data_path}/test_FD00{fd}.txt",  sep=r"\s+", header=None)
    rul   = pd.read_csv(f"{data_path}/RUL_FD00{fd}.txt",   header=None, names=["RUL"])

    train = train.iloc[:, :26].copy()
    test  = test.iloc[:, :26].copy()

    train.columns = COLUMN_NAMES
    test.columns  = COLUMN_NAMES

    return train, test, rul


# ─────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────

def add_train_rul(df: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
    """Compute and clip RUL for training data."""
    max_cycle = df.groupby('id')['cycle'].max().rename('max_cycle')
    df = df.merge(max_cycle, on='id')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=threshold)
    df.drop('max_cycle', axis=1, inplace=True)
    return df


def fit_condition_cluster(df: pd.DataFrame, n_clusters: int = 6) -> KMeans:
    """Fit a KMeans model on operating parameters."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df[['op1', 'op2', 'op3']])
    return kmeans


def apply_condition_cluster(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    df = df.copy()
    df['condition'] = kmeans.predict(df[['op1', 'op2', 'op3']])
    return df


def fit_condition_scalers(
    df: pd.DataFrame,
    features: List[str],
    n_clusters: int = 6
) -> Dict[int, StandardScaler]:
    """Fit per-condition StandardScalers."""
    scalers: Dict[int, StandardScaler] = {}
    for cond in range(n_clusters):
        idx = df['condition'] == cond
        if idx.sum() == 0:
            scalers[cond] = StandardScaler()
            continue
        sc = StandardScaler()
        sc.fit(df.loc[idx, features].astype(float))
        scalers[cond] = sc
    return scalers


def apply_condition_scalers(
    df: pd.DataFrame,
    features: List[str],
    scalers: Dict[int, StandardScaler]
) -> pd.DataFrame:
    df = df.copy()
    # Cast feature columns to float to avoid pandas dtype warning
    df[features] = df[features].astype(float)
    for cond, sc in scalers.items():
        idx = df['condition'] == cond
        if idx.sum() == 0:
            continue
        df.loc[idx, features] = sc.transform(df.loc[idx, features])
    return df


def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame, threshold: int = 100) -> pd.DataFrame:
    """
    For test data the RUL ground truth is in rul_df (one value per engine = RUL at last cycle).
    We attach it to the last cycle row of each engine.
    """
    last_cycles = test_df.groupby('id')['cycle'].max().reset_index()
    last_cycles = last_cycles.rename(columns={'cycle': 'last_cycle'})
    rul_df = rul_df.copy()
    rul_df['id'] = range(1, len(rul_df) + 1)
    merged = last_cycles.merge(rul_df, on='id')
    test_df = test_df.merge(merged[['id', 'last_cycle', 'RUL']], on='id', how='left')
    # Only keep the last cycle row for each engine (that's where ground-truth RUL applies)
    # but we'll return the full df with RUL filled for last rows, NaN elsewhere
    test_df['RUL'] = test_df.apply(
        lambda r: min(r['RUL'], threshold) if r['cycle'] == r['last_cycle'] else np.nan,
        axis=1
    )
    test_df.drop('last_cycle', axis=1, inplace=True)
    return test_df


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class CMAPSSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: List[str], window_size: int = 50):
        self.x: List = []
        self.y: List = []
        self.cond: List = []

        for engine_id in df['id'].unique():
            engine = df[df['id'] == engine_id]
            data   = engine[features].values
            labels = engine['RUL'].values
            conds  = engine['condition'].values

            for i in range(len(data) - window_size):
                self.x.append(data[i:i + window_size])
                self.y.append(labels[i + window_size])
                self.cond.append(conds[i + window_size])

        self.x    = torch.tensor(np.array(self.x),    dtype=torch.float32)
        self.y    = torch.tensor(np.array(self.y),    dtype=torch.float32)
        self.cond = torch.tensor(np.array(self.cond), dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.cond[idx], self.y[idx]


# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position  = torch.arange(0, max_len).unsqueeze(1).float()
        div_term  = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiScaleCNN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.conv3 = nn.Conv1d(input_size, 32, 3, padding=1)
        self.conv5 = nn.Conv1d(input_size, 32, 5, padding=2)
        self.conv7 = nn.Conv1d(input_size, 32, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            torch.relu(self.conv3(x)),
            torch.relu(self.conv5(x)),
            torch.relu(self.conv7(x)),
        ], dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 8)
        self.fc2 = nn.Linear(channels // 8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        y = x.mean(dim=2)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(2)


class ResidualTCN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=4, dilation=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x + residual


class MS_TCT_Condition(nn.Module):
    """Multi-Scale CNN + TCN + Transformer with operating-condition conditioning."""

    def __init__(self, input_size: int, cfg: PipelineConfig):
        super().__init__()
        self.mscnn   = MultiScaleCNN(input_size)
        self.se      = SEBlock(96)
        self.tcn     = ResidualTCN(96)
        self.reduce  = nn.Conv1d(96, cfg.d_model, 1)

        self.pos_enc = PositionalEncoding(cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            batch_first=True,
            dropout=cfg.dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_tf_layers)

        self.cond_emb  = nn.Embedding(cfg.n_clusters, cfg.cond_emb_dim)
        self.attn_pool = nn.Linear(cfg.d_model, 1)
        self.fc        = nn.Linear(cfg.d_model + cfg.cond_emb_dim, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)          # (B, F, T)
        x = self.mscnn(x)               # (B, 96, T)
        x = self.se(x)
        x = self.tcn(x)
        x = self.reduce(x)              # (B, 64, T)

        x = x.permute(0, 2, 1)         # (B, T, 64)
        x = self.pos_enc(x)
        x = self.transformer(x)

        attn = torch.softmax(self.attn_pool(x), dim=1)
        x    = torch.sum(attn * x, dim=1)   # (B, 64)

        cond_emb = self.cond_emb(cond)      # (B, 16)
        x = torch.cat([x, cond_emb], dim=1)
        return self.fc(x).squeeze(-1)


# ─────────────────────────────────────────────
#  Training & evaluation
# ─────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: PipelineConfig,
    device: torch.device,
    progress_callback=None,
) -> Tuple[nn.Module, List[float]]:
    """Train and return (model, epoch_losses)."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.SmoothL1Loss()
    epoch_losses: List[float] = []

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, cond, y in train_loader:
            x, cond, y = x.to(device), cond.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, cond), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        if progress_callback:
            progress_callback(epoch + 1, cfg.epochs, avg_loss)

    return model, epoch_losses


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return (RMSE, MAPE, predictions, ground_truths)."""
    model.eval()
    preds, truths = [], []

    with torch.no_grad():
        for x, cond, y in loader:
            x, cond = x.to(device), cond.to(device)
            pred = model(x, cond).cpu().numpy()
            preds.extend(pred.tolist())
            truths.extend(y.numpy().tolist())

    preds  = np.array(preds)
    truths = np.array(truths)
    rmse   = float(np.sqrt(mean_squared_error(truths, preds)))
    mape   = float(np.mean(np.abs((truths - preds) / np.maximum(truths, 1.0))))

    return rmse, mape, preds, truths


# ─────────────────────────────────────────────
#  Artifact save / load
# ─────────────────────────────────────────────

def save_artifacts(path: str, kmeans: KMeans, scalers: Dict, features: List[str], cfg: PipelineConfig):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(f"{path}/scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    with open(f"{path}/features.pkl", "wb") as f:
        pickle.dump(features, f)
    with open(f"{path}/config.pkl", "wb") as f:
        pickle.dump(cfg, f)


def load_artifacts(path: str):
    with open(f"{path}/kmeans.pkl",  "rb") as f: kmeans   = pickle.load(f)
    with open(f"{path}/scalers.pkl", "rb") as f: scalers  = pickle.load(f)
    with open(f"{path}/features.pkl","rb") as f: features = pickle.load(f)
    with open(f"{path}/config.pkl",  "rb") as f: cfg      = pickle.load(f)
    return kmeans, scalers, features, cfg


def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str, input_size: int, cfg: PipelineConfig, device: torch.device) -> nn.Module:
    model = MS_TCT_Condition(input_size, cfg)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
#  High-level runner
# ─────────────────────────────────────────────

def run_full_pipeline(
    data_path: str,
    fd: int,
    cfg: PipelineConfig,
    device: torch.device,
    save_dir: Optional[str] = None,
    progress_callback=None,
) -> dict:
    """
    Full train + val pipeline for one FD sub-dataset.
    Returns a result dict with metrics, loss curve, predictions.
    """
    # --- Load ---
    train_df, test_df, rul_df = load_cmapss(data_path, fd)

    # --- Preprocess ---
    train_df = add_train_rul(train_df, cfg.rul_threshold)
    kmeans   = fit_condition_cluster(train_df, cfg.n_clusters)
    train_df = apply_condition_cluster(train_df, kmeans)

    features = sorted(train_df.columns.difference(['id', 'cycle', 'RUL', 'condition']).tolist())

    scalers  = fit_condition_scalers(train_df, features, cfg.n_clusters)
    train_df = apply_condition_scalers(train_df, features, scalers)

    # --- Split ---
    all_ids = train_df['id'].unique()
    train_ids, val_ids = train_test_split(all_ids, test_size=cfg.val_split, random_state=cfg.random_state)

    train_data = train_df[train_df['id'].isin(train_ids)]
    val_data   = train_df[train_df['id'].isin(val_ids)]

    train_dataset = CMAPSSDataset(train_data, features, cfg.window_size)
    val_dataset   = CMAPSSDataset(val_data,   features, cfg.window_size)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # --- Train ---
    model = MS_TCT_Condition(len(features), cfg)
    model, loss_curve = train_model(model, train_loader, cfg, device, progress_callback)

    # --- Evaluate ---
    rmse, mape, preds, truths = evaluate_model(model, val_loader, device)

    # --- Save ---
    if save_dir:
        fd_dir = os.path.join(save_dir, f"FD00{fd}")
        save_artifacts(fd_dir, kmeans, scalers, features, cfg)
        save_model(model, os.path.join(fd_dir, "model.pt"))

    return {
        "fd":         fd,
        "rmse":       rmse,
        "mape":       mape,
        "loss_curve": loss_curve,
        "preds":      preds,
        "truths":     truths,
        "features":   features,
        "n_features": len(features),
    }
