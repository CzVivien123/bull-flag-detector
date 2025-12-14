# src/pipeline.py
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import os
import zipfile
import tempfile
import shutil
import requests


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dataset_present(
    data_dir: str,
    csv_name: str,
    json_name: str,
    dataset_url: str | None = None,
    expected_sha256: str | None = None,
):
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, csv_name)
    json_path = os.path.join(data_dir, json_name)

    if os.path.exists(csv_path) and os.path.exists(json_path):
        return csv_path, json_path

    if not dataset_url:
        raise FileNotFoundError(
            f"Missing dataset in '{data_dir}'. Expected {csv_name} and {json_name}."
        )

    print(f"[DATA] Downloading dataset.zip from: {dataset_url}")
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "dataset.zip")

        r = requests.get(dataset_url, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        if expected_sha256:
            got = _sha256(zip_path)
            if got.lower() != expected_sha256.lower():
                raise ValueError(f"SHA256 mismatch. expected={expected_sha256}, got={got}")

        print(f"[DATA] Extracting dataset.zip -> {data_dir}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)

    if not (os.path.exists(csv_path) and os.path.exists(json_path)):
        raise FileNotFoundError(
            "Downloaded dataset.zip, but required files are still missing.\n"
            f"Expected at top-level in zip: {csv_name} and {json_name}.\n"
            "Fix the zip structure or adjust csv_name/json_name in config."
        )

    return csv_path, json_path


# -------------------------
# Utils: resample + features
# -------------------------

def resample_1d(x: np.ndarray, L: int) -> np.ndarray:
    """Lineáris interpolációval újramintavételez L pontra."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == L:
        return x
    if len(x) < 2:
        # extrém rövid szegmens: konstansra
        return np.full(L, x[0] if len(x) == 1 else 0.0, dtype=np.float64)
    t_old = np.linspace(0.0, 1.0, len(x))
    t_new = np.linspace(0.0, 1.0, L)
    return np.interp(t_new, t_old, x)

def safe_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = np.mean(x)
    s = np.std(x)
    return (x - m) / (s + eps)

def detrend(y: np.ndarray) -> np.ndarray:
    """OLS lineáris detrend az index függvényében."""
    y = np.asarray(y, dtype=np.float64)
    t = np.arange(len(y), dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-12)  # numerikailag stabilabb
    A = np.vstack([t, np.ones_like(t)]).T
    # y ~ a*t + b
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    trend = A @ coef
    return y - trend

def log_return(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    lr = np.diff(np.log(close + 1e-12))
    # visszaigazítjuk az eredeti hosszra (első elem 0)
    return np.concatenate([[0.0], lr])

def rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for i in range(len(x)):
        a = max(0, i - win + 1)
        out[i] = np.std(x[a:i+1]) if i - a + 1 >= 2 else 0.0
    return out

def make_channels(seg: pd.DataFrame) -> np.ndarray:
    """
    Bemenet: OHLC szegmens (időrendben).
    Kimenet: (C, T) numpy, C=4.
    """
    close = seg["close"].to_numpy(dtype=np.float64)
    high = seg["high"].to_numpy(dtype=np.float64)
    low  = seg["low"].to_numpy(dtype=np.float64)

    # ATR proxy a szegmensre: átlagos gyertya range
    tr = high - low
    atr = float(np.mean(tr)) + 1e-12

    # 1) detrendelt log-close / ATR
    lc = np.log(close + 1e-12)
    ch0 = detrend(lc) / atr

    # 2) log-return
    ch1 = log_return(close)

    # 3) range / ATR
    ch2 = (high - low) / atr

    # 4) rolling vol (std of returns)
    ch3 = rolling_std(ch1, win=min(10, len(ch1)))

    # flag-szintű z-score a csatornákon (stabilabb tanulás)
    ch0 = safe_zscore(ch0)
    ch1 = safe_zscore(ch1)
    ch2 = safe_zscore(ch2)
    ch3 = safe_zscore(ch3)

    X = np.stack([ch0, ch1, ch2, ch3], axis=0)  # (4, T)
    return X

# -------------------------
# Parse annotations
# -------------------------

def load_annotations(json_path: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Kimenet: [(start_ts, end_ts, label_str), ...]
    A te JSON-od structure-je alapján. :contentReference[oaicite:1]{index=1}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data[0]["annotations"][0]["result"]
    out = []
    for r in results:
        v = r["value"]
        start = pd.to_datetime(v["start"])
        end = pd.to_datetime(v["end"])
        label = v["timeserieslabels"][0]
        out.append((start, end, label))
    return out

# -------------------------
# Dataset
# -------------------------

@dataclass
class Config:
    L: int = 256
    pre_bars: int = 64
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    val_size: float = 0.2
    seed: int = 42

LABELS = [
    "Bearish Normal",
    "Bearish Wedge",
    "Bearish Pennant",
    "Bullish Normal",
    "Bullish Wedge",
    "Bullish Pennant",
]
label2id: Dict[str, int] = {name: i for i, name in enumerate(LABELS)}

class FlagsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32) # (N, C, L)
        self.y = torch.tensor(y, dtype=torch.long) # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_xy(csv_path: str, json_path: str, L: int, pre_bars: int) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    ann = load_annotations(json_path)

    X_list, y_list = [], []
    for start, end, lbl in ann:
        idx = df.index[(df["timestamp"] >= start) & (df["timestamp"] <= end)].to_numpy()
        if len(idx) < 5:
            continue

        i0, i1 = int(idx[0]), int(idx[-1])
        j0 = max(0, i0 - pre_bars) # kontextus eleje
        seg = df.iloc[j0:i1+1].copy() # context + flag

        if len(seg) < 8:
            continue

        X = make_channels(seg) # (C, T)
        Xr = np.stack([resample_1d(X[c], L) for c in range(X.shape[0])], axis=0)  # (C, L)

        X_list.append(Xr)
        y_list.append(label2id[lbl])

    X_all = np.stack(X_list, axis=0)
    y_all = np.array(y_list, dtype=np.int64)
    return X_all, y_all

# -------------------------
# Model
# -------------------------

class CNN1D(nn.Module):
    def __init__(self, in_ch: int = 4, n_classes: int = 6, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        h = self.features(x).squeeze(-1)
        return self.head(h)


# -------------------------
# Train / Eval
# -------------------------

def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()  # skálázás
    return torch.tensor(w, dtype=torch.float32)

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        loss_sum += float(loss.item()) * len(yb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += len(yb)
    return loss_sum / total, correct / total

def plot_confusion_matrix(cm, labels, normalize=False, title="Confusion Matrix"):
    cm = np.array(cm, dtype=np.float64)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    # értékek kiírása a cellákba
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, txt, ha="center", va="center")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title + (" (normalized)" if normalize else ""))
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_p, all_y = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)

        loss_sum += float(loss.item()) * len(yb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += len(yb)

        all_p.append(pred.cpu().numpy())
        all_y.append(yb.cpu().numpy())

    all_p = np.concatenate(all_p)
    all_y = np.concatenate(all_y)
    return loss_sum / total, correct / total, all_y, all_p

