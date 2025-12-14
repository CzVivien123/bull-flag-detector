# src/config.py
from dataclasses import dataclass

@dataclass
class Config:
    dataset_url: str | None = "https://github.com/CzVivien123/bull-flag-detector/releases/download/dataset-v1/dataset.zip"
    dataset_sha256: str | None = None 

    # paths
    data_dir: str = "data" 
    out_dir: str  = "outputs"
    log_path: str = "log/run.log"

    csv_name: str = "EURUSD_15m_007.csv"
    json_name: str = "EURUSD_15m_007_cimkezett.json"

    # model/data
    L: int = 256
    pre_bars: int = 64
    dropout: float = 0.25

    # training
    seed: int = 42
    batch_size: int = 16
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_size: float = 0.2
    patience: int = 10
