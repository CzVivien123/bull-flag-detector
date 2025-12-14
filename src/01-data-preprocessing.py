# src/01-data-preprocessing.py
import os
import numpy as np
from src.config import Config
from src.utils import setup_logger, set_seed, ensure_dir
from src.pipeline import ensure_dataset_present, build_xy, LABELS

def main():
    cfg = Config()
    logger = setup_logger(cfg.log_path)
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    csv_path  = os.path.join(cfg.data_dir, cfg.csv_name)
    json_path = os.path.join(cfg.data_dir, cfg.json_name)

    logger.info("=== DATA PREPROCESSING ===")
    logger.info(f"csv_path={csv_path}")
    logger.info(f"json_path={json_path}")
    logger.info(f"L={cfg.L}, pre_bars={cfg.pre_bars}")

    csv_path, json_path = ensure_dataset_present(
    data_dir=cfg.data_dir,
    csv_name=cfg.csv_name,
    json_name=cfg.json_name,
    dataset_url=cfg.dataset_url,
    expected_sha256=cfg.dataset_sha256,
    )   

    logger.info(f"Using CSV:  {csv_path}")
    logger.info(f"Using JSON: {json_path}")

    X, y = build_xy(csv_path, json_path, L=cfg.L, pre_bars=cfg.pre_bars)
    logger.info(f"Built dataset: X={X.shape}, y={y.shape}")

    # class counts
    counts = np.bincount(y, minlength=len(LABELS))
    logger.info(f"Class counts: {counts.tolist()}")

    out_path = os.path.join(cfg.out_dir, "processed_flags.npz")
    np.savez_compressed(out_path, X=X, y=y)
    logger.info(f"Saved: {out_path}")

if __name__ == "__main__":
    main()