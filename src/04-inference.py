# src/04-inference.py
import os
import numpy as np
import torch

from src.config import Config
from src.utils import setup_logger, set_seed
from src.pipeline import CNN1D, LABELS

def main():
    cfg = Config()
    logger = setup_logger(cfg.log_path)
    set_seed(cfg.seed)

    data_npz = os.path.join(cfg.out_dir, "processed_flags.npz")
    d = np.load(data_npz)
    X = d["X"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(in_ch=X.shape[1], n_classes=len(LABELS), dropout=cfg.dropout).to(device)

    model_path = os.path.join(cfg.out_dir, "flags_cnn1d_final.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = torch.tensor(X[0:1], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

    logger.info("=== INFERENCE ===")
    logger.info(f"Predicted class id: {pred}")
    logger.info(f"Predicted label: {LABELS[pred]}")

if __name__ == "__main__":
    main()