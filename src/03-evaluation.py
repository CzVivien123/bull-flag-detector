# src/03-evaluation.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config
from src.utils import setup_logger, set_seed, ensure_dir
from src.pipeline import FlagsDataset, CNN1D, LABELS, eval_epoch, plot_confusion_matrix

def main():
    cfg = Config()
    logger = setup_logger(cfg.log_path)
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    data_npz = os.path.join(cfg.out_dir, "processed_flags.npz")
    d = np.load(data_npz)
    X, y = d["X"], d["y"]

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=cfg.val_size, stratify=y, random_state=cfg.seed
    )

    val_loader = DataLoader(FlagsDataset(Xva, yva), batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(in_ch=X.shape[1], n_classes=len(LABELS), dropout=cfg.dropout).to(device)

    model_path = os.path.join(cfg.out_dir, "flags_cnn1d_final.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    _, _, y_true, y_pred = eval_epoch(model, val_loader, loss_fn, device)

    cm = confusion_matrix(y_true, y_pred)
    logger.info("=== EVALUATION ===")
    logger.info("Confusion matrix:\n" + str(cm))
    logger.info("\n" + classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    import matplotlib.pyplot as plt
    plot_confusion_matrix(cm, LABELS, normalize=False, title="Confusion Matrix")
    fig_path = os.path.join(cfg.out_dir, "confusion_matrix.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {fig_path}")

if __name__ == "__main__":
    main()