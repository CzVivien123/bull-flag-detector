# src/02-training.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.config import Config
from src.utils import setup_logger, set_seed, ensure_dir, count_parameters
from src.pipeline import FlagsDataset, CNN1D, LABELS, class_weights, train_one_epoch, eval_epoch

def main():
    cfg = Config()
    logger = setup_logger(cfg.log_path)
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    logger.info("=== CONFIGURATION ===")
    logger.info(vars(cfg))

    data_npz = os.path.join(cfg.out_dir, "processed_flags.npz")
    if not os.path.exists(data_npz):
        logger.info("Processed data not found. Run src/01-data-preprocessing.py first.")
        raise FileNotFoundError(data_npz)

    d = np.load(data_npz)
    X, y = d["X"], d["y"]
    logger.info(f"Loaded processed data: X={X.shape}, y={y.shape}")

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=cfg.val_size, stratify=y, random_state=cfg.seed
    )

    train_loader = DataLoader(FlagsDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(FlagsDataset(Xva, yva), batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(in_ch=X.shape[1], n_classes=len(LABELS), dropout=cfg.dropout).to(device)

    trainable, total = count_parameters(model)
    logger.info("=== MODEL ARCHITECTURE ===")
    logger.info(model)
    logger.info(f"Trainable params: {trainable} | Total params: {total}")

    w = class_weights(ytr, n_classes=len(LABELS)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va_loss, va_acc, _, _ = eval_epoch(model, val_loader, loss_fn, device)

        logger.info(
            f"Epoch {ep:03d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}"
        )

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                logger.info("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # FINAL EVAL
    _, _, y_true, y_pred = eval_epoch(model, val_loader, loss_fn, device)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    logger.info("=== FINAL EVALUATION ===")
    logger.info(f"Accuracy: {acc:.3f}")
    logger.info(f"Macro F1: {f1m:.3f}")
    logger.info("Confusion matrix:\n" + str(cm))
    logger.info("\n" + classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    model_path = os.path.join(cfg.out_dir, "flags_cnn1d_final.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model: {model_path}")

    report_path = os.path.join(cfg.out_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))
    logger.info(f"Saved report: {report_path}")

if __name__ == "__main__":
    main()