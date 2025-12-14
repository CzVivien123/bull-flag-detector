# src/utils.py
import logging, os, sys
import numpy as np
import torch

def setup_logger(log_path="log/run.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
