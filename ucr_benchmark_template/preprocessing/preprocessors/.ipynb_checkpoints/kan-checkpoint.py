import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import torch

def preprocess(name: str, 
                   raw_splits_dir: Path, 
                   processed_dir: Path) -> None:

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Input paths
    xtr = raw_splits_dir / f"{name}_X_train.npy"
    ytr = raw_splits_dir / f"{name}_y_train.npy"
    xts = raw_splits_dir / f"{name}_X_test.npy"
    yts = raw_splits_dir / f"{name}_y_test.npy"

    # Output paths (no subfolders)
    train_out = processed_dir / f"{name}_train.pt"
    val_out = processed_dir / f"{name}_val.pt"
    test_out = processed_dir / f"{name}_test.pt"

    # Skip if already processed
    if train_out.exists():
        logger.info(f"[KAN] Skipping `{name}` – already processed.")
        return

    # Load splits
    logger.info(f"[KAN] Loading raw data for `{name}`…")
    X_train_full = np.load(xtr)
    y_train_full = np.load(ytr)
    X_test = np.load(xts)
    y_test = np.load(yts)

    # Split train into train and val (stratified if possible)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
    except ValueError as e:
        logger.warning(f"[KAN] Stratified split failed for `{name}`: {e}")
        return

    # Standardize (fit only on training set)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Save all sets
    torch.save({"X": X_train, "y": y_train}, train_out)
    torch.save({"X": X_val, "y": y_val}, val_out)
    torch.save({"X": X_test, "y": y_test}, test_out)

    logger.info(f"[KAN] Saved processed tensors to `{processed_dir}`.")
    