import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pandas as pd
from loguru import logger

def preprocess(name: str, 
                   raw_splits_dir: Path, 
                   processed_dir: Path) -> None:
    """
    Preprocess data for MLP training:
      - Loads raw train/test npy files for the given dataset.
      - Scales X_train and X_test using train-fit scaler.
      - Saves scaled data to processed/mlp/<dataset>.npz

    Assumes raw splits are saved as:
      raw_splits_dir / "<Dataset>_X_train.npy"
      raw_splits_dir / "<Dataset>_y_train.npy"
      raw_splits_dir / "<Dataset>_X_test.npy"
      raw_splits_dir / "<Dataset>_y_test.npy"
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Input paths
    xtr = raw_splits_dir / f"{name}_X_train.npy"
    ytr = raw_splits_dir / f"{name}_y_train.npy"
    xts = raw_splits_dir / f"{name}_X_test.npy"
    yts = raw_splits_dir / f"{name}_y_test.npy"

    # Skip if already processed
    out = processed_dir / f"{name}.npz"
    if out.exists():
        logger.info(f"[MLP] Skipping `{name}` – already processed at {out}")
        return

    # Load splits
    logger.info(f"[MLP] Loading raw data for `{name}`…")
    X_train = np.load(xtr)
    y_train = np.load(ytr)
    X_test = np.load(xts)
    y_test = np.load(yts)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Save
    logger.info(f"[MLP] Saving processed data to {out}")
    np.savez_compressed(
        out,
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test
    )