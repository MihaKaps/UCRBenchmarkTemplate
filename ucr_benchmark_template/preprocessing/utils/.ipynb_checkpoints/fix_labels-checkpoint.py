import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

def fix_labels_and_drop_nans(dataset_name: str, output_dir: Path):
    
    train_path = Path(f"data/raw/{dataset_name}/{dataset_name}_TRAIN.tsv")
    test_path = Path(f"data/raw/{dataset_name}/{dataset_name}_TEST.tsv")

    if not (train_path.exists() and test_path.exists()):
        logger.warning(f"Skipping {name}, missing train/test files")
        return
    
    # Check if processed files already exist
    files_to_check = [
        output_dir / f"{dataset_name}_X_train.npy",
        output_dir / f"{dataset_name}_y_train.npy",
        output_dir / f"{dataset_name}_X_test.npy",
        output_dir / f"{dataset_name}_y_test.npy"
    ]
    if all(f.exists() for f in files_to_check):
        logger.info(f"Skipping {dataset_name}, processed files already exist in {output_dir}")
        return
        
    df_train = pd.read_csv(train_path, sep='\t', header=None)
    df_test = pd.read_csv(test_path, sep='\t', header=None)

    # Drop NaNs
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    if df_train.empty or df_test.empty:
        raise ValueError(f"After dropping NaNs, {dataset_name} train or test is empty.")

    # Fix labels
    y_train = df_train.iloc[:, 0]
    y_test = df_test.iloc[:, 0]

    unique = y_test.unique()

    if -1 in unique:
        y_train = y_train.replace({-1: 0})
        y_test = y_test.replace({-1: 0})
    elif 0 not in unique:
        min_label = min(unique)
        y_train = y_train - min_label
        y_test = y_test - min_label

    # Update dataframe with new labels
    # df_train.iloc[:, 0] = y_train
    # df_test.iloc[:, 0] = y_test

    # Separate features and labels
    X_train = df_train.iloc[:, 1:].to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy(dtype=np.int64)
    X_test = df_test.iloc[:, 1:].to_numpy(dtype=np.float32)
    y_test = y_test.to_numpy(dtype=np.int64)

    # Create output folder if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as separate .npy files
    np.save(output_dir / f"{dataset_name}_X_train.npy", X_train)
    np.save(output_dir / f"{dataset_name}_y_train.npy", y_train)
    np.save(output_dir / f"{dataset_name}_X_test.npy", X_test)
    np.save(output_dir / f"{dataset_name}_y_test.npy", y_test)

    print(f"[✓] Saved processed raw splits to {output_dir}")