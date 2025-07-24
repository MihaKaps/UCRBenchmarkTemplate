import pandas as pd
from pathlib import Path
from typing import Dict, Any

from ucr_benchmark_template.config import RESULTS_DIR

def save_run_results(results_dict: Dict[str, Any]):
    """
    Save model training results to a central CSV.
    Ensures consistent columns, fills missing values with NaN.
    """
    
    expected_columns = [
        "model", "dataset", "accuracy", "f1", "precision", "recall", "train time", "depth", "layer size", "grid size", "spline order", "learning rate", "batch", "epochs", "seed", "convolutions", "filters", "kernel size"
    ]

    row_data = {col: results_dict.get(col, pd.NA) for col in expected_columns}

    # Convert to DataFrame row
    df_row = pd.DataFrame([row_data])

    # Define path
    results_path = RESULTS_DIR / "all_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists, append; else, create
    if results_path.exists():
        df_row.to_csv(results_path, mode="a", index=False, header=False)
    else:
        df_row.to_csv(results_path, index=False)
