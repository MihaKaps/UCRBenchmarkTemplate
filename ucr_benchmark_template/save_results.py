import pandas as pd
from pathlib import Path
from typing import Dict, Any

from ucr_benchmark_template.config import RESULTS_DIR


def save_run_results(results_dict: Dict[str, Any]):
    """
    Save model training results to a central CSV.
    If new columns appear in results_dict, they are added to the CSV,
    and missing values are filled with NaN for other rows.
    """
    results_path = RESULTS_DIR / "all_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert current result to DataFrame
    new_row = pd.DataFrame([results_dict])

    if results_path.exists():
        # Load existing results
        existing_df = pd.read_csv(results_path)

        # Combine columns (union)
        combined_df = pd.concat([existing_df, new_row], ignore_index=True)

        # Ensure all columns are present and ordered
        all_columns = list(set(existing_df.columns) | set(new_row.columns))
        combined_df = combined_df.reindex(columns=all_columns)

        # Save updated file
        combined_df.to_csv(results_path, index=False)
    else:
        # First-time save
        new_row.to_csv(results_path, index=False)