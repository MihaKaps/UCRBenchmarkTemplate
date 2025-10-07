import pandas as pd
from loguru import logger
from ucr_benchmark_template.config import RESULTS_DIR
import typer

ALL_RESULTS = RESULTS_DIR / "all_results.csv"

app = typer.Typer()

def avg_over_seed():
    output_file = RESULTS_DIR / "results_avg_over_seeds.csv"
    
    df = pd.read_csv(ALL_RESULTS)

    # Define standard columns
    metric_cols = ["accuracy", "f1", "precision", "recall", "train time"]
    group_cols_base = ['model', 'dataset'] #, 'seed']

    # Determine config columns dynamically
    config_cols = [col for col in df.columns if col not in group_cols_base + metric_cols]

    # Fill NA in config columns for grouping
    df[config_cols] = df[config_cols].fillna("NA")

    # Group by model + dataset + config, average and std over seeds
    group_cols = ['model', 'dataset'] + config_cols
    agg_df = df.groupby(group_cols).agg({col: ['mean', 'std'] for col in metric_cols}).reset_index()

    # Flatten multi-index columns
    agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in agg_df.columns]

    # Restore NA in config columns for readability
    for col in config_cols:
        agg_df[col] = agg_df[col].replace("NA", pd.NA)

    # Save result
    agg_df.to_csv(output_file, index=False)
    logger.info("Saved averaged results across seeds")


def avg_over_datasets():
    input_file = RESULTS_DIR / "results_avg_over_seeds.csv"
    output_file = RESULTS_DIR / "all_results_summary.csv"

    df = pd.read_csv(input_file)

    model_col = 'model'
    dataset_col = 'dataset'

    # Identify metric columns ending with _mean or _std
    metric_cols = [col for col in df.columns if col.endswith('_mean') or col.endswith('_std')]

    # Identify config columns (not model, dataset, or metrics)
    config_cols = [col for col in df.columns if col not in [model_col, dataset_col] + metric_cols]

    # Fill missing config values for grouping
    df[config_cols] = df[config_cols].fillna("NA")

    group_cols = [model_col] + config_cols

    # Group by model config and average all metric columns across datasets
    agg_df = df.groupby(group_cols).agg({col: "mean" for col in metric_cols})

    # Reset index
    agg_df = agg_df.reset_index()

    # Add number of datasets seen per config
    agg_df["num_datasets"] = df.groupby(group_cols)[dataset_col].nunique().values

    # Replace back "NA" with pd.NA
    for col in config_cols:
        agg_df[col] = agg_df[col].replace("NA", pd.NA)

    # Save to CSV
    agg_df.to_csv(output_file, index=False)
    logger.info("Saved averaged results across datasets")

    
def compare_models_globally():
    input_file = RESULTS_DIR / "all_results_summary.csv"
    output_file = RESULTS_DIR / "best_models_summary.txt"

    df = pd.read_csv(input_file)

    metrics = ["accuracy", "f1", "precision", "recall", "train time"]
    output_lines = []

    for metric in metrics:
        col_mean = f"{metric}_mean"
        col_std = f"{metric}_std"

        if col_mean not in df.columns or col_std not in df.columns:
            continue

        # Use max for all except "train time" (where lower is better)
        if metric == "train time":
            best_row = df.loc[df[col_mean].idxmin()]
        else:
            best_row = df.loc[df[col_mean].idxmax()]

        output_lines.append(f"Best model by {metric.upper()}:")
        output_lines.append(f"  Model: {best_row['model']}")
        output_lines.append(f"  Mean: {best_row[col_mean]:.4f} ± {best_row[col_std]:.4f}")

        # Identify hyperparameter config
        exclude_cols = ['model', 'num_datasets'] + [
            f"{m}_{stat}" for m in metrics for stat in ["mean", "std"]
        ]
        config = {
            k: v for k, v in best_row.items()
            if k not in exclude_cols and pd.notna(v) and v != "NA"
        }

        if config:
            output_lines.append("  Config:")
            for k, v in config.items():
                output_lines.append(f"    {k}: {v}")
        else:
            output_lines.append("  Config: [none]")

        output_lines.append("")

    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    logger.info("Best models summary saved")


def best_model_per_dataset():
    input_file = RESULTS_DIR / "results_avg_over_seeds.csv"
    output_file = RESULTS_DIR / "best_model_per_dataset.csv"

    df = pd.read_csv(input_file)

    if 'f1_mean' not in df.columns:
        raise ValueError("Expected column 'f1_mean' not found in file.")

    config_cols = [col for col in df.columns if col not in ['model', 'dataset', 'f1_mean', 'f1_std']]
    df[config_cols] = df[config_cols].fillna('NA')

    # Select best row per dataset
    best_df = (
        df.loc[df.groupby('dataset')['f1_mean'].idxmax()]
        .sort_values('dataset')
        .reset_index(drop=True)
    )

    # Reorder columns: dataset first
    ordered_cols = ['dataset'] + [col for col in best_df.columns if col != 'dataset']
    best_df = best_df[ordered_cols]

    best_df.to_csv(output_file, index=False)
    logger.info("Best model per dataset saved")


def find_best_config_per_model():
    input_file = RESULTS_DIR / "all_results_summary.csv"
    output_file = RESULTS_DIR / "best_config_per_model.csv"

    df = pd.read_csv(input_file)

    # Ensure we're comparing using f1_mean
    best_rows = df.loc[df.groupby("model")["f1_mean"].idxmax()]

    # Optional: sort alphabetically by model name
    best_rows = best_rows.sort_values("model")

    best_rows.to_csv(output_file, index=False)
    logger.info("Best config per model saved")

def generate_readme():
    readme_file = RESULTS_DIR / "README_generated_files.txt"

    lines = [
        "# Generated Results Summary Files\n",
        "**Descriptions of each generated file in the results pipeline:**\n",
        "1. `all_results.csv` – Raw results across all seeds and datasets for all model configurations.",
        "2. `results_avg_over_seeds.csv` – Averaged results across seeds for each dataset-model-config combo. Contains mean and std of each metric.",
        "3. `all_results_summary.csv` – Averaged results across datasets. Each row is a unique model configuration averaged over all datasets.",
        "4. `best_models_summary.txt` – For each metric (accuracy, f1, etc.), shows the best-performing model configuration globally (across all datasets).",
        "5. `best_model_per_dataset.csv` – For each dataset, selects the best model configuration by F1 score. Dataset column is first and ordered alphabetically.",
        "6. `best_config_per_model.csv` – For each model name (e.g., mlp, kan, inceptiontime), selects the best configuration by F1 score from dataset-averaged results.",
        ""
    ]

    with open(readme_file, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"Generated README file saved to: {readme_file}")


@app.command()
def main():
    avg_over_seed()
    avg_over_datasets()
    compare_models_globally()
    best_model_per_dataset()
    find_best_config_per_model()
    generate_readme()

if __name__ == "__main__":
    main()