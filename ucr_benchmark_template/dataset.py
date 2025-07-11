from pathlib import Path
import zipfile
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

from loguru import logger
from tqdm import tqdm
import typer

from ucr_benchmark_template.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def unzip_data(
    zip_path: Path = Path("data/external/UCRArchive_2018.zip"),
    extract_to: Path = RAW_DATA_DIR,
):
    """
    Unzip the UCR archive data into the raw data directory.
    
    Args:
    -----
    zip_path: Path to the zip archive.
    extract_to: Directory where files will be extracted.
    """
    logger.info(f"Unzipping {zip_path} to {extract_to}...")

    if not zip_path.exists():
        logger.error(f"Zip file does not exist: {zip_path}")
        raise typer.Exit(code=1)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.success(f"Extraction complete to {extract_to}")
    except zipfile.BadZipFile:
        logger.error(f"File is not a zip file or it is corrupted: {zip_path}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error during unzip: {e}")
        raise typer.Exit(code=1)

def preprocess(df_train, df_test):
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    if df_train.empty or df_test.empty:
        return None

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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train.iloc[:, 1:])
    X_test = scaler.transform(df_test.iloc[:, 1:])

    return X_train, X_test, y_train, y_test
    
@app.command()
def preprocess_data(
    input_dir: Path = Path("data/raw/UCRArchive_2018"),
    output_dir: Path = Path("data/processed"),
    datasets: list[str] = typer.Option(None, help="List of datasets to process (default: all)"),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = [d.name for d in input_dir.iterdir() if d.is_dir()]

    if datasets:
        selected = [d for d in datasets if d in all_datasets]
        if not selected:
            typer.echo("No matching datasets found for given names.")
            raise typer.Exit(1)
    else:
        selected = all_datasets

    typer.echo(f"Datasets to process: {selected}")

    for name in selected:
        train_path = input_dir / name / f"{name}_TRAIN.tsv"
        test_path = input_dir / name / f"{name}_TEST.tsv"

        if not (train_path.exists() and test_path.exists()):
            logger.warning(f"Skipping {name}, missing train/test files")
            continue

        df_train = pd.read_csv(train_path, sep='\t', header=None)
        df_test = pd.read_csv(test_path, sep='\t', header=None)

        processed = preprocess(df_train, df_test)
        if processed is None:
            logger.warning(f"Skipping {name}, empty or invalid data after preprocessing")
            continue

        X_train, X_test, y_train, y_test = processed

        np.savez_compressed(
            output_dir / f"{name}.npz",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train.values,
            y_test=y_test.values
        )

        typer.echo(f"Processed dataset {name}")
        
if __name__ == "__main__":
    app()
