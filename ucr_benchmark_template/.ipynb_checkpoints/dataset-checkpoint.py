from pathlib import Path
import zipfile
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import yaml

from loguru import logger
from tqdm import tqdm
import typer

from ucr_benchmark_template.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

@app.command()
def unzip_data(
    zip_path: Path = Path("data/external/UCRArchive_2018.zip"),
    extract_to: Path = RAW_DATA_DIR,
    password: str = "someone" # password to UCRArchive, generally known and unnecessary step
):
    """
    Unzip the UCR archive data into the raw data directory.
    
    Args:
    -----
    zip_path: Path to the zip archive.
    extract_to: Directory where files will be extracted.
    """
    if extract_to.exists() and any(extract_to.iterdir()):
        logger.info(f"Skipping unzip: {extract_to} already contains files.")
        return
        
    logger.info(f"Unzipping {zip_path} to {extract_to}...")

    if not zip_path.exists():
        logger.error(f"Zip file does not exist: {zip_path}")
        raise typer.Exit(code=1)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            try:
                zip_ref.extractall(extract_to)
            except RuntimeError:
                if not password:
                    logger.error("Zip file is password protected but no password was provided.")
                    raise typer.Exit(code=1)
                logger.info("Using provided password to unzip.")
                zip_ref.extractall(extract_to, pwd=password.encode('utf-8'))

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
    params_file: Path = Path("params.yaml")
):
        
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load params
    with open(params_file) as f:
        params = yaml.safe_load(f)

    datasets = params["preprocess"].get("datasets", [])

    # If none specified, use all in input_dir
    if not datasets:
        datasets = [d.name for d in input_dir.iterdir() if d.is_dir()]

    selected = []

    for dataset in datasets:
        output_file = output_dir / f"{dataset}.npz"
        if output_file.exists():
            logger.info(f"Skipping {dataset}: already processed.")
        else:
            selected.append(dataset)

    if not selected:
        typer.echo("All datasets already processed. Nothing to do.")
        return

    typer.echo(f"Datasets to process: {selected}")

    for name in selected:
        output_file = output_dir / f"{name}.npz"
        if output_file.exists():
            logger.info(f"Skipping {name}: already processed.")
            continue
            
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
