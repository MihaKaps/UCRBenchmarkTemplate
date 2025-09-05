from pathlib import Path
import zipfile
import typer
from loguru import logger
import yaml
import tempfile
import shutil

from ucr_benchmark_template.preprocessing.utils.fix_labels import fix_labels_and_drop_nans
from ucr_benchmark_template.preprocessing.preprocessors import dynamic_preprocess_model
from ucr_benchmark_template.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

RAW_SPLITS = PROCESSED_DATA_DIR / "raw_splits"

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
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_path = Path(tmpdir)
                try:
                    zip_ref.extractall(temp_path)
                except RuntimeError:
                    if not password:
                        logger.error("Zip file is password protected but no password was provided.")
                        raise typer.Exit(code=1)
                    logger.info("Using provided password to unzip.")
                    zip_ref.extractall(temp_path, pwd=password.encode('utf-8'))

                # Flatten structure if there's a single top-level folder
                top_level_items = list(temp_path.iterdir())
                extract_to.mkdir(parents=True, exist_ok=True)

                if len(top_level_items) == 1 and top_level_items[0].is_dir():
                    for item in top_level_items[0].iterdir():
                        shutil.move(str(item), extract_to / item.name)
                else:
                    for item in top_level_items:
                        shutil.move(str(item), extract_to / item.name)

        logger.success(f"Extraction complete to {extract_to}")
    except zipfile.BadZipFile:
        logger.error(f"File is not a zip file or it is corrupted: {zip_path}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error during unzip: {e}")
        raise typer.Exit(code=1)


@app.command()
def preprocess_data(
    input_dir: Path = Path("data/raw"),
    raw_output_dir: Path = Path("data/processed/raw_splits"),
    processed_base_dir: Path = Path("data/processed"),
    params_file: Path = Path("params.yaml"),
):
    # Load parameters
    with open(params_file) as f:
        params = yaml.safe_load(f)

    datasets = params["preprocess"].get("datasets", [])
    models = params["preprocess"].get("models", ["mlp", "kan"])  # default to mlp & kan

    # All datasets in input_dir if none provided
    if not datasets:
        datasets = [d.name for d in input_dir.iterdir() if d.is_dir()]

    typer.echo(f"Datasets to process: {datasets}")
    typer.echo(f"Models to process for: {models}")

    # Step 1: Run fix_labels_and_drop_nans for all datasets
    for dataset in datasets:
        try:
            fix_labels_and_drop_nans(dataset, raw_output_dir)
        except Exception as e:
            typer.echo(f"[!] Skipping {dataset} due to error in label fixing: {e}")

    # Step 2: Model-specific preprocessing
    for model in models:
        for dataset in datasets:
            try:
                model_output_dir = processed_base_dir / model
                model_output_dir.mkdir(parents=True, exist_ok=True)

                output_file = model_output_dir / f"{dataset}.npz"

                if output_file.exists():
                    typer.echo(f"Skipping {dataset} for model {model}: output already exists.")
                    continue

                dynamic_preprocess_model(model, dataset, raw_output_dir, model_output_dir)

            except Exception as e:
                typer.echo(f"[!] Skipping {dataset} for model {model} due to error: {e}")



if __name__ == "__main__":
    app()















        