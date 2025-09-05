from pathlib import Path
from sklearn.neural_network import MLPClassifier
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle

from loguru import logger
from tqdm import tqdm
import typer

from ucr_benchmark_template.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR 
from ucr_benchmark_template.save_results import save_run_results

app = typer.Typer()

def load_dataset(dataset: str):
    path = PROCESSED_DATA_DIR / f"mlp/{dataset}.npz"
    data = np.load(path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

def make_model(depth, layer_size, lr, batch, epochs, seed):
    return MLPClassifier(
        hidden_layer_sizes=tuple([layer_size] * depth),
        activation="relu",
        solver="adam",
        alpha=0.1,
        batch_size=batch,
        learning_rate="constant",
        learning_rate_init=lr,
        max_iter=epochs,
        random_state=seed
    )

def train(model, X_train, y_train):
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    return model, end - start

def predict(model, X_test):
    return model.predict(X_test)

def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=1)),
        "recall": float(recall_score(y_true, y_pred, average="macro"))
    }

def save_model(model, dataset, depth, layer_size, lr, epochs, batch, seed):
    mlp_models_dir = MODELS_DIR / "mlp"
    mlp_models_dir.mkdir(parents=True, exist_ok=True)
    
    name = f"mlp_{dataset}_{depth}_{layer_size}_{lr}_{epochs}_{batch}_{seed}.pkl"
    with open(mlp_models_dir / name, "wb") as f:
        pickle.dump(model, f)

def save_results(results, dataset, depth, layer_size, lr, epochs, batch, seed, train_time):
    save_run_results({
        "model": "MLP",
        "dataset": dataset,
        **results,
        "train time": train_time,
        "depth": depth,
        "layer size": layer_size,
        "learning rate": lr,
        "batch": batch,
        "epochs": epochs,
        "seed": seed
        
    })

# -------------------------
# Main pipeline
# -------------------------
@app.command()
def main(
    #datasets: list[str] = typer.Option(None, help="List of datasets to run on (defaults to all)")
    params_file: Path = Path("params.yaml")
):
    # Load params
    with open(params_file) as f:
        params = yaml.safe_load(f)

    datasets = params["preprocess"].get("datasets", [])

    if not datasets:  # handles None, empty list, empty string
        datasets = [f.stem for f in (PROCESSED_DATA_DIR / "mlp").glob("*.npz")]
        if not datasets:
            raise ValueError(f"No datasets found in {PROCESSED_DATA_DIR}")

    depths = params["train_mlp"]["depths"]
    layers = params["train_mlp"]["hidden_layers"]
    learning_rates = params["train_mlp"]["learning_rates"]
    seeds = params["train_mlp"]["seeds"]
    epochs_list = params["train_mlp"]["epochs"]
    batch = params["train_mlp"]["batch"]
    
    for depth in depths:
        for layer_size in layers:
            for lr in learning_rates:
                for epochs in epochs_list:
                    for dataset in datasets:
                        X_train, X_test, y_train, y_test = load_dataset(dataset)
                        all_results = []  # collect seed results for this dataset + hyperparams
                        
                        for seed in seeds:
                             # Make model
                            model = make_model(depth, layer_size, lr, batch, epochs, seed)
    
                            # Train
                            model, train_time = train(model, X_train, y_train)
    
                            # Predict
                            y_pred = predict(model, X_test)
    
                            # Evaluate
                            eval_results = evaluate(y_test, y_pred)
    
                            # Save model & results
                            save_model(model, dataset, depth, layer_size, lr, epochs, batch, seed)
                            save_results(eval_results, dataset, depth, layer_size, lr, epochs, batch, seed, train_time)

if __name__ == "__main__":
    app()
