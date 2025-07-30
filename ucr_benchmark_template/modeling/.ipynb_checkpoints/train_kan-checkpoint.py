from efficient_kan import KAN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn

from loguru import logger
from tqdm import tqdm
import typer

from ucr_benchmark_template.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR 
from ucr_benchmark_template.save_results import save_run_results

torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer()

def load_dataset(dataset: str, batch_size: int = 16):
    processed_dir = Path("data/processed/kan")
    
    # Load .pt files
    train = torch.load(processed_dir / f"{dataset}_train.pt", weights_only=False)
    val = torch.load(processed_dir / f"{dataset}_val.pt", weights_only=False)
    test = torch.load(processed_dir / f"{dataset}_test.pt", weights_only=False)

    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]
    X_test, y_test = test["X"], test["y"]

    # Create data loaders
    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    valloader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader

    
def make_model(dataset_name,depth, layer_size, grid_size, spl_order):
    summary_csv=Path("data/external/DataSummary.csv")
    df_meta = pd.read_csv(summary_csv)
    
    # Find matching row by dataset name
    row = df_meta.loc[df_meta['Name'] == dataset_name]
    if row.empty:
        raise ValueError(f"Dataset {dataset_name} not found in summary.")
    
    # Extract input length and number of classes
    input_length = int(row['Length'].values[0])
    num_classes = int(row['Class'].values[0])
    
    # Build architecture list
    architecture = [input_length] + [layer_size] * depth + [num_classes]
    
    #return KAN(architecture, grid_size=grid_size, spline_order=spl_order, random_seed=seed)
    
    return KAN(architecture, grid_size=grid_size, spline_order=spl_order).to(device)

def train(model, trainloader, valloader, learning_rate, epochs, grid_size):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        with tqdm(trainloader, desc=f"Training Epoch {epoch+1}", disable=True) as pbar:
            for data, labels in pbar:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()
                pbar.set_postfix(loss=running_loss/len(pbar), accuracy=running_accuracy/len(pbar), lr=optimizer.param_groups[0]['lr'])

        # VALIDATION
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for data, labels in tqdm(valloader, desc=f"Validation Epoch {epoch+1}", disable=True):
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, labels).item()
                val_accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

    end = time.time()
    
    return model, end - start

def predict(model, testloader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return all_labels, all_preds

def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=1)),
        "recall": float(recall_score(y_true, y_pred, average="macro"))
    }

def save_results(train_time, results, dataset, depth, layer_size, lr, epochs, batch, grid, spline_order):
    save_run_results({
        "model": "KAN",
        "dataset": dataset,
        **results,
        "train time": train_time,
        "depth": depth,
        "layer size": layer_size,
        "grid size": grid,
        "spline order": spline_order,
        "learning rate": lr,
        "batch": batch,
        "epochs": epochs,
        "device": device
        
    })
        
def save_model(model, dataset, depth, layer_size, lr, epochs, batch,  grid):
    kan_models_dir = MODELS_DIR / "kan"
    kan_models_dir.mkdir(parents=True, exist_ok=True)
    
    name = f"kan_{dataset}_{depth}_{layer_size}_{grid}_{lr}_{epochs}_{batch}.pt"
    path = kan_models_dir / name
    
    torch.save(model.state_dict(), path)
        
@app.command()
def main(
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

    depths = params["train_kan"]["depths"]
    layers = params["train_kan"]["hidden_layers"]
    learning_rates = params["train_kan"]["learning_rates"]
    #seeds = params["train"]["seeds"]
    epochs_list = params["train_kan"]["epochs"]
    batch = params["train_kan"]["batch"]
    grids = params["train_kan"]["grid"]
    spline_order = params["train_kan"]["spline_order"]
    
    for depth in depths:
        for layer_size in layers:
            for grid in grids:
                for lr in learning_rates:
                    for epochs in epochs_list:
                        for dataset in datasets:
                            trainloader, valloader, testloader = load_dataset(dataset)
                            
                             # Make model
                            model = make_model(dataset, depth, layer_size, grid, spline_order)
    
                            # Train
                            model, train_time = train(model, trainloader, valloader, lr, epochs, grid)
    
                            # Predict
                            all_labels, all_preds = predict(model, testloader)
    
                            # Evaluate
                            eval_results = evaluate(all_labels, all_preds)
    
                            # Save model & results
                            save_model(model, dataset, depth, layer_size, lr, epochs, batch,  grid)
                            save_results(train_time, eval_results, dataset, depth, layer_size, lr, epochs, batch, grid, spline_order)
                                

if __name__ == "__main__":
    app()