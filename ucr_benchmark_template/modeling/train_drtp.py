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
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
import typer

from ucr_benchmark_template.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR 
from ucr_benchmark_template.save_results import save_run_results

torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = typer.Typer()

class DRTP_Model(nn.Module):
    def __init__(self, input_length, num_classes, channels, kernel_size, dropout, fc_layers):
        super().__init__()

        #CNN
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ))
            
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Flatten())  # flatten only once, at the end
        self.conv_layers = nn.Sequential(*layers)

        for p in self.conv_layers.parameters():
            p.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, channels[0], input_length)
            flat_size = self.conv_layers(dummy).shape[1]

        fc_layers = [flat_size] + fc_layers
        self.num_layers = len(fc_layers) - 1
        
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_layers) - 1):
            self.fc_layers.append(nn.Linear(fc_layers[i], fc_layers[i+1]))
        
        
        #Hidden layers have fixed random connectivity matrices
        self.random_matrices = []
        for i in range(self.num_layers - 1): #last layer doesn't need matrix
            m = torch.randn(fc_layers[i+1], fc_layers[-1])
            self.register_buffer(f'random_matrix_{i}', m)
            self.random_matrices.append(getattr(self, f'random_matrix_{i}'))

    def forward(self, x):
        self.activations = []
        x = self.conv_layers(x)
        for i in range(self.num_layers - 1):
            z = self.fc_layers[i](x)
            y = torch.tanh(z)
            self.activations.append((z, y))
            x = y
        
        z_out = self.fc_layers[-1](x)
        y_out = torch.softmax(z_out, dim = 1)
        self.activations.append((z_out, y_out))
        return y_out

    def update_weights(self, x, y, lr=0.00001):
        y_out = self.forward(x)
        y_star = F.one_hot(y, num_classes=y_out.size(1)).float()
        
        #Update hidden weights (DRTP)
        for i in range(self.num_layers - 1):
            z, _ = self.activations[i]
            
            if i == 0:
                prev_y = x
            else:
                prev_y = self.activations[i-1][1]
            
            m = self.random_matrices[i]  #[hidden_size, num_classes]
            
            delta_yk = torch.matmul(y_star, m.T)
            grad_z = (1 - torch.tanh(z)**2) #f'(z) = tanh(z)' = 1 - tanh(z)**2 
            grad = (delta_yk * grad_z).T @ prev_y
            
            self.fc_layers[i].weight.data += lr * grad
            self.fc_layers[i].bias.data += lr * (delta_yk * grad_z).sum(0)

        #Update last layer
        z_out, y_out = self.activations[-1]
        error = (y_star - y_out)
        last_hidden = self.activations[-2][1]
        
        self.fc_layers[-1].weight.data += (lr / y_out.size(1)) * error.T @ last_hidden
        self.fc_layers[-1].bias.data += (lr / y_out.size(1)) * error.sum(0)



def load_dataset(dataset, batch_size = 16):
    path = PROCESSED_DATA_DIR / f"mlp/{dataset}.npz"
    data = np.load(path)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(data["X_train"], dtype=torch.float32).unsqueeze(1) 
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32).unsqueeze(1) 
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    # Create DataLoaders
    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return trainloader, testloader

    
def make_model(dataset_name, channels, kernel_size, dropout, fc_layers):
    
    summary_csv=Path("data/external/DataSummary.csv")
    df_meta = pd.read_csv(summary_csv)
    
    # Find matching row by dataset name
    row = df_meta.loc[df_meta['Name'] == dataset_name]
    if row.empty:
        raise ValueError(f"Dataset {dataset_name} not found in summary.")
    
    # Extract input length and number of classes
    input_length = int(row['Length'].values[0])
    num_classes = int(row['Class'].values[0])

    if channels[0] != 1:
        chalnnels = [1] + channels

    if fc_layers[-1] != num_classes:
        fc_layers = fc_layers + [num_classes]
    
    return DRTP_Model(input_length, num_classes, channels, kernel_size, dropout, fc_layers).to(device)

def train(model, trainloader, learning_rate, epochs):
    
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        with tqdm(trainloader, desc=f"Training Epoch {epoch+1}", disable=False) as pbar:
            for data, labels in pbar:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                model.update_weights(data, labels, lr=learning_rate)
                running_loss += loss.item()
                running_accuracy += (outputs.argmax(dim=1) == labels).float().mean().item()
                pbar.set_postfix(loss=running_loss/len(pbar), accuracy=running_accuracy/len(pbar), lr=learning_rate)

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

def save_results(train_time, results, dataset, channels, kernel_size, dropout, fc_layers, lr, epochs, batch):
    save_run_results({
        "model": "DRTP",
        "dataset": dataset,
        **results,
        "train time": train_time,
        "channels": channels,
        "kernel size": kernel_size,
        "dropout": dropout,
        "fc layers": fc_layers,
        "learning rate": lr,
        "batch": batch,
        "epochs": epochs,
        "device": device    
    })
        
def save_model(model, dataset, channels, kernel_size, dropout, fc_layers, lr, epochs, batch):
    drtp_models_dir = MODELS_DIR / "drtp"
    drtp_models_dir.mkdir(parents=True, exist_ok=True)
    
    name = f"drtp_{dataset}_{channels}_{kernel_size}_{dropout}_{fc_layers}_{lr}_{epochs}_{batch}.pt"
    path = drtp_models_dir / name
    
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

    channels = params["train_drtp"]["channels"]
    fc_layers = params["train_drtp"]["fc_layers"]
    kernel_sizes = params["train_drtp"]["kernel_sizes"]
    dropouts = params["train_drtp"]["dropouts"]
    learning_rates = params["train_drtp"]["learning_rates"]
    epochs_list = params["train_drtp"]["epochs"]
    batch = params["train_drtp"]["batch"]
    
    for channels in channels:
        for fc_layers in fc_layers:
            for kernel_size in kernel_sizes:
                for dropout in dropouts:
                    for lr in learning_rates:
                        for epochs in epochs_list:
                            for dataset in datasets:
                                trainloader, testloader = load_dataset(dataset)
                                    
                                     # Make model
                                model = make_model(dataset, channels, kernel_size, dropout, fc_layers)
            
                                    # Train
                                model, train_time = train(model, trainloader, lr, epochs)
            
                                    # Predict
                                all_labels, all_preds = predict(model, testloader)
            
                                    # Evaluate
                                eval_results = evaluate(all_labels, all_preds)
            
                                    # Save model & results
                                save_model(model, dataset, channels, kernel_size, dropout, fc_layers, lr, epochs, batch)
                                save_results(train_time, eval_results, dataset, channels, kernel_size, dropout, fc_layers, lr, epochs, batch)
                                

if __name__ == "__main__":
    app()