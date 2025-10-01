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

#=========================
# NoProp Architecture
#=========================

class NoPropBlock(nn.Module):
    def __init__(self, input_length, emb_dimension, num_classes, num_channs = 3):
        super().__init__()
        
        # CNN and FCNN for input
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channs, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, num_channs, input_length, input_length)
            flat_size = self.conv_layers(dummy).shape[1]
 
        self.fc_input = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # FCNN for signal
        self.fc_signal_1 = nn.Sequential(
            nn.Linear(emb_dimension, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.fc_signal_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
        )

        # Merged FC 
        self.fc_merged = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, z):
        fx = self.conv_layers(x)
        fx = self.fc_input(fx)
        
        fz_1 = self.fc_signal_1(z)
        fz_2 = self.fc_signal_2(fz_1) + fz_1
        
        fused = torch.cat([fx, fz_2], dim = -1)
        out = self.fc_merged(fused)
        #out = F.softmax(logits, dim = -1)
        return out

class NoPropModel(nn.Module):
    def __init__(self, num_blocks, input_length, emb_dimension, num_classes, num_channs = 3):
        super().__init__()

        # Blocks
        self.blocks = nn.ModuleList([
            NoPropBlock(input_length, emb_dimension, num_classes, num_channs)
            for _ in range(num_blocks)
        ])
        
        self.num_blocks = num_blocks
        self.input_length = input_length
        self.emb_dimension = emb_dimension

        # Learnable embedding
        self.W_embed = nn.Parameter(
            torch.eye(num_classes, emb_dimension), requires_grad=True
        )


        # Final classifier
        self.out_head = nn.Linear(emb_dimension, num_classes)  
    
    def forward(self, x, z):   
        all_z = []
        for block in self.blocks:
            z = block(x, z)  
            all_z.append(z)
        logits = self.out_head(z)  
        return logits, all_z

#=========================
# //
#=========================


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

    
def make_model(dataset_name, depth, layer_size):
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
    
    return DRTP_Model(architecture).to(device)



def train(model, trainloader, learning_rate, epochs, T, eta): 
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3) 
    alphas, alpha_hats = noise_schedule(T)
    
    start = time.time() 
    
    for epoch in range(epochs): 
        model.train()
        for t in range(1, T+1): 
            running_loss = 0.0 
            running_accuracy = 0.0
            
            step = 0
            with tqdm(trainloader, desc=f"Training Epoch {epoch+1} Block {t}") as pbar: 
                for data, labels in pbar: 
                    data, labels = data.to(device), labels.to(device) 
                    
                    # Collect embeddings
                    #uy = one_hot(labels, model.signal_length) # For one-hot
                    uy = model.W_embed[labels]

                    # Sample z_(t-1), z_T
                    z_tm1 = sample(alpha_hats, t-1, uy)
                    z_T = sample(alpha_hats, T, uy)
                    
                    # Local forward pass
                    z_pred = model.blocks[t-1](data, z_tm1) # Indexed from 0
                    z_pred = F.softmax(z_pred, dim=-1)
                    #u_hat = z_pred @ model.W_embed.weights
                    
                    u_hat = z_pred @ model.W_embed
                    #print(model.W_embed)
                    # Compute local L2 loss
                    diff_snr = snr(t, alpha_hats) - snr(t-1, alpha_hats)
                    local_loss = diff_snr * F.mse_loss(u_hat, uy, reduction="mean") * (T / 2) * eta

                    # Compute global CE loss
                    logits = model.out_head(z_T)
                    global_loss = F.cross_entropy(logits, labels)

                    # Compute KL Divergence
                    alpha_0 = alpha_hats[0]
                    kl_loss = kl_divergence_term(uy, alpha_0)
                    
                    # Total loss
                    loss = local_loss + global_loss + kl_loss
  
                    optimizer.zero_grad() 
                    loss.backward()
                    optimizer.step() 
                    running_loss += local_loss.item()

                    step += 1
                    pbar.set_postfix(loss=running_loss/step, lr=learning_rate)
    
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

def save_results(train_time, results, dataset, depth, layer_size, lr, epochs, batch):
    save_run_results({
        "model": "DRTP",
        "dataset": dataset,
        **results,
        "train time": train_time,
        "depth": depth,
        "layer size": layer_size,
        "learning rate": lr,
        "batch": batch,
        "epochs": epochs,
        "device": device    
    })
        
def save_model(model, dataset, depth, layer_size, lr, epochs, batch):
    drtp_models_dir = MODELS_DIR / "drtp"
    drtp_models_dir.mkdir(parents=True, exist_ok=True)
    
    name = f"drtp_{dataset}_{depth}_{layer_size}_{lr}_{epochs}_{batch}.pt"
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

    depths = params["train_drtp"]["depths"]
    layers = params["train_drtp"]["hidden_layers"]
    learning_rates = params["train_drtp"]["learning_rates"]
    epochs_list = params["train_drtp"]["epochs"]
    batch = params["train_drtp"]["batch"]
    
    for depth in depths:
        for layer_size in layers:
            for lr in learning_rates:
                for epochs in epochs_list:
                    for dataset in datasets:
                        trainloader, valloader, testloader = load_dataset(dataset)
                            
                             # Make model
                        model = make_model(dataset, depth, layer_size)
    
                            # Train
                        model, train_time = train(model, trainloader, valloader, lr, epochs)
    
                            # Predict
                        all_labels, all_preds = predict(model, testloader)
    
                            # Evaluate
                        eval_results = evaluate(all_labels, all_preds)
    
                            # Save model & results
                        save_model(model, dataset, depth, layer_size, lr, epochs, batch)
                        save_results(train_time, eval_results, dataset, depth, layer_size, lr, epochs, batch)
                                

if __name__ == "__main__":
    app()