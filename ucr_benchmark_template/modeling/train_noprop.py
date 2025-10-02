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

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoPropBlock(nn.Module):
    def __init__(self, input_length, emb_dimension, num_classes):
        super().__init__()
        
        # CNN and FCNN for input
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
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
        return out

class NoPropModel(nn.Module):
    def __init__(self, num_blocks, input_length, emb_dimension, num_classes, learn_emb):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.input_length = input_length

        # Learnable embedding
        if learn_emb:
            self.W_embed = torch.nn.Parameter(torch.randn(num_classes, emb_dimension), requires_grad=True)
            self.emb_dimension = emb_dimension
        else:
            self.W_embed = nn.Parameter(torch.eye(num_classes, num_classes), requires_grad=False)
            self.emb_dimension = num_classes

        # Blocks
        self.blocks = nn.ModuleList([
            NoPropBlock(input_length, self.emb_dimension, num_classes)
            for _ in range(num_blocks)
        ])
            
        # Final classifier
        self.out_head = nn.Linear(self.emb_dimension, num_classes) 

        self.alphas, self.alpha_hats = noise_schedule(num_blocks)
        self.at = torch.sqrt(self.alpha_hats[1:]) * (1 - self.alphas[:-1]) / (1 - self.alpha_hats[:-1])
        self.bt = torch.sqrt(self.alphas[:-1]) * (1 - self.alpha_hats[1:]) / (1 - self.alpha_hats[:-1])
        self.ct = (1 - self.alpha_hats[1:]) * (1 - self.alphas[:-1]) / (1 - self.alpha_hats[:-1])
    
    def forward(self, x, z):   
        # Pass through all T blocks
        for i in range(self.num_blocks):
            block = self.blocks[i]
            logits = block(x, z) @ self.W_embed
            noise = torch.randn_like(z)
            z = self.at[i] * logits + self.bt[i] * z + torch.sqrt(self.ct[i]) * noise
        
        # Final prediction
        logits = self.out_head(z)

        return logits

#=========================
# Functions
#=========================

def noise_schedule(T, s=0.008):
    steps = torch.arange(0, T+1, dtype=torch.float32)
    alpha_hats = torch.cos((((steps / T) + s) / (1 + s)) * torch.pi / 2) ** 2
    alpha_hats = alpha_hats / alpha_hats[0]
    
    alphas = torch.ones_like(alpha_hats)
    alphas[1:] = alpha_hats[1:] / alpha_hats[:-1]
    
    return torch.flip(alphas, dims = [0]), torch.flip(alpha_hats, dims = [0])

def snr(t, alpha_hats):
    return alpha_hats[t] / (1 - alpha_hats[t] + 1e-8)

def sample(alpha_hats, t, uy):
    mean = (alpha_hats[t]**0.5) * uy 
    std = (1 - alpha_hats[t])**0.5 
    noise = torch.randn_like(uy) 
    z_t = mean + std * noise 
    return z_t

def kl_divergence_term(uy, alpha_0):
    mu_q = torch.sqrt(alpha_0) * uy
    sigma2_q = 1.0 - alpha_0

    mu_norm2 = mu_q.pow(2).sum(dim=1)

    d = uy.shape[1]
    kl = 0.5 * (mu_norm2 + d * (sigma2_q - 1 - torch.log(sigma2_q)))
    return kl.mean()

def ensure_channels_first(X):
    # if 2D: [B, L] -> [B, 1, L]
    if X.dim() == 2:
        X = X.unsqueeze(1)
    # if 3D: [B, H, W] -> [B, 1, H, W]
    elif X.dim() == 3:
        X = X.unsqueeze(1)
    return X

#=========================
# Data
#=========================

def load_dataset(dataset: str, batch_size: int = 16):
    processed_dir = Path("data/processed/kan")
   
    # Load .pt files
    train = torch.load(processed_dir / f"{dataset}_train.pt", weights_only=False)
    val = torch.load(processed_dir / f"{dataset}_val.pt", weights_only=False)
    test = torch.load(processed_dir / f"{dataset}_test.pt", weights_only=False)

    X_train, y_train = ensure_channels_first(train["X"]), train["y"]
    X_val, y_val = ensure_channels_first(val["X"]), val["y"]
    X_test, y_test = ensure_channels_first(test["X"]), test["y"]

    # Create data loaders
    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    valloader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader

#=========================
# Model
#=========================

def make_model(dataset_name, emb_dimension, num_blocks, learnable_emb = False):
    summary_csv=Path("data/external/DataSummary.csv")
    df_meta = pd.read_csv(summary_csv)
    
    # Find matching row by dataset name
    row = df_meta.loc[df_meta['Name'] == dataset_name]
    if row.empty:
        raise ValueError(f"Dataset {dataset_name} not found in summary.")
    
    # Extract input length and number of classes
    input_length = int(row['Length'].values[0])
    num_classes = int(row['Class'].values[0])
    
    return NoPropModel(num_blocks, input_length, emb_dimension, num_classes, learnable_emb).to(device)

#=========================
# Train & Eval
#=========================

def train(model, trainloader, valloader, epochs, T, eta, lr, wd): 
    
    alphas, alpha_hats = noise_schedule(T)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) 
    start = time.time() 
    
    for epoch in range(epochs): 
        
        # TRAINING
        model.train()
        for t in range(1, T+1): 
            running_loss = 0.0 
            running_accuracy = 0.0
            
            step = 0
            with tqdm(trainloader, desc=f"Training Epoch {epoch+1} Block {t}") as pbar: 
                for data, labels in pbar: 
                    data, labels = data.to(device), labels.to(device) 
                    
                    # Collect embeddings
                    uy = model.W_embed[labels]

                    # Sample z_(t-1), z_T
                    z_tm1 = sample(alpha_hats, t-1, uy)
                    z_T = sample(alpha_hats, T, uy)
                    
                    # Local forward pass
                    z_pred = model.blocks[t-1](data, z_tm1) # Indexed from 0 therefore t-1
                    z_pred = F.softmax(z_pred, dim=-1)
                    u_hat = z_pred @ model.W_embed
                    
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
                    pbar.set_postfix(loss=running_loss/step, lr=lr)

        # VALIDATION
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for data, labels in tqdm(valloader, desc=f"Validation Epoch {epoch+1}", disable=True):
                data, labels = data.to(device), labels.to(device)
                
                # Start z0 ~ N(0, 1)
                z = torch.randn(data.size(0), model.emb_dimension, device=device)
                
                logits = model(data, z)
                val_loss += F.cross_entropy(logits, labels)
                val_accuracy += (logits.argmax(dim=1) == labels).float().mean().item()
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
            
            # Start z0 ~ N(0, 1)
            z = torch.randn(data.size(0), model.emb_dimension, device=device)

            logits = model(data, z)
            preds = logits.argmax(dim=1)
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

def save_results(train_time, results, dataset, t, learnable_emb, emb_dimension, eta, lr, wd, epochs, batch):
    save_run_results({
        "model": "NoProp",
        "dataset": dataset,
        **results,
        "train time": train_time,
        "T": t,
        "learnable embedding": learnable_emb,
        "embedding dimension": emb_dimension,
        "eta": eta,
        "learning rate": lr,
        "weight decay": wd,
        "batch": batch,
        "epochs": epochs,
        "device": device    
    })
        
def save_model(model, dataset, t, learnable_emb, emb_dimension, eta, lr, wd, epochs, batch):
    noprop_models_dir = MODELS_DIR / "noprop"
    noprop_models_dir.mkdir(parents=True, exist_ok=True)
    
    name = f"noprop_{dataset}_{t}_{learnable_emb}_{emb_dimension}_{eta}_{lr}_{wd}_{epochs}_{batch}.pt"
    path = noprop_models_dir / name
    
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

    ts = params["train_noprop"]["ts"]
    learnable_embs = params["train_noprop"]["learnable_embs"]
    emb_dimensions = params["train_noprop"]["emb_dimensions"]
    etas = params["train_noprop"]["etas"]
    learning_rates = params["train_noprop"]["learning_rates"]
    weight_decays = params["train_noprop"]["weight_decays"]
    epoch_list = params["train_noprop"]["epochs"]
    batch = params["train_noprop"]["batch"]
    
    for t in ts:
        for l_emb in learnable_embs:
            for emb_d in emb_dimensions:
                for eta in etas:
                    for lr in learning_rates:
                        for wd in weight_decays:
                            for epochs in epoch_list:
                                for dataset in datasets:
                                    trainloader, valloader, testloader = load_dataset(dataset)
                                        
                                         # Make model
                                    model = make_model(dataset, emb_d, t, l_emb)
                
                                        # Train
                                    model, train_time = train(model, trainloader, valloader, epochs, t, eta, lr, wd)
                                    
                                        # Predict
                                    all_labels, all_preds = predict(model, testloader)
                
                                        # Evaluate
                                    eval_results = evaluate(all_labels, all_preds)
                
                                        # Save model & results
                                    save_model(model, dataset, t, l_emb, emb_d, eta, lr, wd, epochs, batch)
                                    save_results(train_time, eval_results, dataset, t, l_emb, emb_d, eta, lr, wd, epochs, batch)
                                            

if __name__ == "__main__":
    app()