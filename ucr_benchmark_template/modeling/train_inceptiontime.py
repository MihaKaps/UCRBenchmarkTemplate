import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pandas as pd
from loguru import logger
import torch
import torch.utils.data as data
import torchtime
#import torchtime.transforms as transforms
import torchtime.models as models
import torch.nn as nn
import torch.optim as optim
import typer
import yaml
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from ucr_benchmark_template.config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR 
from ucr_benchmark_template.save_results import save_run_results

app = typer.Typer()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

criterion = nn.CrossEntropyLoss()

def load_dataset(dataset, batch_size):
    path = PROCESSED_DATA_DIR / f"mlp/{dataset}.npz"
    data = np.load(path)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.tensor(data["X_train"], dtype=torch.float32).unsqueeze(1) 
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32).unsqueeze(1) 
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    # Create DataLoaders
    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return trainloader, testloader
    
def make_model(dataset_name, depth, n_convolutions, n_filters, kernel_size):
    summary_csv=Path("data/external/DataSummary.csv")
    df_meta = pd.read_csv(summary_csv)
    
    # Find matching row by dataset name
    row = df_meta.loc[df_meta['Name'] == dataset_name]
    if row.empty:
        raise ValueError(f"Dataset {dataset_name} not found in summary.")
    
    # Extract input length and number of classes
    input_length = int(row['Length'].values[0])
    num_classes = int(row['Class'].values[0])
    
    net = models.InceptionTime(
        n_inputs=1, 
        n_classes=num_classes,
        use_residual=True, 
        use_bottleneck=True, 
        depth=depth, 
        n_convolutions = n_convolutions, 
        n_filters= n_filters, 
        kernel_size=kernel_size, 
        initialization='kaiming_uniform'
    )
    net.to(device)

    return net

def train(net, epochs, learning_rate, trainloader):
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    start = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    end = time.time()
    
    return net, end - start

def save_model(net, dataset, lr, epochs, batch):
    path=MODELS_DIR / "inceptiontime"
    
    path.mkdir(parents=True, exist_ok=True)

    name = f"inceptiontime_{dataset}_{lr}_{epochs}_{batch}.pth"
    
    torch.save(net.state_dict(), path / name)

def predict(net, testloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in testloader:
            sequences, labels = data
            sequences = sequences.to(device)
            labels = labels.to(device)
            # calculate outputs by running sequences through the network
            outputs = net(sequences)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            all_preds.append(predicted)
            all_labels.append(labels)

    return torch.cat(all_labels), torch.cat(all_preds)

def evaluate(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=1)),
        "recall": float(recall_score(y_true, y_pred, average="macro"))
    }

def save_results(results, dataset, lr, epochs, batch, train_time, depth, n_convolutions, n_filters, kernel_size):
    save_run_results({
        "model": "InceptionTime",
        "dataset": dataset,
        **results,
        "train time": train_time,
        "learning rate": lr,
        "batch": batch,
        "epochs": epochs,
        "depth": depth,
        "convolutions": n_convolutions,
        "filters": n_filters,
        "kernel size": kernel_size,
        "device": device
    })
        
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

    depths = params["train_inceptiontime"]["depths"]
    learning_rates = params["train_inceptiontime"]["learning_rates"]
    epochs_list = params["train_inceptiontime"]["epochs"]
    batch = params["train_inceptiontime"]["batch"]
    convolutions= params["train_inceptiontime"]["n_convolutions"]
    filters= params["train_inceptiontime"]["n_filters"]
    kernels= params["train_inceptiontime"]["kernel_sizes"]

    for depth in depths:
        for lr in learning_rates:
            for n_convolutions in convolutions:
                for n_filters in filters:
                    for kernel_size in kernels:
                        for epoch in epochs_list:
                            for dataset in datasets:
                    
                                model=make_model(dataset, depth, n_convolutions, n_filters, kernel_size)
                    
                                trainloader, testloader = load_dataset(dataset, batch)
                                
                                model, train_time = train(model, epoch, lr, trainloader)
                    
                                y_true, y_pred = predict(model, testloader)
                    
                                eval_results = evaluate(y_true, y_pred)
                    
                                save_model(model, dataset, lr, epoch, batch)
                                save_results(eval_results, dataset, lr, epoch, batch, train_time, depth, n_convolutions, n_filters, kernel_size)


if __name__ == "__main__":
    app()
