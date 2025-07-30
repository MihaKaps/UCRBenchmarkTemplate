# UC Benchmark Template

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party source - UCRBenchmark_2018.zip and its metadata file
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
|   │   ├── raw_splits <- fixed labels and dropped nans in original train/test split from unzipped folders
|   │   ├── kan        <- tarin/validation/test split
|   │   └── mlp        <- train/test split
│   ├── results        <- Results produced after training and evaluation.
│   └── raw            <- Original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models
│   ├── inceptiontime  <- Saved InceptionTime models
│   ├── kan            <- Saved Kolmogorov–Arnold Network models
│   └── mlp            <- Saved MLP models
│
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ucr_benchmark_template and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── dvc.yaml           <- DVC pipeline definition file
│
├── params.yaml        <- Parameters for preprocessing and training
│
└── ucr_benchmark_template.   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ucr_benchmark_template a Python module
    │
    ├── __main__.py             <- Optional main execution entry point
    │
    ├── config.py               <- Shared configuration and paths
    │
    ├── dataset.py              <- Functions to load or prepare datasets
    │
    ├── analyze_results.py      <- Analyze and summarize results across models
    │
    ├── save_results.py         <- Save metrics and metadata into CSV logs
    │
    ├── preprocessing          
    |   |  
    │   ├── preprocessors       
    |   |   ├── __init__.py     
    |   |   ├── mlp.py          <- Preprocesses data for MLP (train/test split)
    |   |   └── kan.py          <- Preprocesses data for KAN (train/val/test split)
    |   |
    │   ├── utils                          
    |   |   └── fix_labels.py   <- Normalize class labels to start from 0
    |   |
    │   └── preprocess.py       <- Pipeline to preprocess selected datasets
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── train.py                  <- Train, predict, evaluate MLP
    │   ├── train_kan.py              <- Train, predict, evaluate KAN
    │   └── train_inceptiontime.py    <- Train, predict, evaluate InceptionTime
    │
    └── plots.py                <- Code to create visualizations
```

# How to Add a Training Script to the DVC Pipeline

This section explains how to plug a new training script (like `train_kan.py`) into the DVC pipeline.

## Prerequisites

### Install dependencies
```bash
# For exact reproducibility (recommended)
pip install -r requirements-lock.txt

# For development (if you need newer versions)
pip install -r requirements.txt
```

### Add UCR dataset
Place `UCRArchive_2018.zip` in:
```
data/external/UCRArchive_2018.zip
```

## Step-by-Step Integration of a New Training Script

### 1. Write your training script

Put your training code under:
```
ucr_benchmark_template/modeling/train_<yourmodel>.py
```

The script should:
- Load datasets from `data/processed/<type>/`
- Accept parameters from `params.yaml`
- Train model for all hyperparameter combinations
- Save results by calling `save_run_results` with a dictionary containing model name, dataset, evaluation metrics, training time, and all hyperparameters
- Save trained model to `models/<yourmodel>/`

### Example: train_kan.py

- Loads `data/processed/kan`
- Builds KAN with `make_model(...)`
- Evaluates using accuracy, F1, precision, recall
- Stores model in `models/kan/`
- Saves results in `data/results/all_results.csv` 

### 2. Add model-specific hyperparameters to params.yaml

```yaml
train_kan:
  depths: [2]
  hidden_layers: [10]
  learning_rates: [0.001]
  epochs: [500]
  batch: 16
  grid: [5]
  spline_order: 3
```

### 3. Add your training stage to dvc.yaml

```yaml
train_kan:
  cmd: python -m ucr_benchmark_template.modeling.train_kan
  deps:
    - data/processed/
    - ucr_benchmark_template/modeling/train_kan.py
    - params.yaml
  outs:
    - models/kan/
  params:
    - train_kan.depths
    - train_kan.hidden_layers
    - train_kan.learning_rates
    - train_kan.epochs
    - train_kan.batch
    - train_kan.grid
    - train_kan.spline_order
```

# Running Experiments

Once your models are integrated, here's how to run and track experiments.

## Basic Pipeline Commands

### Run the Full Pipeline
```bash
# Run all stages from data preprocessing to results analysis
dvc repro

# Check what stages will run
dvc status
```

### Run Individual Stages
```bash
# Run specific training stage
dvc repro train_mlp
dvc repro train_kan  
dvc repro train_inceptiontime

# Run preprocessing only
dvc repro preprocess_data

# Run results analysis
dvc repro process_results
```

## Experiment Tracking

### Named Experiments
```bash
# Run experiment with custom name
dvc exp run -n "baseline_experiment"

# Run specific stage as experiment
dvc exp run -n "mlp_test" train_mlp
```

### Setting Hyperparameters
```bash
# Modify MLP parameters
dvc exp run -n "mlp_lr_01" --set-param train_mlp.learning_rates="[0.01]"

# Modify KAN architecture
dvc exp run -n "kan_depth_3" --set-param train_kan.depths="[3]"

# Change datasets to test on
dvc exp run -n "small_test" --set-param preprocess.datasets="[Coffee, Beef, OliveOil]"

# Train only specific models
dvc exp run -n "mlp_only" --set-param preprocess.models="[mlp]"
```

### Queue Multiple Experiments
```bash
# Queue several experiments
dvc exp run --queue -n "mlp_lr_001" --set-param train_mlp.learning_rates="[0.001]"
dvc exp run --queue -n "mlp_lr_01" --set-param train_mlp.learning_rates="[0.01]"
dvc exp run --queue -n "mlp_lr_1" --set-param train_mlp.learning_rates="[0.1]"

# Run all queued experiments
dvc exp run --run-all
```

## Viewing Results

### Show Experiments
```bash
# List all experiments
dvc exp show

# Show only changed parameters and metrics
dvc exp show --only-changed

# Show specific experiments
dvc exp show baseline_experiment mlp_lr_01
```

## How the Pipeline Works

- **unzip_data**: unzips `UCRArchive_2018.zip` to `data/raw/` (skipped if already extracted)
- **preprocess_data** creates:
  - `data/processed/raw_splits`: fixed labels (unified class 0) from raw UCR
  - `data/processed/mlp`: train/test splits
  - `data/processed/kan`: train/val/test splits
- **train_***: model training stages consume processed data and store results/models
- **process_results**: analyzes all results and creates summary files

## File Locations Summary

| Purpose | Path |
|---------|------|
| Raw UCR data | `data/external/UCRArchive_2018.zip` |
| Preprocessed | `data/processed/` |
| Saved models | `models/<model_name>/` |
| Run results | `data/results/all_results.csv` |
| Params | `params.yaml` |
| Training code | `ucr_benchmark_template/modeling/` |