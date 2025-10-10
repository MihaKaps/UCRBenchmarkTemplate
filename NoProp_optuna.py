import importlib
from ucr_benchmark_template.modeling import train_noprop

importlib.reload(train_noprop)

from ucr_benchmark_template.modeling.train_noprop import (
    load_dataset, make_model, train, predict, evaluate, save_model
)

import optuna
import numpy as np
import pandas as pd
import os
import logging
import traceback

LOG_DIR = "."
LOG_FILE = os.path.join(LOG_DIR, "optuna_run_full.log")

# Ensure the folder exists
os.makedirs(LOG_DIR, exist_ok=True)

# Clear any existing handlers (important if Optuna or Jupyter adds one)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Now configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()  # optional: also show logs in console
    ]
)

logging.info("=== Logging system initialized ===")

def objective(trial):
    try:
        datasets = [
        "ACSF1", "Adiac", "ArrowHead", "BME", "Beef", "BeetleFly", "BirdChicken", "CBF", "Car", "Chinatown",
        "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers","CricketX", "CricketY", "CricketZ", "Crop",
        "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW",
        "ECG200", "ECG5000", "ECGFiveDays", "EOGHorizontalSignal", "EOGVerticalSignal", "Earthquakes", "ElectricDevices",
        "EthanolLevel", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "FreezerRegularTrain",
        "FreezerSmallTrain", "GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "Ham",
        "HandOutlines", "Haptics", "Herring", "HouseTwenty", "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain",
        "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat",
        "MedicalImages", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW",
        "MixedShapesRegularTrain", "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1",
        "NonInvasiveFetalECGThorax2", "OSULeaf", "OliveOil", "PhalangesOutlinesCorrect", "Phoneme", "PigAirwayPressure",
        "PigArtPressure", "PigCVP", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect",
        "ProximalPhalanxTW", "RefrigerationDevices", "Rock", "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2",
        "SemgHandSubjectCh2", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SmoothSubspace",
        "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols",
        "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UMD",
        "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine",
        "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"
    ]
    
        # --- Hyperparameters ---
        T = trial.suggest_int("T", 1, 10)
        emb_d = trial.suggest_categorical("embedding_dim", [0, 16, 32, 64, 128])
        eta = trial.suggest_float("eta", 0.1, 1, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        k_size = trial.suggest_categorical("k_size", [3, 5, 7])
        dropout = trial.suggest_float("dropout", 0, 0.4, step=0.1)
        
        n_blocks = trial.suggest_int("n_blocks", 1, 4)
        base_ch = trial.suggest_categorical("base_channels", [8, 16, 32, 64])
        channels = [base_ch * (2 ** i) for i in range(n_blocks)]
        
        n_layers = trial.suggest_int("n_layers", 1, 3)
        fc_layers = n_layers * [256]

        n_merged = trial.suggest_int("n_merged", 1, 4)
        base = trial.suggest_categorical("base_merged", [512, 256, 128])
        fc_merged = [max(base // (2 ** i), 32) for i in range(n_merged)]

        hp_str = (
            f"T={T}, emb={emb_d}, eta={eta}, lr={lr:.1e}, k={k_size}, dpout={dropout}, "
            f"ch={channels}, fc={fc_layers}, merg={fc_merged}"
        )

        logging.info(f"=== Starting Trial {trial.number} ===")
        logging.info(f"Hyperparameters: {hp_str}")
        
        accuracies = []
        for dataset in datasets:
            try:
                logging.info(f"Starting dataset: {dataset}")
                trainloader, testloader = load_dataset(dataset, batch_size=64)
                model = make_model(dataset, emb_d, T, k_size, dropout, channels, fc_layers, fc_merged)
                model, _ = train(model, trainloader, 600, T, eta, lr, 1e-5)
                y_true, y_pred = predict(model, testloader)
                acc = evaluate(y_true, y_pred)["accuracy"]

                accuracies.append(acc)
                logging.info(f"Finished {dataset} → Accuracy: {acc:.4f}")

            except Exception as e:
                error_msg = f"Error on dataset {dataset}: {e}"
                logging.error(error_msg)
                logging.error(traceback.format_exc())
                accuracies.append(np.nan)
                continue
        
        avg_acc = np.nanmean(accuracies)
        logging.info(f"Trial {trial.number} completed with average accuracy: {avg_acc:.4f}")
    
        # --- Save results to CSV ---
        csv_path = "optuna_results_full.csv"
    
        if not os.path.exists(csv_path):
            df = pd.DataFrame({"Dataset": datasets + ["AVERAGE"]})
        else:
            df = pd.read_csv(csv_path)
    
        trial_name = f"Trial_{trial.number+1}"
        run_data = accuracies + [avg_acc]
        df[trial_name] = run_data

        # Explicitly create string column for hyperparameters
        col_name = f"{trial_name}_params"
        if col_name not in df.columns:
            df[col_name] = pd.Series(dtype="object")
        df.loc[df["Dataset"] == "AVERAGE", col_name] = hp_str

        df.to_csv(csv_path, index=False)
    
        return avg_acc

    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {e}")
        logging.error(traceback.format_exc())
        return np.nan

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
