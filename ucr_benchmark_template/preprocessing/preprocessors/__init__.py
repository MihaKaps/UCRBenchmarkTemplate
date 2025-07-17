import importlib
from pathlib import Path

def dynamic_preprocess_model(model_name: str, dataset_name: str, input_dir: Path, output_dir: Path):
    """
    Dynamically imports the model-specific preprocessing module and calls its preprocess function.
    """
    try:
        module = importlib.import_module(f"ucr_benchmark_template.preprocessing.preprocessors.{model_name}")
        if hasattr(module, "preprocess"):
            module.preprocess(dataset_name, input_dir, output_dir)
        else:
            raise AttributeError(f"Module {model_name} has no function `preprocess`.")
    except ModuleNotFoundError:
        raise ImportError(f"No preprocessing module found for model: {model_name}")
