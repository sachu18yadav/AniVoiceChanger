import os
from typing import List

def list_onnx_models(models_dir: str) -> List[str]:
    """Return a list of available ONNX model files in the directory."""
    if not os.path.exists(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith(".onnx")]

def get_model_path(models_dir: str, model_name: str) -> str:
    """Get the full path to a model file."""
    return os.path.join(models_dir, model_name)
