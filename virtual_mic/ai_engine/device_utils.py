try:
    import torch
except ImportError:
    torch = None

def get_best_device():
    """Detect if CUDA is available, otherwise fallback to CPU."""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def is_torch_available():
    return torch is not None
