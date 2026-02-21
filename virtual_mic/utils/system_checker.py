import os
import json
try:
    import torch
except ImportError:
    torch = None
import sounddevice as sd
from typing import Optional, Dict

class SystemChecker:
    @staticmethod
    def check_microphone() -> bool:
        """Check if any microphone is available."""
        devices = sd.query_devices()
        for dev in devices:
            if dev['max_input_channels'] > 0:
                return True
        return False

    @staticmethod
    def check_virtual_cable() -> Optional[int]:
        """Check for VB-Cable output device (Input for voice changer)."""
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            name = dev.get('name', '')
            if "CABLE Input" in name or "CABLE Output" in name:
                return idx
        return None

    @staticmethod
    def check_gpu() -> Dict[str, str]:
        """Check for GPU availability and type."""
        if torch is not None and torch.cuda.is_available():
            return {"status": "available", "type": "CUDA", "name": torch.cuda.get_device_name(0)}
        elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {"status": "available", "type": "MPS", "name": "Apple Metal"}
        return {"status": "not_available", "type": "CPU", "name": "None"}

    @staticmethod
    def check_models() -> bool:
        """Check if models folder exists and contains at least one model."""
        model_path = os.path.join(os.getcwd(), "models")
        if not os.path.exists(model_path):
            return False
        
        # Check for .pth files
        models = [f for f in os.listdir(model_path) if f.endswith(".pth")]
        return len(models) > 0

    @classmethod
    def get_readiness_report(cls) -> Dict:
        """Get a full report of system readiness."""
        return {
            "mic": cls.check_microphone(),
            "virtual_cable": cls.check_virtual_cable() is not None,
            "gpu": cls.check_gpu(),
            "models": cls.check_models()
        }
