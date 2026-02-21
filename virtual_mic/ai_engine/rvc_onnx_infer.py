import numpy as np
import os
try:
    import onnxruntime as ort
except ImportError:
    ort = None

class RVCOnnxConverter:
    """Lightweight RVC inference using ONNX Runtime."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.session = None
        self.providers = self._get_providers(device)
        
        if ort and os.path.exists(model_path):
            self._load_model()

    def _get_providers(self, device):
        if not ort: return []
        available = ort.get_available_providers()
        
        if device == "gpu":
            if "CUDAExecutionProvider" in available: return ["CUDAExecutionProvider"]
            if "DmlExecutionProvider" in available: return ["DmlExecutionProvider"]
        
        return ["CPUExecutionProvider"]

    def _load_model(self):
        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            print(f"Loaded RVC ONNX model from {self.model_path} using {self.session.get_providers()}")
        except Exception as e:
            print(f"Error loading RVC ONNX model: {e}")

    def convert(self, audio_np: np.ndarray) -> np.ndarray:
        """Process audio block through RVC ONNX model."""
        if not self.session:
            return audio_np # Fallback to passthrough
            
        try:
            # RVC ONNX usually expects [1, samples] or specific input names
            # This is a generic implementation assuming standard RVC ONNX exports
            input_name = self.session.get_inputs()[0].name
            
            # Ensure shape is [1, -1]
            input_data = audio_np.reshape(1, -1).astype(np.float32)
            
            outputs = self.session.run(None, {input_name: input_data})
            return outputs[0].flatten().astype(np.float32)
        except Exception as e:
            print(f"Inference error: {e}")
            return audio_np
