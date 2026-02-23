import numpy as np
try:
    import torch
except ImportError:
    torch = None
from .device_utils import get_best_device, is_torch_available

class RVCVoiceConverter:
    def __init__(self, model_path, sample_rate=48000):
        self.device = get_best_device()
        self.sample_rate = sample_rate
        self.model = None
        self.info = None
        self.index = None
        self.big_npy = None
        self.pitch = 12

        # Initialize the actual RVC model securely
        self._load_model(model_path)

    def _load_model(self, model_path):
        """Load the PyTorch model via rvc_infer."""
        import sys
        import os
        rvc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'AniVoiceChanger'))
        if rvc_path not in sys.path:
            sys.path.append(rvc_path)
            
        try:
            import rvc_infer
            self.model, self.info = rvc_infer.load_rvc_model(model_path)
            
            if self.model is None:
                print(f"Warning: RVC Wrapper failed to load {model_path}. File may be missing or uncompiled.")
                return
            
            # Auto-detect nearby index file if it exists
            model_dir = os.path.dirname(model_path)
            idxs = [f for f in os.listdir(model_dir) if f.endswith(".index")]
            if idxs:
                index_path = os.path.join(model_dir, idxs[0])
                self.index, self.big_npy = rvc_infer.load_index(index_path)
                
            print(f"RVC Wrapper loaded model successfully: {model_path}")
        except Exception as e:
            print(f"Failed to load AI model in RVC Wrapper: {e}")

    def convert(self, audio_np):
        """
        Processes a block of audio through the RVC AI model.
        Returns a processed numpy array at 48k.
        """
        if self.model is None:
            return audio_np  # Passthrough if no model

        import sys
        import os
        rvc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'AniVoiceChanger'))
        if rvc_path not in sys.path: sys.path.append(rvc_path)
        import rvc_infer
        
        try:
            # RVC requires flat arrays, usually 16k minimum. rvc_infer handles the internal conversions.
            converted, out_sr = rvc_infer.infer(
                audio_np.flatten(), 
                self.sample_rate, 
                self.model, 
                self.info, 
                f0_up_key=self.pitch, 
                index=self.index, 
                big_npy=self.big_npy
            )
            
            if out_sr != self.sample_rate:
                import librosa
                converted = librosa.resample(converted, orig_sr=out_sr, target_sr=self.sample_rate)
                
            return converted.reshape(-1, 1).astype(np.float32)
        except Exception as e:
            print(f"AI Streaming Error: {e}")
            return audio_np
