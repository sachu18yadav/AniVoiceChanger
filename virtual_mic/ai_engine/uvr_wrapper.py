import os
import tempfile
import threading
import numpy as np
import soundfile as sf
from typing import Optional

# We delay the import so the parent application doesn't crash if the module isn't installed
try:
    from audio_separator.separator import Separator
    HAS_SEPARATOR = True
except ImportError:
    HAS_SEPARATOR = False

class UVRPreprocessor:
    """
    In-memory wrapper for the Ultimate Vocal Remover (UVR) VR Architecture denoise models.
    Takes live microphone chunks as Numpy arrays, passes them securely through the heavy 
    AI AI separator on a dedicated thread, and returns the noise-reduced chunk.
    """
    def __init__(self, sample_rate: int = 48000, model_name: str = "UVR-DeNoise-Lite"):
        self.sample_rate = sample_rate
        # "UVR-DeNoise-Lite.pth" is extremely fast. "UVR-DeNoise.pth" is full quality.
        self.model_name = model_name
        self._lock = threading.Lock()
        
        self.separator = None
        self.is_ready = False
        
        if not HAS_SEPARATOR:
            print("WARNING: audio-separator not installed. UVR Denoising is disabled.")
            return
            
        # The separator requires a hard directory to download its ONNX/PTH weights to
        self.model_dir = os.path.join(os.getcwd(), "models", "uvr_models")
        self.output_dir = os.path.join(os.getcwd(), "models", "uvr_temp")
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        import logging
        # Initialize the hardware separator pipeline configuration
        self.separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=self.model_dir,
            output_dir=self.output_dir,
            output_format="WAV",
            use_autocast=True # Hardware acceleration enabled
        )
        
        # In a background thread, load the model (downloads automatically if missing)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            print(f"[UVR Engine] Initializing AI Denoise Architecture ({self.model_name})...")
            # This triggers download and ONNX compilation
            self.separator.load_model(model_filename=self.model_name + ".pth")
            self.is_ready = True
            print(f"[UVR Engine] [OK] DeNoise Model Loaded Successfully.")
        except Exception as e:
            print(f"[UVR Engine] Failed to load UVR model: {e}")
            self.separator = None

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Takes a highly chaotic real-time Numpy block [N, 1] ranging from -1.0 to 1.0, 
        and applies strict neural denoising without bleeding context.
        """
        if not self.is_ready or self.separator is None:
            return audio_chunk # Pass-through if booting or failed
            
        with self._lock:
            # Most audio-separator implementations require a physical file path 
            # because they rely on underlying C-based ffmpeg extractors.
            # We bypass this using Windows NamedTemporaryFiles for rapid pseudo-RAM disk I/O.
            try:
                # 1. Dump real-time chunk to a temporary wave file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.output_dir) as temp_in:
                    input_path = temp_in.name
                
                sf.write(input_path, audio_chunk.flatten(), self.sample_rate)
                print(f"[UVR Engine] Chunk saved to {input_path}")
                
                # 2. Run VR Architecture Separator
                print(f"[UVR Engine] Starting separation...")
                stems = self.separator.separate(input_path)
                print(f"[UVR Engine] Separation done. Stems: {stems}")
                
                if not stems:
                    print(f"[UVR Engine] No stems returned!")
                    return audio_chunk
                    
                # The first stem is typically the Primary Stem (e.g., Vocals / Clean)
                primary_stem_path = stems[0]
                
                # 3. Read the cleaned output back into pure numpy
                clean_path = os.path.join(self.output_dir, primary_stem_path)
                print(f"[UVR Engine] Reading clean audio from {clean_path}")
                clean_audio, sr = sf.read(clean_path)
                
                # Resample safety net (audio-separator may forcibly lock to 44100Hz)
                if sr != self.sample_rate:
                    print(f"[UVR Engine] Resampling from {sr} to {self.sample_rate}")
                    import librosa
                    clean_audio = librosa.resample(clean_audio, orig_sr=sr, target_sr=self.sample_rate)
                
                # 4. Clean up temporary files
                print(f"[UVR Engine] Cleaning up temp files...")
                try: os.remove(input_path)
                except: pass
                
                for stem in stems:
                    try: os.remove(os.path.join(self.output_dir, stem))
                    except: pass
                
                # 5. Prevent silent array collapsing
                if len(clean_audio) == 0:
                    print(f"[UVR Engine] Empty clean_audio!")
                    return audio_chunk 
                    
                # Standardize array geometry
                print(f"[UVR Engine] Processing complete.")
                return clean_audio.reshape(-1, 1).astype(np.float32)
                
            except Exception as e:
                import traceback
                print(f"[UVR Drop] {e}")
                # traceback.print_exc()
                return audio_chunk
