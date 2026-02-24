import os
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import sys

# Add virtual_mic to path for relative imports if run from inside it
sys.path.append(os.path.dirname(__file__))

# Import Engine 
from ai_engine import rvc_wrapper
from core_state import AppState

def generate_debug_graphs():
    # 1. Grab Mic
    target_in = sd.default.device[0]
    for i, dev in enumerate(sd.query_devices()):
        name = dev['name'].lower()
        if dev['max_input_channels'] > 0 and 'cable' not in name and 'steam' not in name:
            if 'realtek' in name or 'microphone array' in name:
                target_in = i
                break
                
    print(f"Recording 5 seconds of audio from device {target_in}...")
    recording = sd.rec(int(5 * 48000), samplerate=48000, channels=1, dtype='float32', device=target_in)
    sd.wait()
    recording = recording.flatten()
    print("Record done. Max:", np.max(np.abs(recording)))
    
    # 2. Boot Engine and Run AI
    print("Loading AI Model...")
    app_state = AppState()
    model_dir = os.path.join("models", app_state.profile)
    model_pth = os.path.join(model_dir, f"{app_state.profile}.pth")
    if not os.path.exists(model_pth): model_pth = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_pth):
        pths = [f for f in os.listdir(model_dir) if f.endswith(".pth") and not f.startswith("D_")]
        if pths: model_pth = os.path.join(model_dir, pths[0])
        
    converter = rvc_wrapper.RVCVoiceConverter(model_pth, sample_rate=48000)
    converter.pitch = app_state.pitch
    
    print("Processing 600ms chunks to map PTT logic...")
    chunk_size = 28800
    processed_chunks = []
    
    pad_len = chunk_size - (len(recording) % chunk_size)
    padded_rec = np.pad(recording, (0, pad_len), mode='constant') if pad_len != chunk_size else recording
    
    for c_idx in range(len(padded_rec) // chunk_size):
        start = c_idx * chunk_size
        chunk_48k = padded_rec[start:start+chunk_size]
        
        chunk_out_48k = converter.convert(chunk_48k)
        if chunk_out_48k is not None:
            processed_chunks.append(chunk_out_48k.flatten())
        else:
            processed_chunks.append(chunk_48k)
            
    final_audio = np.concatenate(processed_chunks) if processed_chunks else recording
    if pad_len != chunk_size: final_audio = final_audio[:-pad_len]
    
    # 3. Create Plots
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(recording, color='blue', alpha=0.9)
    plt.title("RAW INPUT (Realtek Microphone, 48kHz)")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(final_audio, color='orange', alpha=0.9)
    plt.title("AI OUTPUT (RVC Synthesized Audio, 48kHz)")
    plt.xlabel("Samples (5 Seconds)")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to artifacts explicitly
    artifact_path = r"C:\Users\comei\.gemini\antigravity\brain\7393e4d1-0e33-4a26-9b51-a2fc20a32bd0\audio_pipeline_analysis.png"
    plt.savefig(artifact_path)
    print(f"Plots saved successfully to: {artifact_path}")

if __name__ == "__main__":
    generate_debug_graphs()
