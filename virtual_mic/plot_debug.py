import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

def test_mic():
    # 1. Print devices
    print("Default input device:", sd.default.device[0])
    target_in = sd.default.device[0]
    for i, dev in enumerate(sd.query_devices()):
        name = dev['name'].lower()
        if dev['max_input_channels'] > 0 and 'cable' not in name and 'steam' not in name:
            if 'realtek' in name or 'microphone array' in name:
                target_in = i
                break
                
    print("Selected Input Device:", target_in, sd.query_devices()[target_in]['name'])
    
    # Record 5 seconds directly
    print("Recording 5 seconds...")
    try:
        recording = sd.rec(int(5 * 48000), samplerate=48000, channels=1, dtype='float32', device=target_in)
        sd.wait()
        print("Recording finished. Array shape:", recording.shape, "Max val:", np.max(np.abs(recording)))
        print("Zeros count:", np.sum(recording == 0.0), "/", len(recording))
        
        plt.figure(figsize=(10, 4))
        plt.plot(recording, alpha=0.8, color='blue', label='Raw Microphone Input (48kHz)')
        plt.title(f"Audio Input Debug - Device {target_in}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig("debug_input_plot.png")
        print("Saved plot to debug_input_plot.png")
    except Exception as e:
        print("Failed to record:", e)

if __name__ == "__main__":
    test_mic()
