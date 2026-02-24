import sys
import numpy as np
import os
import time

def generate_sine_wave(freq, duration, sample_rate=48000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)

def test_pipeline():
    print("Initializing pipeline diagnostic...")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ai_engine import rvc_wrapper
    
    model_path = r"models\EarthshakerRVC\EarthshakerRVC.pth"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    converter = rvc_wrapper.RVCVoiceConverter(model_path, sample_rate=48000)
    converter.pitch = 0 # No pitch shift, keep it pure 

    # Generate strictly 1.8 seconds of pure 440Hz tone (exactly 3 chunks of 600ms)
    # 1.8s * 48000 = 86400 samples
    chunk_size = 28800
    duration = 1.8
    test_signal = generate_sine_wave(440.0, duration)
    
    with open("sine_diagnostic.txt", "w", encoding="utf-8") as clog:
        clog.write(f"Input Signal: {len(test_signal)} samples, Max={np.max(test_signal):.4f}, Min={np.min(test_signal):.4f}\n")
        
    processed_chunks = []
    
    for i in range(3):
        start = i * chunk_size
        end = start + chunk_size
        chunk = test_signal[start:end]
        
        t0 = time.time()
        out_chunk = converter.convert(chunk)
        t1 = time.time()
        
        proc_time = t1 - t0
        
        with open("sine_diagnostic.txt", "a", encoding="utf-8") as clog:
            if out_chunk is None:
                clog.write(f"Chunk {i+1}: FAILED INFERENCE\n")
                continue
                
            out_chunk = out_chunk.flatten()
            clog.write(f"Chunk {i+1}: In={len(chunk)} Out={len(out_chunk)} Time={proc_time:.2f}s Max={np.max(out_chunk):.4f} Min={np.min(out_chunk):.4f}\n")
        processed_chunks.append(out_chunk)
        
    if not processed_chunks:
        return
        
    final_output = np.concatenate(processed_chunks)
    with open("sine_diagnostic.txt", "a", encoding="utf-8") as clog:
        clog.write(f"Final Output Signal: {len(final_output)} samples\n")
        
        c1_end = processed_chunks[0][-5:]
        c2_start = processed_chunks[1][:5]
        clog.write(f"Boundary 1 (C1 end):   {c1_end}\n")
        clog.write(f"Boundary 1 (C2 start): {c2_start}\n")
        
    import scipy.io.wavfile as wavf
    wavf.write("diagnostic_in.wav", 48000, test_signal)
    wavf.write("diagnostic_out.wav", 48000, final_output)
    print("Saved diagnostic_in.wav and diagnostic_out.wav for review.")

if __name__ == "__main__":
    test_pipeline()
