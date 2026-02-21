import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import time
from virtual_mic import VoiceChangerEngine

def test_basic_on_sample(filepath, effect="anime_girl"):
    print(f"Testing basic engine flow on {filepath}...")
    sample_rate, data = wavfile.read(filepath)
    
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32) / np.max(np.abs(data))
        
    if len(data.shape) > 1:
        data = data[:, 0]
        
    print(f"Original audio: {len(data)} samples, {len(data)/sample_rate:.2f} seconds")
        
    engine = VoiceChangerEngine(sample_rate=sample_rate, block_size=1024)
    
    start_time = time.time()
    
    # Process the whole block at once for highest Pedalboard DSP quality
    block = data.reshape(-1, 1)
    final_audio = engine._process_block(block, effect).flatten()
    
    print(f"Processed audio: {len(final_audio)} samples, {len(final_audio)/sample_rate:.2f} seconds")
    
    # Play to default speaker
    print("Playing processed audio...")
    sd.play(final_audio, sample_rate)
    sd.wait()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        test_basic_on_sample(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        test_basic_on_sample(sys.argv[1])
    else:
        print("Usage: python test_basic_sample.py <file> [effect]")
