import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import sys
import os

# Add local path to import engine/effects
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from effects import AnimeGirlVoice

def test_file(file_path, semitones=0.0):
    print(f"Loading {file_path}...")
    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Convert to float32 and mono if necessary
    if data.dtype != np.float32:
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        else:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
    
    if len(data.shape) > 1:
        data = data[:, 0]
    
    print(f"Processing with Pedalboard (Anime Girl)...")
    voice = AnimeGirlVoice(sample_rate=sample_rate)
    
    # Process in blocks to simulate real-time
    block_size = 1024
    processed_blocks = []
    
    for i in range(0, len(data), block_size):
        block = data[i:i + block_size].reshape(-1, 1)
        if len(block) < block_size:
            block = np.pad(block, ((0, block_size - len(block)), (0, 0)))
        
        # Process block
        processed = voice.process(block, semitones)
        processed_blocks.append(processed.flatten())
    
    final_audio = np.concatenate(processed_blocks)
    
    print("Playing original...")
    sd.play(data, sample_rate)
    sd.wait()
    
    print("Playing modulated (Anime Girl)...")
    sd.play(final_audio, sample_rate)
    sd.wait()
    
    # Save result
    output_path = "modulated_test.wav"
    wavfile.write(output_path, sample_rate, (final_audio * 32767).astype(np.int16))
    print(f"Saved modulated audio to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file(sys.argv[1])
    else:
        print("Usage: python test_modulation_file.py <path_to_wav>")
        print("Example: python test_modulation_file.py sample.wav")
