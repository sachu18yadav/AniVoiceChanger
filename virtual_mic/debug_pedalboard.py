import numpy as np
import sounddevice as sd
import time
from pedalboard import Pedalboard, PitchShift, Compressor, HighpassFilter

def test_pedalboard_blocks():
    # 2 seconds of a 440hz sine wave
    sr = 48000
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    board = Pedalboard([PitchShift(semitones=5)])
    
    print("Testing processing the whole array at once...")
    processed_whole = board(audio, sr, reset=False)
    print(f"Whole array max: {np.max(np.abs(processed_whole))}")
    
    print("Testing 1024 sample blocks...")
    chunks = np.array_split(audio, len(audio) // 1024)
    processed_blocks = []
    board_2 = Pedalboard([PitchShift(semitones=5)])
    for chunk in chunks:
        # Needs to be 2D for pedalboard
        chunk = chunk.reshape(1, -1)
        res = board_2(chunk, sr, reset=False)
        processed_blocks.append(res.flatten())
        
    final_blocks = np.concatenate(processed_blocks)
    print(f"Blocks array max: {np.max(np.abs(final_blocks))}")
    
    # Check if the blocks one went to 0
    zeros = np.sum(np.abs(final_blocks) < 0.001)
    print(f"Samples near zero in blocks array: {zeros} out of {len(final_blocks)}")

if __name__ == "__main__":
    test_pedalboard_blocks()
