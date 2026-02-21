import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from virtual_mic import VoiceChangerEngine

def test_block_processing():
    print("Testing VoiceChangerEngine._process_block...")
    engine = VoiceChangerEngine()
    
    # Create a dummy block (1 second) with enough energy to be audible
    block = (np.sin(np.linspace(0, 5000, 48000)) * 0.5).reshape(-1, 1).astype(np.float32)
    
    # Test passthrough
    output = engine._process_block(block, "passthrough")
    assert output.shape == block.shape, f"Expected shape {block.shape}, got {output.shape}"
    assert np.allclose(output, block), "Passthrough failed to return identical block"
    print("OK: Passthrough check passed.")
    
    # Test anime_girl (DSP)
    # This should return a block of the same size
    print("Running anime_girl effect...")
    output_anime = engine._process_block(block, "anime_girl")
    print(f"Input block: shape={block.shape}, max={np.max(np.abs(block)):.4f}")
    print(f"Output block: shape={output_anime.shape}, max={np.max(np.abs(output_anime)):.4f}")
    
    if np.allclose(output_anime, block):
        # Check if it was because it returned exactly the same array
        if output_anime is block:
            print("ERROR: Result is the EXACT SAME object (identity).")
        else:
            print(f"ERROR: Result values are identical to input. Diff sum: {np.sum(np.abs(output_anime - block))}")
            
    assert output_anime.shape == block.shape, f"Expected shape {block.shape}, got {output_anime.shape}"
    assert not np.allclose(output_anime, block), "Anime Girl effect returned unchanged block"
    print("OK: Anime Girl DSP check passed.")
    
    print("\nAll block processing tests passed!")

if __name__ == "__main__":
    test_block_processing()
