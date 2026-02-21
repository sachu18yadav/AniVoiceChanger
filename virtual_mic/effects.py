try:
    import numpy as np
    from scipy import signal
    import pedalboard
    from pedalboard import Pedalboard, PitchShift, Chorus, HighpassFilter, Compressor, PeakFilter, HighShelfFilter
except ImportError as e:
    print(f"Pedalboard Import Error: {e}")
    np = None
    signal = None
    Pedalboard = None
    PitchShift = None
    Chorus = None
    HighpassFilter = None
    Compressor = None
    PeakFilter = None
    HighShelfFilter = None
except Exception as e:
    print(f"Pedalboard Generic Load Error: {e}")
    Pedalboard = None

class AnimeGirlVoice:
    """Professional voice modulation chain for the Anime Girl effect.
    Uses pedalboard for spectral sculpting to match researched anime vocal profiles.
    """
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
        self.board = None
        
        # Check if all components are available
        components = [Pedalboard, Compressor, HighpassFilter, PitchShift, PeakFilter, HighShelfFilter, Chorus]
        if any(c is None for c in components):
            return

        # Researched targets:
        # 1. High F0 (PitchShift)
        # 2. Narrow F1 resonance (PeakFilter)
        # 3. Bright mids/highs (HighShelf Filter)
        self.board = Pedalboard([
            Compressor(threshold_db=-20.0, ratio=4.0),
            HighpassFilter(cutoff_frequency_hz=350), # Aggressive low-end cut for 'youthful' clarity
            PitchShift(semitones=8.0),
            PeakFilter(cutoff_frequency_hz=1200, gain_db=4.0, q=1.5), # Enhance clarity 'moe' range
            HighShelfFilter(cutoff_frequency_hz=5000, gain_db=6.0), # Spectral brightness
            Chorus(rate_hz=0.8, depth=0.15, mix=0.08) # Subtle softening
        ])
        
    def process(self, audio_block, semitones=None):
        """Process a block of audio.
        semitones: if provided, overrides the default +8.0
        """
        if self.board is None or np is None:
            return audio_block
            
        # Update pitch if requested (index 2 in the chain)
        if semitones is not None and semitones != 0:
            self.board[2].semitones = semitones
        elif semitones == 0:
            self.board[2].semitones = 8.0 # Default Anime Girl
            
        # Pedalboard expects (channels, samples) or (samples,)
        audio = audio_block.T
        
        # Process through the board
        try:
            processed = self.board(audio, self.sample_rate, reset=False)
            
            if processed.size == 0:
                 return audio_block
                 
            return processed.T.astype(audio_block.dtype)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return audio_block

def noise_gate(audio_block, threshold=0.01):
    """Simple noise gate to remove background hiss."""
    if np is None: return audio_block
    mask = np.abs(audio_block) > threshold
    return audio_block * mask

def simple_echo(audio_block, delay_ms=200, decay=0.4, sample_rate=48000):
    """Placeholder for echo if needed, could also be a Pedalboard effect."""
    return audio_block # Simplified for now to focus on Anime Girl quality

# Legacy compatibility / functional wrappers
def pitch_shift(audio_block, semitones, sample_rate=48000):
    """Stateless pitch shift using pedalboard."""
    if Pedalboard is None: return audio_block
    shift = PitchShift(semitones=semitones)
    return shift(audio_block.T, sample_rate).T
