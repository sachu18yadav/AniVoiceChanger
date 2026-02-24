import numpy as np

class StreamBuffer:
    """
    Sliding-window buffer for realtime RVC inference.
    target_size : number of samples fed to the model each inference step
                  (needs to be large enough for RVC to extract valid pitch)
    step_size   : how many NEW samples advance the window each time a frame fires
                  (= target_size gives non-overlapping chunks;
                   < target_size gives overlapping chunks \u2014 smoother boundaries)
    """
    def __init__(self, target_size=24000, step_size=None):
        # 24000 @ 48 kHz = 500 ms \u2014 minimum for coherent RVC inference
        self.target_size = target_size
        # Default: 50 % overlap  (step = 12000 \u2192 250 ms cadence)
        self.step_size = step_size if step_size is not None else target_size // 2
        self.buffer = np.array([], dtype=np.float32)

    def add(self, chunk):
        """
        Append chunk and return a full window when ready.
        Returns the full `target_size` window every `step_size` new samples,
        keeping the history so boundary context is preserved.
        """
        self.buffer = np.concatenate([self.buffer, chunk.flatten()])

        if len(self.buffer) >= self.target_size:
            output = self.buffer[:self.target_size].copy()
            # Slide window: discard only the oldest step_size samples
            self.buffer = self.buffer[self.step_size:]
            return output

        return None

    def clear(self):
        self.buffer = np.array([], dtype=np.float32)
