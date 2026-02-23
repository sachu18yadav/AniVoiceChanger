import numpy as np

class StreamBuffer:
    def __init__(self, target_size=6144): # ~128ms of audio at 48k for ultra-fast latency
        self.target_size = target_size
        self.buffer = np.array([], dtype=np.float32)

    def add(self, chunk):
        """Append a chunk and return the full buffer if target_size reached."""
        # Ensure chunk is flat
        self.buffer = np.concatenate([self.buffer, chunk.flatten()])

        if len(self.buffer) >= self.target_size:
            output = self.buffer[:self.target_size]
            # Keep the remainder
            self.buffer = self.buffer[self.target_size:]
            return output
        
        return None

    def clear(self):
        self.buffer = np.array([], dtype=np.float32)
