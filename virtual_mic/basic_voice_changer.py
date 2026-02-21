import time
import queue
import threading
import sounddevice as sd
import numpy as np

try:
    import keyboard
except ImportError:
    print("Please install keyboard: pip install keyboard")
    keyboard = None

from virtual_mic import VoiceChangerEngine

# --- Configuration (Equivalent to AniVoiceChanger .env) ---
RECORD_KEY = 'v'
EFFECT_MODE = 'anime_girl' # or 'ai', 'passthrough'
INPUT_DEVICE = None # Default
OUTPUT_CABLE = None # We will try to auto-detect
OUTPUT_SPEAKER = None # Default

def find_cable_device():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if 'CABLE Input' in dev['name']:
            return i
    return None

class BasicVoiceChanger:
    def __init__(self):
        self.engine = VoiceChangerEngine(block_size=1024)
        self.is_recording = False
        self.record_queue = []
        self.sample_rate = 48000
        self.mic_stream = None
        
        self.cable_out = find_cable_device()
        self.speaker_out = sd.default.device[1]
        
    def _mic_callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self.record_queue.append(indata.copy())

    def on_press(self, event):
        if not self.is_recording:
            print(f"\n[RECORDING] Started. Speak now! (Holding '{RECORD_KEY}')")
            self.is_recording = True
            self.record_queue = []
            
            # Start mic stream
            in_dev = INPUT_DEVICE if INPUT_DEVICE is not None else sd.default.device[0]
            self.mic_stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._mic_callback, device=in_dev)
            self.mic_stream.start()

    def on_release(self, event):
        if self.is_recording:
            self.is_recording = False
            if self.mic_stream:
                self.mic_stream.stop()
                self.mic_stream.close()
                self.mic_stream = None
            
            if not self.record_queue:
                print("No audio recorded.")
                return
                
            print("[PROCESSING] Converting voice...")
            start_time = time.time()
            
            # Concatenate recorded block
            audio_data = np.concatenate(self.record_queue).flatten()
            
            # Ensure it's long enough
            if len(audio_data) < self.sample_rate * 0.5:
                print("Recording too short.")
                return

            # Process the entire recorded phrase at once for the highest quality DSP
            # (Pedalboard PitchShift requires larger context than 1024 samples)
            block = audio_data.reshape(-1, 1)
            final_audio = self.engine._process_block(block, EFFECT_MODE).flatten()
            
            print(f"[DONE] Time taken: {time.time() - start_time:.2f}s")
            
            # Playback
            self.play_audio(final_audio)

    def play_audio(self, audio_data):
        def _play(device_id, name):
            if device_id is not None:
                try:
                    # Using OutputStream instead of global sd.play to allow parallel playback
                    # convert 1D to 2D for sounddevice 
                    out_data = audio_data.reshape(-1, 1)
                    with sd.OutputStream(device=device_id, samplerate=self.sample_rate, channels=1) as stream:
                        stream.write(out_data)
                except Exception as e:
                    print(f"Failed to play on {name}: {e}")

        # Play in parallel to cable and speaker
        threads = []
        
        if self.cable_out is not None:
            threads.append(threading.Thread(target=_play, args=(self.cable_out, "Cable Output")))
        else:
            print("Warning: VB-Cable not found. Not routing to virtual mic.")
            
        threads.append(threading.Thread(target=_play, args=(self.speaker_out, "Speaker")))
        
        for t in threads: t.start()
        for t in threads: t.join()

def main():
    if not keyboard:
        return
        
    print("="*50)
    print(" ANI VOICE CHANGER (BASIC STANDALONE) ")
    print("="*50)
    
    app = BasicVoiceChanger()
    print(f"Push-to-talk key: '{RECORD_KEY}'")
    print(f"Active Effect: {EFFECT_MODE}")
    print("\nReady! Hold the key to talk, release to convert and send to game.")
    
    keyboard.on_press_key(RECORD_KEY, app.on_press)
    keyboard.on_release_key(RECORD_KEY, app.on_release)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
