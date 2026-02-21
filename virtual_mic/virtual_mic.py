import queue
import threading
import sys
import numpy as np
import time
import wave
import os
from typing import Optional, List, Callable

try:
    import sounddevice as sd
except ImportError:
    sd = None

from effects import pitch_shift, AnimeGirlVoice
from ai_engine.stream_buffer import StreamBuffer

# Default audio settings
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 1
DEFAULT_BLOCK_SIZE = 512 # Lower block size for <50ms processing latency

class VoiceChangerEngine:
    """Enhanced audio pipeline for production-grade voice changing."""
    
    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, block_size=DEFAULT_BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        self.current_effect = "passthrough"
        self.next_effect = None
        self.semitones = 0.0
        self.fade_index = 0
        self.fade_steps = 10 # Number of blocks for crossfade
        
        # New: Effects Configuration
        self.use_noise_gate = False
        self.gate_threshold = 0.002 # Lower default threshold
        self.use_echo = False
        self.echo_delay = 200
        self.echo_decay = 0.4

        self.worker_thread = None
        self.input_stream = None
        
        # AI Engine Components
        from effects import AnimeGirlVoice
        self.anime_voice = AnimeGirlVoice(sample_rate=self.sample_rate)
        self.ai_converter = None
        self.ai_buffer = StreamBuffer(target_size=4096)
        self.is_loading_ai = False
        self.use_fp16 = True # Default to half-precision for GPU
        
        # Test Voice Components
        self.is_testing = False
        self.test_frames = []
        self.last_recorded_original = None
        self.last_recorded_processed = None
        self.last_static_processed = None
        
        # Status & Level
        self.current_level = 0.0
        
        # Device Config
        self.input_device = None
        self.output_device = None

    def set_devices(self, input_idx: int, output_idx: int):
        self.input_device = input_idx
        self.output_device = output_idx

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[Audio Status] {status}", file=sys.stderr)
        # Update level for UI
        self.current_level = np.max(np.abs(indata))
        self.audio_queue.put(indata.copy())

    def _process_block(self, block, effect):
        """Apply the selected voice transformation and DSP chain."""
        processed = block.copy()
        
        # 1. Noise Gate (Early in chain)
        if self.use_noise_gate:
            from effects import noise_gate
            processed = noise_gate(processed, threshold=self.gate_threshold)

        # 2. Main Pitch/AI Conversion
        if effect == "anime_girl":
            processed = self.anime_voice.process(processed, self.semitones)
        elif effect == "pitch":
            # For raw pitch, we can still use anime_voice.process but it might apply filters
            # Let's use a specialized pitch shift if we want raw
            from effects import pitch_shift
            processed = pitch_shift(processed, self.semitones, self.sample_rate)
        elif effect == "ai":
            if self.ai_converter:
                frame = self.ai_buffer.add(processed)
                if frame is not None:
                    ai_output = self.ai_converter.convert(frame)
                    processed = ai_output.reshape(-1, 1)
                else:
                    processed = np.zeros_like(processed)
            else:
                # Fallback to Anime Girl DSP if AI not loaded but selected
                processed = self.anime_voice.process(processed, self.semitones)
        
        # 3. Modular Effects (Late in chain)
        if self.use_echo:
            from effects import simple_echo
            processed = simple_echo(processed, delay_ms=self.echo_delay, decay=self.echo_decay, sample_rate=self.sample_rate)
            
        return processed

    def _process_and_play(self):
        """Main processing loop with crossfade support."""
        try:
            # Query supported channels for the output device
            try:
                if self.output_device is None:
                    device_info = sd.query_devices(kind='output')
                else:
                    device_info = sd.query_devices(self.output_device)
                
                # Robustly get channels
                if isinstance(device_info, dict):
                    channels = min(DEFAULT_CHANNELS, device_info.get('max_output_channels', 1))
                else:
                    # Fallback if device_info is not a dict
                    channels = 1
            except:
                channels = 1
            
            if channels == 0: channels = 1 # Fallback
            
            with sd.OutputStream(device=self.output_device,
                                 samplerate=self.sample_rate,
                                 channels=channels,
                                 blocksize=self.block_size) as out_stream:
                while not self.stop_event.is_set():
                    try:
                        block = self.audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    # Handle Profile Switching with Crossfade
                    if self.next_effect and self.next_effect != self.current_effect:
                        fade_out = self._process_block(block, self.current_effect)
                        fade_in = self._process_block(block, self.next_effect)
                        
                        # Apply linear crossfade
                        alpha = self.fade_index / self.fade_steps
                        processed = (1.0 - alpha) * fade_out + alpha * fade_in
                        
                        self.fade_index += 1
                        if self.fade_index >= self.fade_steps:
                            self.current_effect = self.next_effect
                            self.next_effect = None
                            self.fade_index = 0
                    else:
                        processed = self._process_block(block, self.current_effect)

                    out_stream.write(processed)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Engine Error: {e}")
            self.stop_event.set()

    def start(self, effect="passthrough", semitones=0.0):
        if sd is None: return False
        
        self.current_effect = effect
        self.semitones = semitones
        self.stop_event.clear()
        self.audio_queue = queue.Queue()
        self.ai_buffer.clear()

        # Start output worker
        self.worker_thread = threading.Thread(target=self._process_and_play, daemon=True)
        self.worker_thread.start()

        # Start input stream
        try:
            self.input_stream = sd.InputStream(device=self.input_device,
                                              samplerate=self.sample_rate,
                                              channels=DEFAULT_CHANNELS,
                                              blocksize=self.block_size,
                                              callback=self._audio_callback)
            self.input_stream.start()
            return True
        except Exception as e:
            print(f"Input Error: {e}")
            self.stop_event.set()
            return False

    def stop(self):
        self.stop_event.set()
        if self.input_stream:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except: pass
            self.input_stream = None
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

    def switch_profile(self, new_effect):
        """Initiate a smooth transition to a new effect."""
        if self.is_loading_ai: return # Wait for AI to load
        
        if new_effect == "ai" and self.ai_converter is None:
            # Preload in background
            threading.Thread(target=self._load_ai_model, args=(new_effect,), daemon=True).start()
        else:
            self.next_effect = new_effect
            self.fade_index = 0

    def _load_ai_model(self, target_effect):
        self.is_loading_ai = True
        try:
            from ai_engine.model_utils import list_onnx_models, get_model_path
            onnx_models = list_onnx_models("models")
            
            if onnx_models:
                from ai_engine.rvc_onnx_infer import RVCOnnxConverter
                model_path = get_model_path("models", onnx_models[0])
                self.ai_converter = RVCOnnxConverter(model_path)
            else:
                # Fallback to PyTorch wrapper if ONNX not found
                try:
                    from ai_engine.rvc_wrapper import RVCVoiceConverter
                    pth_models = [f for f in os.listdir("models") if f.endswith(".pth")]
                    if pth_models:
                        model_path = get_model_path("models", pth_models[0])
                        self.ai_converter = RVCVoiceConverter(model_path, sample_rate=self.sample_rate)
                except ImportError:
                    print("RVC Wrapper not found.")
            
            if self.ai_converter:
                self.next_effect = target_effect
                self.fade_index = 0
            else:
                print("No AI models found in /models folder.")
        except Exception as e:
            print(f"Failed to load AI model: {e}")
        finally:
            self.is_loading_ai = False

    def get_devices(self):
        """Return lists of input and output devices."""
        if sd is None: return [], []
        try:
            devices = sd.query_devices()
            inputs = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            outputs = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_output_channels'] > 0]
            return inputs, outputs
        except:
            return [], []

    def run_test(self, duration=5.0, effect_mode=None, callback: Optional[Callable] = None):
        """Record, process in blocks, and playback to default output."""
        if effect_mode:
            self.current_effect = effect_mode
        def _test_thread():
            if sd is None: return
            
            try:
                # 1. Record
                # Prioritize self.input_device if set, otherwise use default
                in_dev = self.input_device if self.input_device is not None else sd.default.device[0]
                print(f"Recording {duration}s test from device {in_dev}...")
                
                recording = []
                def _rec_callback(indata, frames, time, status):
                    recording.append(indata.copy())
                    self.current_level = np.max(np.abs(indata))
                
                self.is_testing = True
                with sd.InputStream(device=in_dev, samplerate=self.sample_rate, channels=1, callback=_rec_callback):
                    time.sleep(duration)
                self.is_testing = False
                
                recording = np.concatenate(recording).flatten().astype(np.float32)
                rec_vol = np.max(np.abs(recording))
                mean_vol = np.mean(np.abs(recording))
                print(f"Recording finished. Samples: {len(recording)}. Max vol: {rec_vol:.4f}, Mean vol: {mean_vol:.4f}")
                
                if rec_vol < 0.001:
                    print("WARNING: Recording is silent or very quiet. Check your microphone device!")

                # 2. Process EVERYTHING at once for highest Pedalboard/PhaseVocoder quality
                print(f"Processing test phrase ({self.current_effect})...")
                
                # Reset effects state
                self.ai_buffer.clear()
                
                block = recording.reshape(-1, 1)
                final_audio = self._process_block(block, self.current_effect).flatten()
                
                # Normalize to prevent clipping/distortion in playback
                max_val = np.max(np.abs(final_audio))
                if max_val > 0.1: # Only normalize if it's loud enough to care
                    final_audio = final_audio / max_val * 0.9
                
                # Store for manual comparison (Layer 2)
                self.last_recorded_original = recording
                self.last_recorded_processed = final_audio
                
                # 3. Playback to DEFAULT output (physical speakers)
                out_dev = sd.default.device[1] 
                print(f"Playing back transformed test to device {out_dev}...")
                sd.play(final_audio, self.sample_rate, device=out_dev)
                sd.wait()
            except Exception as e:
                print(f"Test Error: {e}")
            
            if callback: 
                callback()

        threading.Thread(target=_test_thread, daemon=True).start()

    def run_static_test(self, file_path: str, callback: Optional[Callable] = None):
        """Layer 1: Process a static file and play it back."""
        def _static_thread():
            if not os.path.exists(file_path):
                print(f"Static file not found: {file_path}")
                if callback: callback()
                return
            
            try:
                import scipy.io.wavfile as wavfile
                sr, data = wavfile.read(file_path)
                
                # Convert to mono float32
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype != np.float32:
                    data = data.astype(np.float32) / np.max(np.abs(data))
                
                if len(data.shape) > 1:
                    data = data[:, 0]
                
                # Process in blocks
                processed_blocks = []
                for i in range(0, len(data), self.block_size):
                    block = data[i:i + self.block_size].reshape(-1, 1)
                    if len(block) < self.block_size:
                        block = np.pad(block, ((0, self.block_size - len(block)), (0, 0)))
                    
                    processed = self._process_block(block, self.current_effect)
                    processed_blocks.append(processed.flatten())
                
                final_audio = np.concatenate(processed_blocks)
                self.last_static_processed = final_audio
                
                # Playback
                out_dev = sd.default.device[1]
                print(f"Playing static test from {file_path}...")
                sd.play(final_audio, self.sample_rate, device=out_dev)
                sd.wait()
            except Exception as e:
                print(f"Static Test Error: {e}")
            
            if callback: callback()
        
        threading.Thread(target=_static_thread, daemon=True).start()

    def replay_recorded(self, original=False):
        """Replay Layer 2 audio (original or processed)."""
        audio = self.last_recorded_original if original else self.last_recorded_processed
        if audio is not None:
            out_dev = sd.default.device[1]
            sd.play(audio, self.sample_rate, device=out_dev)

    def replay_test(self):
        """Replay the last recorded test audio in a separate thread."""
        def _replay():
            if self.last_test_audio is not None and sd is not None:
                try:
                    out_dev = sd.default.device[1]
                    print(f"Replaying last test ({len(self.last_test_audio)} samples) to device {out_dev}...")
                    sd.play(self.last_test_audio, self.sample_rate, device=out_dev)
                    sd.wait()
                except Exception as e:
                    print(f"Replay Error: {e}")
            else:
                print("No test audio to replay or sounddevice unavailable.")
        threading.Thread(target=_replay, daemon=True).start()

    def get_performance_stats(self):
        """Return dict of performance metrics."""
        import psutil
        cpu = psutil.cpu_percent()
        q_size = self.audio_queue.qsize()
        # Estimate latency based on queue size and block size
        latency_ms = (q_size * self.block_size / self.sample_rate) * 1000
        
        return {
            "cpu": cpu,
            "latency": latency_ms,
            "buffer_health": max(0, 100 - (q_size * 5)) # Simple heuristic
        }
