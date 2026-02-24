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
        
        # Effects Configuration
        self.use_noise_gate = False
        self.gate_threshold = 0.002
        self.use_echo = False
        self.echo_delay = 200
        self.echo_decay = 0.4

        self.worker_thread = None
        self.ai_worker_thread = None  # Dedicated AI inference thread
        self.input_stream = None
        self.input_device = None
        
        # Auto-detect VB-Audio Cable for Output
        self.output_device = None
        if sd:
            try:
                for i, dev in enumerate(sd.query_devices()):
                    if dev['max_output_channels'] > 0 and 'cable input' in dev['name'].lower() and 'vb-audio' in dev['name'].lower():
                        self.output_device = i
                        break
            except Exception: pass
        
        # AI Engine Components
        from effects import AnimeGirlVoice
        self.anime_voice = AnimeGirlVoice(sample_rate=self.sample_rate)
        self.ai_converter = None
        # Decoupled AI pipeline queues:
        # audio_queue -> [AI worker] -> ai_out_queue -> [output thread]
        self.ai_in_queue = queue.Queue(maxsize=4)   # raw input chunks for inference
        self.ai_out_queue_samples = []              # flat float32 sample output ring
        self.ai_out_lock = threading.Lock()
        # 28800 (600ms) window, step 24000 (500ms) = 100ms true overlap context
        self.ai_buffer = StreamBuffer(target_size=28800, step_size=24000)
        self._last_ai_overlap = None                # Holds the previous overlap for crossfading
        self._last_played_audio = np.array([], dtype=np.float32)
        self._starving_frames = 0

        self.is_loading_ai = False
        self.use_fp16 = True
        
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
        
        # Apply static Pre-Gain to cleanly boost microphone levels without destroying dynamics
        processed = processed * 3.0
        processed = np.clip(processed, -1.0, 1.0)
        
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
                # Stage 1: buffer input into chunks, push completed frames to inference queue
                frame = self.ai_buffer.add(processed)
                if frame is not None:
                    try:
                        self.ai_in_queue.put_nowait(frame)  # non-blocking; drop if full
                    except queue.Full:
                        pass  # inference can't keep up; skip frame, avoid blocking output
                
                # Stage 2: drain the output sample ring to fill this block
                with self.ai_out_lock:
                    avail = len(self.ai_out_queue_samples)
                    need = len(processed.flatten())
                    if avail >= need:
                        chunk = self.ai_out_queue_samples[:need]
                        self.ai_out_queue_samples = self.ai_out_queue_samples[need:]
                        processed = np.array(chunk, dtype=np.float32).reshape(-1, 1)
                        # Update history
                        hist_len = len(self._last_played_audio)
                        if need >= hist_len:
                            self._last_played_audio = processed[-hist_len:].flatten().copy()
                        else:
                            self._last_played_audio = np.roll(self._last_played_audio, -need)
                            self._last_played_audio[-need:] = processed.flatten()
                        self._starving_frames = 0
                    else:
                        # Starvation! CPU is slower than realtime.
                        # Instead of a jarring chunk cutoff or a buzz loop, we do a soft fade to silence.
                        fallback = np.zeros_like(processed.flatten(), dtype=np.float32)
                        
                        # If we just started starving, apply a quick 10ms fade out to the very last played sample
                        # to prevent a speaker pop
                        if self._starving_frames == 0 and len(self._last_played_audio) > 0:
                            fade_len = min(need, 480) # 10ms
                            last_val = self._last_played_audio[-1]
                            fade = np.linspace(last_val, 0.0, fade_len, dtype=np.float32)
                            fallback[:fade_len] = fade
                            
                        self._starving_frames += 1
                        processed = fallback.reshape(-1, 1)
            else:
                processed = self.anime_voice.process(processed, self.semitones)
        
        # 3. Modular Effects (Late in chain)
        if self.use_echo:
            from effects import simple_echo
            processed = simple_echo(processed, delay_ms=self.echo_delay, decay=self.echo_decay, sample_rate=self.sample_rate)
            
        return processed.astype(np.float32)

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

    def _ai_inference_worker(self):
        """Dedicated background thread: pulls buffered frames, runs RVC inference."""
        # 100ms overlap
        xfade_samples = int(self.sample_rate * 0.10)
        
        while not self.stop_event.is_set():
            try:
                frame = self.ai_in_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                ai_output = self.ai_converter.convert(frame)
                if ai_output is not None:
                    ai_flat = ai_output.flatten()
                    
                    # Exact length expected based on ratios
                    target_len = self.ai_buffer.target_size
                    step_len = self.ai_buffer.step_size
                    
                    # Ensure we have enough audio for OLA
                    if len(ai_flat) >= target_len:
                        
                        # The newly generated chunk (e.g. 600ms)
                        new_chunk = ai_flat[-target_len:].copy()
                        
                        # 1. Crossfade the FIRST 100ms with the saved overlap from the PREVIOUS chunk
                        if self._last_ai_overlap is not None and len(self._last_ai_overlap) == xfade_samples:
                            fade_in = np.linspace(0, 1, xfade_samples, dtype=np.float32)
                            fade_out = np.linspace(1, 0, xfade_samples, dtype=np.float32)
                            new_chunk[:xfade_samples] = (new_chunk[:xfade_samples] * fade_in) + (self._last_ai_overlap * fade_out)
                            
                        # 2. Extract the payload we actually want to play (the 500ms step)
                        # This includes the crossfaded part, plus the middle part
                        play_chunk = new_chunk[:step_len].copy()
                        
                        # 3. Save the LAST 100ms as the overlap for the NEXT chunk
                        self._last_ai_overlap = new_chunk[step_len:target_len].copy()
                        
                        samples = play_chunk.tolist()
                    else:
                        # Fallback if array size is weird (shouldn't happen with strict streaming)
                        samples = ai_flat.tolist()
                        self._last_ai_overlap = None
                    
                    with self.ai_out_lock:
                        self.ai_out_queue_samples.extend(samples)
                        # Prevent unbounded growth: cap to ~2s of audio
                        cap = self.sample_rate * 2
                        if len(self.ai_out_queue_samples) > cap:
                            self.ai_out_queue_samples = self.ai_out_queue_samples[-cap:]
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[AI Worker] Inference error: {e}")

    def start(self, effect="passthrough", semitones=0.0):
        if sd is None: return False
        
        self.current_effect = effect
        self.semitones = semitones
        self.stop_event.clear()
        self.audio_queue = queue.Queue()
        self.ai_in_queue = queue.Queue(maxsize=4)
        with self.ai_out_lock:
            self.ai_out_queue_samples = []
        self.ai_buffer.clear()
        self._last_ai_overlap = None

        # Start AI inference worker first (so output thread always has a consumer)
        if effect == "ai":
            self.ai_worker_thread = threading.Thread(target=self._ai_inference_worker, daemon=True)
            self.ai_worker_thread.start()

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
        if self.ai_worker_thread:
            self.ai_worker_thread.join(timeout=2.0)
            self.ai_worker_thread = None

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
                    pth_models = [f for f in os.listdir("models") if f.endswith(".pth") and not f.startswith("D_") and not f.startswith("G_")]
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
