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
        
        from effects import SoftGate
        self.soft_gate = SoftGate(threshold=self.gate_threshold)

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
        
        # New Feature: UVR Preprocessing
        self.use_uvr = False
        try:
            from ai_engine.uvr_wrapper import UVRPreprocessor
            self.uvr_preprocessor = UVRPreprocessor(sample_rate=self.sample_rate)
        except ImportError:
            self.uvr_preprocessor = None
            
        # Decoupled AI pipeline queues:
        self.ai_in_queue = queue.Queue(maxsize=4)
        self.ai_out_ring = np.zeros(sample_rate * 2, dtype=np.float32) # 2s ring buffer
        self.ai_out_read_ptr = 0
        self.ai_out_write_ptr = 0
        self.ai_out_lock = threading.Lock()

        # 28800 (600ms) window, step 24000 (500ms) = 100ms true overlap context
        self.ai_buffer = StreamBuffer(target_size=28800, step_size=24000)
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
        
        # 0. Pre-processing: DC offset removal and Soft Limiter (replaces hard * 3.0)
        from effects import dc_offset_remover, soft_limiter
        processed = dc_offset_remover(processed)
        processed = soft_limiter(processed, drive=2.5) # Increased drive for better presence
        
        # 1. Noise Gate (Early in chain)
        if self.use_noise_gate:
            processed = self.soft_gate.process(processed)

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
                    need = len(processed.flatten())
                    filled = (self.ai_out_write_ptr - self.ai_out_read_ptr) % len(self.ai_out_ring)
                    
                    # Log health periodically
                    if not hasattr(self, '_health_counter'): self._health_counter = 0
                    self._health_counter += 1
                    if self._health_counter % 50 == 0:
                        print(f"[Buffer Health] {filled} samples ({filled/self.sample_rate:.2f}s) available", file=sys.stderr)

                    # Starvation Recovery: if we were starving, wait for a 200ms buffer before resuming to prevent stuttering
                    min_resumption = int(self.sample_rate * 0.2)
                    if self._starving_frames > 0 and filled < min_resumption:
                        can_play = False
                    else:
                        can_play = filled >= need

                    if can_play:
                        # Extract from ring buffer
                        end_ptr = (self.ai_out_read_ptr + need) % len(self.ai_out_ring)
                        if end_ptr > self.ai_out_read_ptr:
                            samples = self.ai_out_ring[self.ai_out_read_ptr:end_ptr]
                        else:
                            samples = np.concatenate([self.ai_out_ring[self.ai_out_read_ptr:], self.ai_out_ring[:end_ptr]])
                        
                        self.ai_out_read_ptr = end_ptr
                        processed = samples.reshape(-1, 1)
                        
                        # Update history for starvation recovery
                        self._last_played_audio = samples.copy()
                        self._starving_frames = 0
                    else:
                        # Starvation! CPU is slower than realtime.
                        fallback = np.zeros_like(processed.flatten(), dtype=np.float32)
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
        # Professional Overlap-Add setup
        target_len = self.ai_buffer.target_size
        step_len = self.ai_buffer.step_size
        xfade_len = target_len - step_len # 100ms (4800 samples)
        
        # Hann window for smooth crossfading
        hann_window = np.hanning(xfade_len * 2)
        fade_out_win = hann_window[xfade_len:]
        fade_in_win = hann_window[:xfade_len]
        
        overlap_buffer = np.zeros(xfade_len, dtype=np.float32)
        
        def find_optimal_shift(prev_tail, current_head, search_range):
            """SOLA: Find optimal phase alignment using cross-correlation."""
            if len(prev_tail) < search_range or len(current_head) < search_range:
                return 0
            
            # Simple autocorrelation
            best_corr = -1
            best_shift = 0
            
            # We search for the best shift that aligns the waveforms
            # prev_tail is the 'reference'
            # current_head is the 'sliding' part
            # Reduced search range for real-time performance (e.g., 5ms @ 48k = 240 samples)
            L = min(search_range, len(prev_tail) // 2)
            for shift in range(L):
                # Calculate correlation coefficient for the overlapping parts
                # We can use a simpler 'mean absolute difference' for speed or dot product
                corr = np.dot(prev_tail[:L], current_head[shift:shift+L])
                if corr > best_corr:
                    best_corr = corr
                    best_shift = shift
            return best_shift

        while not self.stop_event.is_set():
            try:
                frame = self.ai_in_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                if self.use_uvr and self.uvr_preprocessor is not None:
                    frame = self.uvr_preprocessor.process(frame)
                    
                ai_output = self.ai_converter.convert(frame)
                if ai_output is not None:
                    ai_flat = ai_output.flatten()
                    
                    if len(ai_flat) >= target_len:
                        new_chunk = ai_flat[-target_len:].copy()
                        
                        # 1. SOLA Alignment
                        # We use a search range of 10ms (480 samples) to find the best phase match
                        shift = find_optimal_shift(overlap_buffer, new_chunk, 480)
                        
                        # 2. Synchronized Overlap-Add
                        # Blend the shifted new chunk with the previous tail
                        target_region = new_chunk[shift : shift + xfade_len]
                        target_region[:] = (target_region * fade_in_win) + (overlap_buffer * fade_out_win)
                        
                        # 3. Extract the 'step' we want to play, starting from the aligned head
                        payload = new_chunk[shift : shift + step_len].copy()
                        
                        # 4. Store the tail for the NEXT iteration's head
                        overlap_buffer = new_chunk[shift + step_len : shift + step_len + xfade_len].copy()
                        
                        # If we ran out of buffer in new_chunk, pad overlap_buffer
                        if len(overlap_buffer) < xfade_len:
                            overlap_buffer = np.pad(overlap_buffer, (0, xfade_len - len(overlap_buffer)))
                        
                        # Push to Ring Buffer using fast numpy slices
                        with self.ai_out_lock:
                            n = len(payload)
                            ring_len = len(self.ai_out_ring)
                            end_ptr = (self.ai_out_write_ptr + n) % ring_len
                            
                            if end_ptr > self.ai_out_write_ptr:
                                self.ai_out_ring[self.ai_out_write_ptr:end_ptr] = payload
                            else:
                                first_part = ring_len - self.ai_out_write_ptr
                                self.ai_out_ring[self.ai_out_write_ptr:] = payload[:first_part]
                                self.ai_out_ring[:end_ptr] = payload[first_part:]
                            
                            self.ai_out_write_ptr = end_ptr
                    else:
                        # Fallback for weird sizes
                        with self.ai_out_lock:
                            n = len(ai_flat)
                            ring_len = len(self.ai_out_ring)
                            end_ptr = (self.ai_out_write_ptr + n) % ring_len
                            if end_ptr > self.ai_out_write_ptr:
                                self.ai_out_ring[self.ai_out_write_ptr:end_ptr] = ai_flat
                            else:
                                first_part = ring_len - self.ai_out_write_ptr
                                self.ai_out_ring[self.ai_out_write_ptr:] = ai_flat[:first_part]
                                self.ai_out_ring[:end_ptr] = ai_flat[first_part:]
                            self.ai_out_write_ptr = end_ptr
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
            self.ai_out_ring.fill(0)
            self.ai_out_read_ptr = 0
            self.ai_out_write_ptr = 0
        self.ai_buffer.clear()

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
