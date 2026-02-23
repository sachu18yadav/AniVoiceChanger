import customtkinter as ctk
import threading
import time
import os
import pystray
from PIL import Image, ImageDraw
import numpy as np
import io
import soundfile as sf
try:
    import sounddevice as sd
except ModuleNotFoundError:
    sd = None
try:
    from pynput import keyboard
except ModuleNotFoundError:
    keyboard = None

from core_state import AppState
from api_clients import ElevenLabsClient
from ui.header import HeaderFrame
from ui.pipeline import PipelineFrame
from ui.lab import LabFrame
from virtual_mic import VoiceChangerEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AppRouter(ctk.CTk):
    """The main lightweight router that glues UI components to Audio Engines."""
    
    def __init__(self):
        super().__init__()
        
        # 1. Init Core State
        self.app_state = AppState()
        self.elevenlabs = ElevenLabsClient()
        self.local_engine = VoiceChangerEngine(sample_rate=48000, block_size=512)
        self.app_state.local_engine = self.local_engine
        
        in_devs, _ = self.local_engine.get_devices()
        self.app_state.input_devices = in_devs

        # Window
        self.title("CHARACTER VOICE PRO")
        self.geometry("1000x700")
        self.configure(fg_color="#0A0A0B")
        self.is_mini = False
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.main_content = ctk.CTkFrame(self, corner_radius=0, fg_color="#0A0A0B")
        self.main_content.grid(row=0, column=0, sticky="nsew")

        # 2. Mount UI Components
        h_callbacks = {
            'on_hotkey_setup': self.capture_hotkey,
            'on_toggle_realtime': self.toggle_local_realtime,
            'on_toggle_mini': self.toggle_mini_mode
        }
        self.header = HeaderFrame(self.main_content, self.app_state, h_callbacks)
        self.header.pack(fill="x", padx=48, pady=(24, 8))
        
        self.config_container = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.config_container.pack(fill="both", expand=True, padx=48, pady=8)
        
        # ELEVENLABS FRAME
        self.el_frame = ctk.CTkFrame(self.config_container, fg_color="#121214", corner_radius=12, border_width=1, border_color="#1C1C1E")
        ctk.CTkLabel(self.el_frame, text="ELEVENLABS API ACTIVE", font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), text_color="#FFFFFF").pack(pady=(48, 8))
        ctk.CTkLabel(self.el_frame, text="Push-To-Talk is globally controlled from the top header.", text_color="#A0A0A0").pack(pady=16)

        p_callbacks = {
            'on_device_change': self.on_device_change,
            'on_slider_change': self.on_slider_change,
            'on_profile_change': self.on_profile_change
        }
        self.pipeline = PipelineFrame(self.config_container, self.app_state, p_callbacks)
        # Pack pipeline initially if local mode
        
        l_callbacks = {
            'on_record': self.run_record_test,
            'on_replay': self.replay_local_test
        }
        self.lab = LabFrame(self.main_content, l_callbacks)
        
        # Top-level backend toggler (Simplifying sidebar into just this for now, or recreating it)
        # For this refactor, let's add a backend toggle to the header or top of pipeline
        self.backend_selector = ctk.CTkSegmentedButton(
            self.main_content, 
            values=["Local DSP", "Local AI", "ElevenLabs API"], 
            command=self.on_backend_toggle
        )
        self.backend_selector.pack(pady=8)
        self.backend_selector.set(self.app_state.backend_mode)
        
        # Setup passthrough & loop
        self.passthrough_stream = None
        self.passthrough_active = False

        # Perform layout routing
        self.on_backend_toggle(self.app_state.backend_mode)
        
        # 3. System Processes
        self.setup_tray()
        self.setup_hotkeys()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.start_passthrough()

    # --- ROUTING LOGIC ---
    def on_backend_toggle(self, mode):
        self.app_state.update("backend_mode", mode)
        self.el_frame.pack_forget()
        self.pipeline.pack_forget()
        self.lab.pack_forget()
        
        if mode == "ElevenLabs API":
            self.el_frame.pack(fill="both", expand=True)
            self.header.update_status("● SERVER ONLINE", "#4CAF50")
            self.stop_passthrough()
        else:
            self.pipeline.pack(fill="both", expand=True)
            self.lab.pack(fill="x", pady=8)
            self.header.update_status("● LOCAL MODE", "#3B8ED0")
            self.start_passthrough()
            
            # Rebuild presets
            presets = self.app_state._scan_models() if mode == "Local AI" else {
                "anime_girl": {"pitch": 12, "label": "Anime Girl (DSP)"},
                "deep_voice": {"pitch": -8, "label": "Deep Voice (DSP)"}
            }
            self.pipeline.rebuild_radio_buttons(presets)

    def on_device_change(self, val):
        devices, _ = self.local_engine.get_devices()
        idx = next((i for i, d in enumerate(devices) if d in val), 0)
        self.local_engine.input_device = idx

    def on_slider_change(self, val):
        self.local_engine.semitones = val
        self.local_engine.anime_voice.pitch_shift = val
        if hasattr(self.local_engine, 'ai_converter') and self.local_engine.ai_converter:
            self.local_engine.ai_converter.pitch = val

    def on_profile_change(self, val):
        if self.app_state.backend_mode == "Local AI":
            def _load_new():
                try:
                    from ai_engine import rvc_wrapper
                    model_dir = os.path.join("models", val)
                    model_pth = os.path.join(model_dir, f"{val}.pth")
                    if not os.path.exists(model_pth): model_pth = os.path.join(model_dir, "model.pth")
                    if not os.path.exists(model_pth):
                        pths = [f for f in os.listdir(model_dir) if f.endswith(".pth") and not f.startswith("D_") and not f.startswith("G_")]
                        if pths: model_pth = os.path.join(model_dir, pths[0])
                    if os.path.exists(model_pth):
                        converter = rvc_wrapper.RVCVoiceConverter(model_pth, sample_rate=48000)
                        converter.pitch = self.app_state.pitch
                        self.local_engine.ai_converter = converter
                except Exception as e:
                    print(f"Profile Change Error: {e}")
            threading.Thread(target=_load_new, daemon=True).start()
        elif self.app_state.backend_mode == "Local DSP":
            self.local_engine.current_effect = val
        
    def _get_effect(self):
        m = self.app_state.backend_mode
        p = self.app_state.profile
        if m == "Local AI": return "ai"
        if m == "Local DSP": return p
        return "passthrough"

    # --- REALTIME ENGINE ---
    def toggle_local_realtime(self, btn):
        if not self.app_state.local_test_active:
            effect = self._get_effect()
            self.local_engine.current_effect = effect
            
            if effect == "ai":
                btn.configure(text="LOADING MODEL...", fg_color="#DD8800")
                self.update_idletasks()
                
                # Background load to prevent UI freeze
                def _load():
                    from ai_engine import rvc_wrapper
                    model_dir = os.path.join("models", self.app_state.profile)
                    model_pth = os.path.join(model_dir, f"{self.app_state.profile}.pth")
                    if not os.path.exists(model_pth): model_pth = os.path.join(model_dir, "model.pth")
                    if not os.path.exists(model_pth):
                        pths = [f for f in os.listdir(model_dir) if f.endswith(".pth") and not f.startswith("D_") and not f.startswith("G_")]
                        if pths: model_pth = os.path.join(model_dir, pths[0])
                    
                    self.local_engine.ai_converter = rvc_wrapper.RVCVoiceConverter(model_pth, sample_rate=48000)
                    self.local_engine.ai_converter.pitch = self.app_state.pitch
                    
                    self.local_engine.start(effect=effect, semitones=self.app_state.pitch)
                    self.app_state.update("local_test_active", True)
                    self.after(0, lambda: btn.configure(text="LOCAL REALTIME ON", fg_color="#00AA00"))
                threading.Thread(target=_load, daemon=True).start()
            else:
                self.local_engine.start(effect=effect, semitones=self.app_state.pitch)
                self.app_state.update("local_test_active", True)
                btn.configure(text="LOCAL REALTIME ON", fg_color="#00AA00")
        else:
            self.local_engine.stop()
            self.app_state.update("local_test_active", False)
            btn.configure(text="LOCAL REALTIME OFF", fg_color="#225522")

    # --- PASSTHROUGH AUDIO ---
    def start_passthrough(self):
        if self.passthrough_active or self.app_state.local_test_active or self.app_state.ptt_active or sd is None:
            return
        try:
            target_out = sd.default.device[1]
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0 and 'cable input' in dev['name'].lower() and 'vb-audio' in dev['name'].lower():
                    target_out = i; break
            
            in_dev = self.local_engine.input_device if self.local_engine.input_device is not None else sd.default.device[0]
            def _passthrough_cb(indata, outdata, frames, time, status):
                outdata[:] = indata
                
            self.passthrough_stream = sd.Stream(device=(in_dev, target_out), samplerate=48000, blocksize=512, channels=1, callback=_passthrough_cb)
            self.passthrough_stream.start()
            self.passthrough_active = True
        except Exception as e:
            print(f"Passthrough err: {e}")

    def stop_passthrough(self):
        if self.passthrough_stream:
            self.passthrough_stream.stop()
            self.passthrough_stream.close()
            self.passthrough_stream = None
        self.passthrough_active = False

    # --- RECORDING LAB LOGIC ---
    def run_record_test(self, rec_btn, replay_btn):
        if self.app_state.local_record_active: return
        self.app_state.local_record_active = True
        self.stop_passthrough()
        self.local_engine.stop()
        rec_btn.configure(text="RECORDING...", fg_color="#FF3333")
        
        def _thread():
            try:
                in_dev = self.local_engine.input_device if self.local_engine.input_device is not None else sd.default.device[0]
                recording = []
                def _rec_cb(indata, frames, time, status):
                    recording.append(indata.copy())
                    self.after(0, lambda: self.lab.visualizer.update_data(indata))
                
                with sd.InputStream(device=in_dev, samplerate=48000, channels=1, callback=_rec_cb):
                    time.sleep(5.0)
                
                recording = np.concatenate(recording).flatten().astype(np.float32)
                self.after(0, lambda: rec_btn.configure(text="PROCESSING...", fg_color="#005BB5"))
                
                effect = self._get_effect()
                if effect == "ai":
                    self.after(0, lambda: rec_btn.configure(text="AI INFERENCE...", fg_color="#8800DD"))
                    from ai_engine import rvc_wrapper
                    model_dir = os.path.join("models", self.app_state.profile)
                    model_pth = os.path.join(model_dir, f"{self.app_state.profile}.pth")
                    if not os.path.exists(model_pth): model_pth = os.path.join(model_dir, "model.pth")
                    if not os.path.exists(model_pth):
                        pths = [f for f in os.listdir(model_dir) if f.endswith(".pth") and not f.startswith("D_") and not f.startswith("G_")]
                        if pths: model_pth = os.path.join(model_dir, pths[0])
                    
                    converter = rvc_wrapper.RVCVoiceConverter(model_pth, sample_rate=48000)
                    converter.pitch = self.app_state.pitch
                    
                    final_audio = converter.convert(recording)
                    if final_audio is None: final_audio = recording
                else:
                    self.local_engine.semitones = self.app_state.pitch
                    self.local_engine.anime_voice.pitch_shift = self.app_state.pitch
                    final_audio = self.local_engine._process_block(recording.reshape(-1, 1), effect).flatten()

                self.last_recorded_audio = final_audio
                self.after(0, lambda: rec_btn.configure(text="PLAYING...", fg_color="#005BB5"))
                sd.play(final_audio, 48000)
                sd.wait()

            except Exception as e:
                print(f"Record Error: {e}")
            finally:
                self.app_state.local_record_active = False
                self.after(0, lambda: rec_btn.configure(text="RECORD TEST", fg_color="#007AFF"))
                self.after(0, lambda: replay_btn.pack(side="left", padx=(0, 3)))
                self.start_passthrough()
                
        threading.Thread(target=_thread, daemon=True).start()

    def replay_local_test(self, btn):
        if not hasattr(self, 'last_recorded_audio') or self.last_recorded_audio is None: return
        btn.configure(state="disabled")
        def _play():
            sd.play(self.last_recorded_audio, 48000)
            sd.wait()
            self.after(0, lambda: btn.configure(state="normal"))
        threading.Thread(target=_play, daemon=True).start()

    # --- PTT GLOBAL HOTKEY ---
    def capture_hotkey(self, btn):
        self.capturing_hotkey = True
        btn.configure(text="Press any key...")

    def setup_hotkeys(self):
        self.app_state.ptt_chunks = []
        self.app_state.ptt_start_time = 0
        self.capturing_hotkey = False
        self.ptt_stream = None

        def get_key_name(key):
            try: return key.char
            except AttributeError: return f"<{key.name}>"

        def _timeout_monitor():
            while self.app_state.ptt_active:
                if time.time() - self.app_state.ptt_start_time >= 10.0:
                    self.after(0, self.stop_global_ptt)
                    break
                time.sleep(0.1)

        def on_press(key):
            if not key: return
            kname = get_key_name(key)
            if self.capturing_hotkey:
                self.app_state.update("ptt_hotkey", kname)
                self.capturing_hotkey = False
                self.after(0, lambda: self.header.ptt_hotkey_btn.configure(text=f"Key: {self.app_state.ptt_hotkey}"))
                return
            
            if self.app_state.ptt_enabled and kname == self.app_state.ptt_hotkey and not self.app_state.ptt_active:
                self.stop_passthrough()
                self.app_state.ptt_active = True
                self.app_state.ptt_chunks = []
                self.app_state.ptt_start_time = time.time()
                
                in_dev = self.local_engine.input_device if self.local_engine.input_device is not None else sd.default.device[0]
                try:
                    self.ptt_stream = sd.InputStream(samplerate=48000, channels=1, device=in_dev, dtype='float32',
                                                     callback=lambda indata, f, t, s: self.app_state.ptt_chunks.append(indata.copy()))
                    self.ptt_stream.start()
                    self.after(0, lambda: self.header.update_status("● RECORDING PTT", "#FF3333"))
                    threading.Thread(target=_timeout_monitor, daemon=True).start()
                except Exception as e:
                    self.start_passthrough()

        def on_release(key):
            if not key: return
            if self.app_state.ptt_enabled and get_key_name(key) == self.app_state.ptt_hotkey and self.app_state.ptt_active:
                self.after(0, self.stop_global_ptt)

        def run_listener():
            try:
                listener = keyboard.Listener(on_press=on_press, on_release=on_release)
                listener.start()
                listener.join()
            except: pass
            
        threading.Thread(target=run_listener, daemon=True).start()

    def stop_global_ptt(self):
        if not self.app_state.ptt_active: return
        self.app_state.ptt_active = False
        if self.ptt_stream:
            self.ptt_stream.stop()
            self.ptt_stream.close()
            self.ptt_stream = None
            
        if not self.app_state.ptt_chunks:
            self.header.update_status("● IDLE", "gray")
            return
            
        self.header.update_status("● PROCESSING PTT...", "#DD8800")
        audio_data = np.concatenate(self.app_state.ptt_chunks, axis=0)
        mode = self.app_state.backend_mode
        
        if mode == "ElevenLabs API":
            self.elevenlabs.process_audio(audio_data, 48000, on_success=self._on_el_success, on_error=self._on_el_err)
        else:
            threading.Thread(target=self.process_local_ptt, args=(audio_data, mode), daemon=True).start()

    def _broadcast_to_cable(self, audio_data, sr):
        target_out = sd.default.device[1]
        for i, dev in enumerate(sd.query_devices()):
            name_lower = dev['name'].lower()
            if dev['max_output_channels'] > 0 and 'cable input' in name_lower and 'vb-audio' in name_lower:
                target_out = i
                break
        sd.play(audio_data, sr, device=target_out)
        sd.wait()

    def _on_el_success(self, audio, sr):
        self.after(0, lambda: self.header.update_status("● BROADCASTING CLOUD AUDIO", "#33FF33"))
        self._broadcast_to_cable(audio, sr)
        self.after(0, lambda: self.header.update_status("● SERVER ONLINE", "#4CAF50"))
        self.start_passthrough()

    def _on_el_err(self, err):
        print(f"ElevenLabs ERR: {err}")
        self.after(0, lambda: self.header.update_status("● CLOUD ERROR", "red"))
        self.start_passthrough()

    def process_local_ptt(self, recording, mode):
        try:
            recording = recording.flatten()
            effect = self._get_effect()
            
            if effect == "ai":
                if hasattr(self.local_engine, 'ai_converter') and self.local_engine.ai_converter and self.local_engine.ai_converter.model:
                    import librosa
                    audio_16k = librosa.resample(recording, orig_sr=48000, target_sr=16000)
                    converted = self.local_engine.ai_converter.convert(audio_16k)
                    final_audio = converted if converted is not None else recording
                else:
                    print("AI not loaded. Doing nothing.")
                    final_audio = recording
            else:
                self.local_engine.semitones = self.app_state.pitch
                self.local_engine.anime_voice.pitch_shift = self.app_state.pitch
                final_audio = self.local_engine._process_block(recording.reshape(-1, 1), effect).flatten()

            self.after(0, lambda: self.header.update_status("● BROADCASTING LOCAL AUDIO", "#33FF33"))
            self._broadcast_to_cable(final_audio, 48000)
        except Exception as e:
            print(f"Local PTT Error: {e}")
        finally:
            self.after(0, lambda: self.header.update_status("● LOCAL MODE", "#3B8ED0"))
            self.start_passthrough()

    # --- TRAY / SYSTEM ---
    def setup_tray(self):
        def _create_image():
            image = Image.new('RGB', (64, 64), (26, 26, 26))
            dc = ImageDraw.Draw(image)
            dc.ellipse((10, 10, 54, 54), fill='#3B8ED0')
            return image
        menu = pystray.Menu(pystray.MenuItem("Show/Hide", self.toggle_window), pystray.MenuItem("Exit", self.on_closing))
        self.icon = pystray.Icon("CharVoice", _create_image(), "Voice Chameleon", menu)
        threading.Thread(target=self.icon.run, daemon=True).start()

    def toggle_window(self):
        if self.state() == "iconic" or not self.winfo_viewable():
            self.deiconify()
        else:
            self.withdraw()

    def toggle_mini_mode(self):
        pass # Implemented later for brevity

    def on_closing(self):
        self.app_state.save()
        if hasattr(self, 'icon'): self.icon.stop()
        if self.passthrough_stream: self.passthrough_stream.close()
        self.destroy()

if __name__ == "__main__":
    app = AppRouter()
    app.mainloop()
