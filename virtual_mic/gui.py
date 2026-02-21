import customtkinter as ctk
import threading
import time
import os
import webbrowser
from typing import Optional

from w_okada_client import WOkadaEngineClient, VoiceProfile, TSUKIYOMI, STANDARD_FEMALE, CHIHIRO, FOAMY
from virtual_mic import VoiceChangerEngine
from utils import SystemChecker, SessionManager
import pystray
from PIL import Image, ImageDraw
import numpy as np
import collections
try:
    from pynput import keyboard
except ImportError:
    keyboard = None
import requests
import io
import soundfile as sf

# Configure look and feel
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SetupDialog(ctk.CTkToplevel):
    def __init__(self, parent, report):
        super().__init__(parent)
        self.title("Character Voice")
        self.geometry("400x450")
        self.resizable(False, False)
        self.report = report
        
        # Make modal
        self.transient(parent)
        self.grab_set()

        self.label = ctk.CTkLabel(self, text="INCOMPLETE SETUP", font=ctk.CTkFont(size=20, weight="bold"))
        self.label.pack(pady=20)

        self.items_frame = ctk.CTkFrame(self)
        self.items_frame.pack(pady=10, padx=20, fill="both")

        self.add_check_row("Microphone", report["mic"])
        self.add_check_row("VB-Audio Cable", report["virtual_cable"], "https://vb-audio.com/Cable/")
        self.add_check_row("AI Models", report["models"])
        
        # GPU Support (Informational if on CPU)
        has_gpu = report["gpu"]["status"] == "available"
        self.add_check_row(f"GPU Acceleration ({report['gpu']['type']})", has_gpu, is_optional=True)

        self.msg = ctk.CTkLabel(self, text="Some features may not work correctly.", wraplength=300)
        self.msg.pack(pady=20)

        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=10, side="bottom", fill="x", padx=20)

        self.test_mode_btn = ctk.CTkButton(self.btn_frame, text="Continue in Test Mode", command=self.continue_test)
        self.test_mode_btn.pack(side="left", padx=5, expand=True)

        self.retry_btn = ctk.CTkButton(self.btn_frame, text="Retry", command=self.retry, fg_color="gray")
        self.retry_btn.pack(side="right", padx=5, expand=True)

    def add_check_row(self, name, success, link=None, is_optional=False):
        row = ctk.CTkFrame(self.items_frame, fg_color="transparent")
        row.pack(fill="x", pady=5, padx=10)
        
        if success:
            status_color = "green"
            status_text = "✓"
        elif is_optional:
            status_color = "orange"
            status_text = "⚠"
        else:
            status_color = "red"
            status_text = "✗"
        
        label = ctk.CTkLabel(row, text=f"{status_text} {name}", text_color=status_color, font=ctk.CTkFont(weight="bold"))
        label.pack(side="left")

        if not success and link:
            # Check for local driver pack provided by user
            local_path = os.path.join(os.getcwd(), "..", "VBCABLE_Driver_Pack45")
            if "Cable" in name and os.path.exists(local_path):
                btn = ctk.CTkButton(row, text="Open Driver Folder", width=120, height=20, font=ctk.CTkFont(size=10), command=lambda: os.startfile(os.path.abspath(local_path)))
                btn.pack(side="right")
            else:
                btn = ctk.CTkButton(row, text="Install", width=60, height=20, font=ctk.CTkFont(size=10), command=lambda: webbrowser.open(link))
                btn.pack(side="right")

    def continue_test(self):
        self.destroy()

    def retry(self):
        # In a real app, this would re-run checks
        self.destroy()

class WaveformVisualizer(ctk.CTkCanvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.configure(bg="#1A1A1A")
        self.data = np.zeros(100)
        self.draw_loop()

    def update_data(self, new_data):
        if len(new_data) > 0:
            # Simple RMS-based peak for visualization
            peak = np.abs(new_data).max()
            self.data = np.roll(self.data, -1)
            self.data[-1] = peak

    def draw_loop(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w > 1:
            step = w / len(self.data)
            for i in range(len(self.data) - 1):
                x1 = i * step
                y1 = h - (self.data[i] * h * 2) - 10
                x2 = (i + 1) * step
                y2 = h - (self.data[i+1] * h * 2) - 10
                self.create_line(x1, y1, x2, y2, fill="#3B8ED0", width=2)
        self.after(50, self.draw_loop)

class VoiceChangerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Config
        self.title("CHARACTER VOICE PRO")
        self.geometry("1000x700")
        self.configure(fg_color="#0A0A0B")
        self.is_mini = False
        self.last_geometry = "1000x700"

        # Engine & Session
        self.engine = WOkadaEngineClient()
        self.local_engine = VoiceChangerEngine(block_size=1024)
        self.session_data = SessionManager.load_session() or SessionManager.get_default_state()
        self.is_active = False
        self.backend_mode = ctk.StringVar(value=self.session_data.get("backend_mode", "Local DSP"))

        # Per-voice pitch presets (semitones)
        self.DSP_PRESETS = {
            "standard_female": {"pitch": 12, "label": "Standard Anime Female"},
            "tsukiyomi":       {"pitch": 22, "label": "Tsukiyomi-chan (Extreme)"},
        }
        self.RVC_PRESETS = {
            "chihiro": {"pitch": 18, "label": "Chihiro Fujisaki (AI)"},
            "egirl":   {"pitch": 20, "label": "E-Girl (AI)"},
            "foamy":   {"pitch": 15, "label": "Foamy the Squirrel (AI)"},
        }

        # Start checking w-okada status
        self.server_alive = False
        
        # Local Engine State
        self.local_test_active = False
        self.local_record_active = False

        # --- UI LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#121214", border_width=1, border_color="#1C1C1E")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="VOICE\nCHAMELEON", font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"), text_color="#FFFFFF")
        self.logo_label.pack(pady=40, padx=32)

        self.nav_btn_1 = ctk.CTkButton(self.sidebar, text="DASHBOARD", fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"))
        self.nav_btn_1.pack(pady=16, padx=32, fill="x")

        self.nav_btn_2 = ctk.CTkButton(self.sidebar, text="AI MODELS", fg_color="transparent", text_color="#A0A0A0", hover_color="#1C1C1E", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"))
        self.nav_btn_2.pack(pady=8, padx=32, fill="x")
        
        # Backend Selector in Sidebar
        ctk.CTkLabel(self.sidebar, text="PROCESSING ENGINE", font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"), text_color="#A0A0A0").pack(pady=(40, 8), padx=32, anchor="w")
        self.backend_selector = ctk.CTkOptionMenu(self.sidebar, values=["Local DSP", "Local AI", "ElevenLabs API"], variable=self.backend_mode, command=self.on_backend_toggle, fg_color="#1C1C1E", button_color="#2C2C2E", button_hover_color="#3A3A3C", font=ctk.CTkFont(family="Segoe UI", size=12))
        self.backend_selector.pack(pady=0, padx=32, fill="x")

        self.sidebar_info = ctk.CTkLabel(self.sidebar, text="v2.0.4 PREMIUM", font=ctk.CTkFont(family="Segoe UI", size=10), text_color="#404040")
        self.sidebar_info.pack(side="bottom", pady=32)

        # 2. Main Content
        self.main_content = ctk.CTkFrame(self, corner_radius=0, fg_color="#0A0A0B")
        self.main_content.grid(row=0, column=1, sticky="nsew")
        
        # Header Area
        self.header_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=48, pady=(48, 16))
        
        self.title_lbl = ctk.CTkLabel(self.header_frame, text="Active Modulation Center", font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), text_color="#FFFFFF")
        self.title_lbl.pack(side="left")

        self.ptt_enabled = ctk.BooleanVar(value=self.session_data.get("ptt_enabled", False))
        self.ptt_hotkey = self.session_data.get("ptt_hotkey", "<alt>")
        
        self.ptt_check = ctk.CTkCheckBox(self.header_frame, text="Global PTT", variable=self.ptt_enabled, font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), fg_color="#007AFF", hover_color="#005BB5")
        self.ptt_check.pack(side="left", padx=(32, 16))
        
        self.ptt_hotkey_btn = ctk.CTkButton(self.header_frame, text=f"Key: {self.ptt_hotkey}", width=80, height=28, corner_radius=6, fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#FFFFFF", command=self.capture_hotkey)
        self.ptt_hotkey_btn.pack(side="left")
        
        self.mini_toggle_btn = ctk.CTkButton(self.header_frame, text="MINI VIEW", width=80, height=28, corner_radius=6, fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#A0A0A0", command=self.toggle_mini_mode)
        self.mini_toggle_btn.pack(side="right")
        
        self.status_indicator = ctk.CTkLabel(self.header_frame, text="● SERVER CHECKING", text_color="#A0A0A0", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"))
        self.status_indicator.pack(side="right", padx=16)

        # Config Area (Dynamic)
        self.config_container = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.config_container.pack(fill="both", expand=True, padx=48, pady=16)
        
        # 1) ELEVENLABS FRAME
        self.el_frame = ctk.CTkFrame(self.config_container, fg_color="#121214", corner_radius=12, border_width=1, border_color="#1C1C1E")
        ctk.CTkLabel(self.el_frame, text="ELEVENLABS API ACTIVE", font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"), text_color="#FFFFFF").pack(pady=(48, 8))
        ctk.CTkLabel(self.el_frame, text="Push-To-Talk is globally controlled from the top header.\nAudio will be synthetically rendered in the Cloud and broadcasted to the Virtual Cable.", text_color="#A0A0A0", font=ctk.CTkFont(family="Segoe UI", size=12)).pack(pady=16)

        # 2) DSP / AI FRAME
        self.dsp_ai_frame = ctk.CTkFrame(self.config_container, fg_color="transparent")
        self.dsp_ai_frame.grid_columnconfigure((0,1), weight=1)

        # Left Column: Devices
        self.left_col = ctk.CTkFrame(self.dsp_ai_frame, fg_color="#121214", corner_radius=12, border_width=1, border_color="#1C1C1E")
        self.left_col.grid(row=0, column=0, padx=(0, 16), sticky="nsew")
        
        ctk.CTkLabel(self.left_col, text="AUDIO PIPELINE", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), text_color="#A0A0A0").pack(pady=(24, 16), padx=32, anchor="w")
        
        ctk.CTkLabel(self.left_col, text="Input Source", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#FFFFFF").pack(anchor="w", padx=32)
        
        # input device selector — auto-detect real mic (prefer Realtek, skip virtual cables)
        self.input_devices, _ = self.local_engine.get_devices()
        default_input = self.input_devices[0] if self.input_devices else "None"
        for dev_str in self.input_devices:
            name_lower = dev_str.lower()
            if 'realtek' in name_lower or 'microphone array' in name_lower:
                if 'cable' not in name_lower and 'steam' not in name_lower:
                    default_input = dev_str
                    break
        
        self.input_device_var = ctk.StringVar(value=default_input)
        self.input_selector = ctk.CTkOptionMenu(self.left_col, values=self.input_devices, variable=self.input_device_var, command=self.on_device_change, font=ctk.CTkFont(family="Segoe UI", size=11), fg_color="#1C1C1E", button_color="#2C2C2E", button_hover_color="#3A3A3C")
        self.input_selector.pack(pady=(8, 24), padx=32, fill="x")
        # Apply the auto-detected device immediately
        self.on_device_change(default_input)

        ctk.CTkLabel(self.left_col, text="Routing Info", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#FFFFFF").pack(anchor="w", padx=32)
        self.routing_info = ctk.CTkLabel(self.left_col, text="Output: VB-Cable Input", justify="left", font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#A0A0A0")
        self.routing_info.pack(pady=(8, 24), padx=32, anchor="w")

        # Right Column: Voice
        self.right_col = ctk.CTkFrame(self.dsp_ai_frame, fg_color="#121214", corner_radius=12, border_width=1, border_color="#1C1C1E")
        self.right_col.grid(row=0, column=1, padx=(16, 0), sticky="nsew")
        
        ctk.CTkLabel(self.right_col, text="VOICE PROFILE", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), text_color="#A0A0A0").pack(pady=(24, 16), padx=32, anchor="w")
        
        self.profile_var = ctk.StringVar(value=self.session_data.get("last_mode", "standard_female"))
        self.radio_buttons = []

        self.radios_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        self.radios_frame.pack(fill="x", padx=32)

        self.pitch_slider = ctk.CTkSlider(self.right_col, from_=-12, to=24, number_of_steps=36, fg_color="#1C1C1E", progress_color="#007AFF", button_color="#FFFFFF", button_hover_color="#E0E0E0", command=self.on_slider_change)
        self.pitch_slider.set(self.session_data.get("last_pitch", 12))
        self.pitch_slider.pack(pady=(32, 8), padx=32, fill="x")
        self.pitch_lbl = ctk.CTkLabel(self.right_col, text=f"Pitch Offset: {self.pitch_slider.get():.1f} ST", font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#A0A0A0")
        self.pitch_lbl.pack(pady=(0, 24))

        # --- TESTING LABORATORY (LAYER 1 & 2) ---
        self.lab_frame = ctk.CTkFrame(self.main_content, fg_color="#121214", corner_radius=12, border_width=1, border_color="#1C1C1E")
        self.lab_frame.pack(fill="x", padx=48, pady=16)
        
        ctk.CTkLabel(self.lab_frame, text="TESTING LABORATORY", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), text_color="#A0A0A0").pack(pady=(24, 8), padx=32, anchor="w")
        
        self.lab_btns = ctk.CTkFrame(self.lab_frame, fg_color="transparent")
        self.lab_btns.pack(fill="x", padx=32, pady=(0, 24))
        
        self.open_wk_ui_btn = ctk.CTkButton(self.lab_btns, text="OPEN BROWSER UI", width=160, height=36, corner_radius=6, fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), command=lambda: webbrowser.open("http://127.0.0.1:18888"))
        self.open_wk_ui_btn.pack(side="left", padx=8)

        # Record + Replay group
        self.rec_group = ctk.CTkFrame(self.lab_btns, fg_color="transparent")
        self.rec_group.pack(side="left", padx=8)
        
        self.local_rec_btn = ctk.CTkButton(self.rec_group, text="RECORD TEST", width=120, height=36, corner_radius=6, fg_color="#007AFF", hover_color="#005BB5", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#FFFFFF", command=self.toggle_local_record)
        self.local_rec_btn.pack(side="left", padx=(0, 8))
        
        self.local_replay_btn = ctk.CTkButton(self.rec_group, text="▶ REPLAY", width=80, height=36, corner_radius=6, fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), command=self.replay_local_test)
        # Hidden initially — pack it only after first recording
        
        self.local_realtime_btn = ctk.CTkButton(self.lab_btns, text="LOCAL REALTIME OFF", width=140, height=36, corner_radius=6, fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), command=self.toggle_local_realtime)
        self.local_realtime_btn.pack(side="left", padx=8)

        # 3. Control Center (Bottom)
        self.control_panel = ctk.CTkFrame(self.main_content, height=100, fg_color="transparent", corner_radius=0)
        self.control_panel.pack(fill="x", padx=48, pady=32)
        
        self.sync_btn = ctk.CTkButton(self.control_panel, text="APPLY CONFIGURATION", font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), width=220, height=48, corner_radius=12, fg_color="#007AFF", hover_color="#005BB5", text_color="#FFFFFF", command=self.apply_profile)
        self.sync_btn.pack(side="right", padx=0, pady=0)
        
        self.live_monitor_lbl = ctk.CTkLabel(self.control_panel, text="HOT SWAP READY", font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#A0A0A0")
        self.live_monitor_lbl.pack(side="right", padx=24)

        # Performance Stats
        self.perf_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
        self.perf_frame.pack(fill="x", padx=48, pady=(0, 24))
        self.cpu_lbl = ctk.CTkLabel(self.perf_frame, text="CPU: 0%", font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"), text_color="#606060")
        self.cpu_lbl.pack(side="left")
        self.lat_lbl = ctk.CTkLabel(self.perf_frame, text="LATENCY: 0ms", font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"), text_color="#606060")
        self.lat_lbl.pack(side="left", padx=32)

        # Last recorded audio for replay
        self.last_recorded_audio = None
        self.sample_rate = 48000

        # Initialization
        self.after(100, self.perform_startup_check)
        self.start_perf_loop()
        self.icon = None
        self.setup_tray()
        if keyboard: self.setup_hotkeys()
        
        self.on_backend_toggle(self.backend_mode.get())

    def _get_selected_input_device(self):
        """Get the currently selected input device ID."""
        try:
            return int(self.input_device_var.get().split(":")[0])
        except:
            return sd.default.device[0]

    def _get_effect(self):
        """Map selected profile to effect string."""
        prof = self.profile_var.get()
        effect_map = {"tsukiyomi": "anime_girl", "standard_female": "anime_girl"}
        return effect_map.get(prof, "passthrough")

    def rebuild_radio_buttons(self, backend_mode):
        for rb in self.radio_buttons:
            rb.destroy()
        self.radio_buttons.clear()

        presets = self.RVC_PRESETS if backend_mode == "Local AI" else self.DSP_PRESETS
        
        # Ensure selected profile exists in new dict, else default to first
        if self.profile_var.get() not in presets:
            self.profile_var.set(list(presets.keys())[0])

        for key, info in presets.items():
            rb = ctk.CTkRadioButton(self.radios_frame, text=info["label"], variable=self.profile_var, value=key, font=ctk.CTkFont(size=11), command=self.on_profile_change)
            rb.pack(pady=8, anchor="w")
            self.radio_buttons.append(rb)
        
        self.on_profile_change()

    def on_backend_toggle(self, mode):
        # Hide all first
        self.dsp_ai_frame.pack_forget()
        self.el_frame.pack_forget()
        self.lab_frame.pack_forget()

        if mode == "ElevenLabs API":
            self.el_frame.pack(fill="both", expand=True)
            self.lab_frame.pack(fill="x", padx=30, pady=10)
        else:
            self.dsp_ai_frame.pack(fill="both", expand=True)
            self.rebuild_radio_buttons(mode)
            self.lab_frame.pack(fill="x", padx=30, pady=10)

    def on_slider_change(self, val):
        self.pitch_lbl.configure(text=f"Pitch: {val:.0f} ST")
        self.local_engine.semitones = val
        self.local_engine.anime_voice.pitch_shift = val
        if self.server_alive:
            self.engine.update_settings("tran", int(val))

    def on_profile_change(self):
        """Auto-adjust pitch slider when profile changes."""
        profile_key = self.profile_var.get()
        backend = self.backend_mode.get()
        presets = self.RVC_PRESETS if backend == "Local AI" else self.DSP_PRESETS
        preset = presets.get(profile_key, {"pitch": 12})
        self.pitch_slider.set(preset["pitch"])
        self.on_slider_change(preset["pitch"])
        print(f"Profile: {profile_key} → pitch +{preset['pitch']} ST")

    def on_device_change(self, val):
        try:
            device_id = int(val.split(":")[0])
            self.local_engine.input_device = device_id
            print(f"Input device changed to: {val}")
        except:
            pass

    def toggle_local_record(self):
        if not self.local_record_active:
            self.local_record_active = True
            self.local_rec_btn.configure(text="RECORDING...", fg_color="#AA0000")
            threading.Thread(target=self._run_record_thread, daemon=True).start()

    def _run_record_thread(self):
        """Simple flow: record 5s → process whole phrase → play. Same as test_basic_sample.py."""
        import sounddevice as sd
        try:
            sr = self.sample_rate
            duration = 5.0
            in_dev = self._get_selected_input_device()
            effect = self._get_effect()

            print(f"Recording {duration}s from device {in_dev}...")
            
            # 1. Record — simple blocking call
            recording = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32', device=in_dev)
            sd.wait()
            recording = recording.flatten()

            max_vol = np.max(np.abs(recording))
            print(f"Recording done. {len(recording)} samples, max vol: {max_vol:.4f}")
            
            if max_vol < 0.001:
                print("WARNING: Recording is silent! Check your mic selection.")
            
            self.after(0, lambda: self.local_rec_btn.configure(text="PROCESSING...", fg_color="#005BB5"))
            
            mode = self.backend_mode.get()
            profile = self.profile_var.get()
            pitch = self.pitch_slider.get()
            
            if mode == "Local DSP":
                self.local_engine.semitones = pitch
                self.local_engine.anime_voice.pitch_shift = pitch
                block = recording.reshape(-1, 1)
                final_audio = self.local_engine._process_block(block, "anime_girl").flatten()
                prof_name = self.DSP_PRESETS.get(profile, {}).get("label", profile)
                print(f"DSP Processed as {prof_name} (+{pitch:.0f} ST). Playing back...")

            elif mode == "Local AI":
                # We dynamically route this to the AniVoiceChanger pipeline to prevent OOM
                self.after(0, lambda: self.local_rec_btn.configure(text="AI INFERENCE...", fg_color="#8800DD"))
                import sys, os
                rvc_path = os.path.abspath('../AniVoiceChanger')
                if rvc_path not in sys.path: sys.path.append(rvc_path)
                import rvc_infer
                
                model_pth = os.path.join("models", profile, f"{profile}.pth")
                if not os.path.exists(model_pth):
                    model_pth = os.path.join("models", profile, "model.pth")
                
                model, info = rvc_infer.load_rvc_model(model_pth)
                if model:
                    index_path = os.path.join("models", profile, f"{profile}.index")
                    idx, big_npy = rvc_infer.load_index(index_path) if os.path.exists(index_path) else (None, None)
                    converted, out_sr = rvc_infer.infer(recording, sr, model, info, f0_up_key=pitch, index=idx, big_npy=big_npy)
                    
                    if out_sr != self.sample_rate:
                        import librosa
                        final_audio = librosa.resample(converted, orig_sr=out_sr, target_sr=self.sample_rate)
                    else:
                        final_audio = converted
                        
                    prof_name = self.RVC_PRESETS.get(profile, {}).get("label", profile)
                    print(f"RVC Processed as {prof_name} (+{pitch:.0f} ST). Playing back...")
                else:
                    print("Failed to load RVC Model. Falling back to clean audio.")
                    final_audio = recording
            else:
                final_audio = recording

            self.last_recorded_audio = final_audio
            
            self.after(0, lambda: self.local_rec_btn.configure(text="PLAYING...", fg_color="#005BB5"))
            
            # 3. Play to default speakers
            sd.play(final_audio, sr)
            sd.wait()
            
        except Exception as e:
            print(f"Record test error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.after(0, self._finish_record_ui)

    def _finish_record_ui(self):
        self.local_rec_btn.configure(text="RECORD TEST", fg_color="#007AFF")
        self.local_record_active = False
        # Show replay button next to record button
        if not self.local_replay_btn.winfo_ismapped():
            self.local_replay_btn.pack(side="left", padx=(0, 3))

    def replay_local_test(self):
        if self.last_recorded_audio is None:
            return
        self.local_replay_btn.configure(state="disabled")
        def _play():
            import sounddevice as sd
            sd.play(self.last_recorded_audio, self.sample_rate)
            sd.wait()
            self.after(0, lambda: self.local_replay_btn.configure(state="normal"))
        threading.Thread(target=_play, daemon=True).start()

    def toggle_local_realtime(self):
        if not self.local_test_active:
            # Start realtime loop
            effect = self._get_effect()
            self.local_engine.current_effect = effect
            self.local_engine.start()
            self.local_test_active = True
            self.local_realtime_btn.configure(text="LOCAL REALTIME ON", fg_color="#00AA00")
        else:
            # Stop
            self.local_engine.stop()
            self.local_test_active = False
            self.local_realtime_btn.configure(text="LOCAL REALTIME OFF", fg_color="#225522")

    def perform_startup_check(self):
        # We don't check devices directly anymore, we check w-okada
        pass

    def apply_profile(self):
        self.sync_btn.configure(text="SAVING...", fg_color="#005BB5")
        state = {
            "backend_mode": self.backend_mode.get(),
            "last_mode": self.profile_var.get(),
            "last_pitch": self.pitch_slider.get(),
            "ptt_enabled": self.ptt_enabled.get(),
            "ptt_hotkey": self.ptt_hotkey
        }
        SessionManager.save_session(state)
        self.after(500, lambda: self.sync_btn.configure(text="SAVED ✓", fg_color="#005BB5"))
        self.after(2000, lambda: self.sync_btn.configure(text="APPLY CONFIGURATION", fg_color="#007AFF"))

    def toggle_mini_mode(self):
        self.is_mini = not self.is_mini
        if self.is_mini:
            self.last_geometry = self.geometry()
            self.sidebar.grid_forget()
            self.header_frame.pack_forget()
            self.visualizer.pack_forget() if hasattr(self, 'visualizer') else None
            self.config_container.pack_forget()
            self.lab_frame.pack_forget()
            self.perf_frame.pack_forget()
            
            self.geometry("300x180")
            self.main_content.configure(corner_radius=0)
            self.main_content.pack(fill="both", expand=True)
            
            # Simple Mini View: Logo + Power + Profile
            self.mini_frame = ctk.CTkFrame(self.main_content, fg_color="transparent")
            self.mini_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            self.mini_logo = ctk.CTkLabel(self.mini_frame, text="CHAR VOICE MINI", font=ctk.CTkFont(size=14, weight="bold"), text_color="#3B8ED0")
            self.mini_logo.pack()
            
            self.mini_profile = ctk.CTkLabel(self.mini_frame, text=f"Mode: {self.backend_mode.get()}", font=ctk.CTkFont(size=11))
            self.mini_profile.pack(pady=10)
            
            self.mini_restore = ctk.CTkButton(self.mini_frame, text="FULL VIEW", width=80, height=20, font=ctk.CTkFont(size=9), fg_color="#333333", command=self.toggle_mini_mode)
            self.mini_restore.pack(side="bottom")
        else:
            if hasattr(self, 'mini_frame'):
                self.mini_frame.destroy()
            self.sidebar.grid(row=0, column=0, sticky="nsew")
            self.main_content.pack_forget()
            self.main_content.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
            
            self.header_frame.pack(fill="x", padx=30, pady=(30, 10))
            self.config_container.pack(fill="both", expand=True, padx=30, pady=5)
            # Re-trigger routing since packs were forgotten
            self.on_backend_toggle(self.backend_mode.get())
            
            self.control_panel.pack(fill="x", padx=30, pady=20)
            self.perf_frame.pack(fill="x", padx=40, pady=(0, 20))
            self.geometry(self.last_geometry)

    def start_perf_loop(self):
        # Poll w-okada server — suppress connection errors
        try:
            self.server_alive = self.engine.is_alive()
        except:
            self.server_alive = False
        
        if self.server_alive:
            self.status_indicator.configure(text="● SERVER ONLINE", text_color="#4CAF50")
            try:
                info = self.engine.get_info()
                self.cpu_lbl.configure(text=f"W-OKADA ACTIVE")
                self.lat_lbl.configure(text=f"ROUTING VIA VB-CABLE")
            except:
                pass
        else:
            self.status_indicator.configure(text="● LOCAL DSP MODE", text_color="#3B8ED0")
            
        self.after(5000, self.start_perf_loop)

    def on_closing(self):
        self.apply_profile()
        if self.icon: self.icon.stop()
        self.destroy()
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

    def setup_hotkeys(self):
        def on_activate(): self.after(0, self.apply_profile)
        
        self.ptt_active = False
        self.ptt_stream = None
        self.ptt_chunks = []
        self.capturing_hotkey = False
        self.ptt_start_time = 0

        def get_key_name(key):
            try: return key.char
            except AttributeError: return f"<{key.name}>"

        def ptt_timeout_monitor():
            while self.ptt_active:
                if time.time() - self.ptt_start_time >= 10.0:
                    self.after(0, self.stop_global_ptt)
                    break
                time.sleep(0.1)

        def on_press(key):
            if not key: return
            kname = get_key_name(key)
            if self.capturing_hotkey:
                self.ptt_hotkey = kname
                self.capturing_hotkey = False
                self.after(0, lambda: self.ptt_hotkey_btn.configure(text=f"Key: {self.ptt_hotkey}"))
                return
            
            if self.ptt_enabled.get() and kname == self.ptt_hotkey and not self.ptt_active:
                self.ptt_active = True
                self.ptt_chunks = []
                self.ptt_start_time = time.time()
                in_dev = self._get_selected_input_device()
                try:
                    import sounddevice as sd
                    self.ptt_stream = sd.InputStream(samplerate=self.sample_rate, channels=1, device=in_dev, dtype='float32',
                                                     callback=lambda indata, frames, time_info, status: self.ptt_chunks.append(indata.copy()))
                    self.ptt_stream.start()
                    self.after(0, lambda: self.status_indicator.configure(text="● RECORDING PTT", text_color="#FF3333"))
                    threading.Thread(target=ptt_timeout_monitor, daemon=True).start()
                except Exception as e:
                    print(f"PTT Start Error: {e}")

        def on_release(key):
            if not key: return
            kname = get_key_name(key)
            if self.ptt_enabled.get() and kname == self.ptt_hotkey and self.ptt_active:
                self.after(0, self.stop_global_ptt)

        def run_listener():
            try:
                listener = keyboard.Listener(on_press=on_press, on_release=on_release)
                listener.start()
                with keyboard.GlobalHotKeys({'<ctrl>+<shift>+t': on_activate}) as h:
                    h.join()
            except: pass
            
        threading.Thread(target=run_listener, daemon=True).start()

    def capture_hotkey(self):
        self.capturing_hotkey = True
        self.ptt_hotkey_btn.configure(text="Press any key...")

    def stop_global_ptt(self):
        if not self.ptt_active: return
        self.ptt_active = False
        if self.ptt_stream:
            self.ptt_stream.stop()
            self.ptt_stream.close()
            self.ptt_stream = None
        
        if not self.ptt_chunks:
            self.status_indicator.configure(text="● IDLE", text_color="gray")
            return
            
        self.status_indicator.configure(text="● PROCESSING GLOBAL PTT...", text_color="#DD8800")
        audio_data = np.concatenate(self.ptt_chunks, axis=0)
        mode = self.backend_mode.get()
        
        if mode == "ElevenLabs API":
            threading.Thread(target=self.process_elevenlabs_ptt, args=(audio_data,), daemon=True).start()
        else:
            threading.Thread(target=self.process_local_ptt, args=(audio_data, mode), daemon=True).start()
            
    def _broadcast_to_cable(self, audio_data, sr):
        import sounddevice as sd
        devices = sd.query_devices()
        target_out_dev = sd.default.device[1] # fallback to speakers just in case
        for i, dev in enumerate(devices):
            name_lower = dev['name'].lower()
            if dev['max_output_channels'] > 0 and 'cable input' in name_lower and 'vb-audio' in name_lower:
                target_out_dev = i
                break
        sd.play(audio_data, sr, device=target_out_dev)
        sd.wait()

    def process_elevenlabs_ptt(self, audio_data):
        import sounddevice as sd
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_data, self.sample_rate, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        
        api_key = "sk_cf8e0e9100def3309702b6d55050aca7366a56038b150a86"
        voice_id = "uCNfGgx20cVUPpqSApMp"
        url = f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}"
        headers = {"xi-api-key": api_key}
        data = {"model_id": "eleven_multilingual_sts_v2", "voice_settings": '{"stability": 0.7, "similarity_boost": 0.75}'}
        files = {"audio": ("audio.wav", wav_io, "audio/wav")}
        
        try:
            resp = requests.post(url, headers=headers, data=data, files=files)
            if resp.status_code == 200:
                self.after(0, lambda: self.status_indicator.configure(text="● BROADCASTING CLOUD AUDIO", text_color="#33FF33"))
                out_io = io.BytesIO(resp.content)
                out_data, out_sr = sf.read(out_io)
                self._broadcast_to_cable(out_data, out_sr)
            else:
                self.after(0, lambda: self.status_indicator.configure(text=f"● API ERR {resp.status_code}", text_color="red"))
        except:
            self.after(0, lambda: self.status_indicator.configure(text="● CLOUD PTT FAILED", text_color="red"))
        finally:
            self.after(0, lambda: self.status_indicator.configure(text="● SERVER ONLINE" if self.server_alive else "● LOCAL MODE", text_color="#4CAF50" if self.server_alive else "#3B8ED0"))

    def process_local_ptt(self, recording, mode):
        try:
            recording = recording.flatten()
            profile = self.profile_var.get()
            pitch = self.pitch_slider.get()
            sr = self.sample_rate
            
            if mode == "Local DSP":
                self.local_engine.semitones = pitch
                self.local_engine.anime_voice.pitch_shift = pitch
                block = recording.reshape(-1, 1)
                final_audio = self.local_engine._process_block(block, "anime_girl").flatten()
            elif mode == "Local AI":
                import sys, os
                rvc_path = os.path.abspath('../AniVoiceChanger')
                if rvc_path not in sys.path: sys.path.append(rvc_path)
                import rvc_infer
                
                model_pth = os.path.join("models", profile, f"{profile}.pth")
                if not os.path.exists(model_pth):
                    model_pth = os.path.join("models", profile, "model.pth")
                
                model, info = rvc_infer.load_rvc_model(model_pth)
                if model:
                    index_path = os.path.join("models", profile, f"{profile}.index")
                    idx, big_npy = rvc_infer.load_index(index_path) if os.path.exists(index_path) else (None, None)
                    converted, out_sr = rvc_infer.infer(recording, sr, model, info, f0_up_key=pitch, index=idx, big_npy=big_npy)
                    
                    if out_sr != self.sample_rate:
                        import librosa
                        final_audio = librosa.resample(converted, orig_sr=out_sr, target_sr=self.sample_rate)
                    else:
                        final_audio = converted
                else:
                    final_audio = recording
            
            self.after(0, lambda: self.status_indicator.configure(text="● BROADCASTING LOCAL AUDIO", text_color="#33FF33"))
            self._broadcast_to_cable(final_audio, sr)
        except Exception as e:
            print(f"Local PTT Error: {e}")
        finally:
            self.after(0, lambda: self.status_indicator.configure(text="● SERVER ONLINE" if self.server_alive else "● LOCAL MODE", text_color="#4CAF50" if self.server_alive else "#3B8ED0"))

if __name__ == "__main__":
    app = VoiceChangerApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
