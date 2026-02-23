import customtkinter as ctk

class HeaderFrame(ctk.CTkFrame):
    """The top navigation ribbon containing global push-to-talk, real-time toggles, and status indicators."""
    
    def __init__(self, master, app_state, callbacks):
        super().__init__(master, fg_color="transparent")
        self.app_state = app_state
        self.callbacks = callbacks # dict of functions like 'on_hotkey_setup', 'on_toggle_realtime', 'on_toggle_mini'
        
        self.title_lbl = ctk.CTkLabel(self, text="Active Modulation Center", font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"), text_color="#FFFFFF")
        self.title_lbl.pack(side="left")

        # PTT Checkbox
        self.ptt_check_var = ctk.BooleanVar(value=self.app_state.ptt_enabled)
        self.ptt_check = ctk.CTkCheckBox(self, text="Global PTT", variable=self.ptt_check_var, font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), fg_color="#007AFF", hover_color="#005BB5", command=self._on_ptt_toggle)
        self.ptt_check.pack(side="left", padx=(32, 16))
        
        # PTT Hotkey Setup
        hotkey_txt = f"Key: {self.app_state.ptt_hotkey}"
        self.ptt_hotkey_btn = ctk.CTkButton(self, text=hotkey_txt, width=80, height=28, corner_radius=6, fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#FFFFFF", command=self._trigger_hotkey_setup)
        self.ptt_hotkey_btn.pack(side="left")
        
        # Real-Time Toggle
        self.local_realtime_btn = ctk.CTkButton(self, text="LOCAL REALTIME OFF", width=140, height=28, corner_radius=6, fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), command=self._trigger_realtime)
        self.local_realtime_btn.pack(side="left", padx=16)

        # Mini View Toggle
        self.mini_toggle_btn = ctk.CTkButton(self, text="MINI VIEW", width=80, height=28, corner_radius=6, fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#A0A0A0", command=self.callbacks.get('on_toggle_mini', lambda: None))
        self.mini_toggle_btn.pack(side="right")
        
        # Status Indicator
        self.status_indicator = ctk.CTkLabel(self, text="● SERVER CHECKING", text_color="#A0A0A0", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"))
        self.status_indicator.pack(side="right", padx=16)
        
    def _on_ptt_toggle(self):
        self.app_state.update("ptt_enabled", self.ptt_check_var.get())
        
    def _trigger_hotkey_setup(self):
        if 'on_hotkey_setup' in self.callbacks:
            self.callbacks['on_hotkey_setup'](self.ptt_hotkey_btn)
            
    def _trigger_realtime(self):
        if 'on_toggle_realtime' in self.callbacks:
            self.callbacks['on_toggle_realtime'](self.local_realtime_btn)

    def update_status(self, text, color):
        self.status_indicator.configure(text=text, text_color=color)
