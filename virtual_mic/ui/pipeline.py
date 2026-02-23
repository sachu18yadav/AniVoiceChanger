import customtkinter as ctk

class PipelineFrame(ctk.CTkFrame):
    """The middle section handling input routing, model selection, and audio profiling."""
    
    def __init__(self, master, app_state, callbacks):
        super().__init__(master, fg_color="transparent")
        self.app_state = app_state
        self.callbacks = callbacks # on_device_change, on_slider_change, on_profile_change
        
        self.grid_columnconfigure((0,1), weight=1)

        # Left Column: Devices
        self.left_col = ctk.CTkFrame(self, fg_color="#121214", corner_radius=12, border_width=1, border_color="#1C1C1E")
        self.left_col.grid(row=0, column=0, padx=(0, 16), sticky="nsew")
        
        ctk.CTkLabel(self.left_col, text="AUDIO PIPELINE", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), text_color="#A0A0A0").pack(pady=(16, 8), padx=32, anchor="w")
        ctk.CTkLabel(self.left_col, text="Input Source", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#FFFFFF").pack(anchor="w", padx=32)
        
        # input device selector
        default_input = self.app_state.input_devices[0] if self.app_state.input_devices else "None"
        for dev_str in self.app_state.input_devices:
            name_lower = dev_str.lower()
            if 'realtek' in name_lower or 'microphone array' in name_lower:
                if 'cable' not in name_lower and 'steam' not in name_lower:
                    default_input = dev_str
                    break
        
        self.input_device_var = ctk.StringVar(value=default_input)
        
        # Only render option menu if we have devices, fallback safely
        vals = self.app_state.input_devices if self.app_state.input_devices else ["None"]
        self.input_selector = ctk.CTkOptionMenu(self.left_col, values=vals, variable=self.input_device_var, command=self._trigger_device_change, font=ctk.CTkFont(family="Segoe UI", size=11), fg_color="#1C1C1E", button_color="#2C2C2E", button_hover_color="#3A3A3C")
        self.input_selector.pack(pady=(4, 16), padx=32, fill="x")
        self._trigger_device_change(default_input)

        ctk.CTkLabel(self.left_col, text="Routing Info", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#FFFFFF").pack(anchor="w", padx=32)
        self.routing_info = ctk.CTkLabel(self.left_col, text="Output: VB-Cable Input", justify="left", font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#A0A0A0")
        self.routing_info.pack(pady=(4, 16), padx=32, anchor="w")

        # Right Column: Voice
        self.right_col = ctk.CTkFrame(self, fg_color="#121214", corner_radius=12, border_width=1, border_color="#1C1C1E")
        self.right_col.grid(row=0, column=1, padx=(16, 0), sticky="nsew")
        
        ctk.CTkLabel(self.right_col, text="VOICE PROFILE", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), text_color="#A0A0A0").pack(pady=(16, 8), padx=32, anchor="w")
        
        self.profile_var = ctk.StringVar(value=self.app_state.profile)
        self.radio_buttons = []

        self.radios_frame = ctk.CTkScrollableFrame(self.right_col, fg_color="transparent", height=80)
        self.radios_frame.pack(fill="x", padx=32)

        self.pitch_slider = ctk.CTkSlider(self.right_col, from_=-12, to=24, number_of_steps=36, fg_color="#1C1C1E", progress_color="#007AFF", button_color="#FFFFFF", button_hover_color="#E0E0E0", command=self._trigger_slider)
        self.pitch_slider.set(self.app_state.pitch)
        self.pitch_slider.pack(pady=(16, 4), padx=32, fill="x")
        self.pitch_lbl = ctk.CTkLabel(self.right_col, text=f"Pitch Offset: {self.pitch_slider.get():.0f} ST", font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#A0A0A0")
        self.pitch_lbl.pack(pady=(0, 16))

    def _trigger_device_change(self, val):
        if 'on_device_change' in self.callbacks:
            self.callbacks['on_device_change'](val)
            
    def _trigger_slider(self, val):
        self.app_state.update("pitch", val)
        self.pitch_lbl.configure(text=f"Pitch Offset: {val:.0f} ST")
        if 'on_slider_change' in self.callbacks:
            self.callbacks['on_slider_change'](val)
            
    def rebuild_radio_buttons(self, presets):
        """Rebuilds the radio buttons list dynamically based on loaded profiles."""
        for rb in self.radio_buttons:
            rb.destroy()
        self.radio_buttons.clear()
        
        for p_id, p_info in presets.items():
            rb = ctk.CTkRadioButton(
                self.radios_frame, text=p_info.get("label", p_id.title()),
                variable=self.profile_var, value=p_id,
                command=self._trigger_profile_change,
                font=ctk.CTkFont(family="Segoe UI", size=12), fg_color="#007AFF"
            )
            rb.pack(anchor="w", pady=4)
            self.radio_buttons.append(rb)
            
        # Select first if current not in presets
        if self.profile_var.get() not in presets and presets:
            first_key = list(presets.keys())[0]
            self.profile_var.set(first_key)
            self._trigger_profile_change()

    def _trigger_profile_change(self):
        new_prof = self.profile_var.get()
        self.app_state.update("profile", new_prof)
        if 'on_profile_change' in self.callbacks:
            self.callbacks['on_profile_change'](new_prof)
