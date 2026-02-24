import customtkinter as ctk
import numpy as np

class WaveformVisualizer(ctk.CTkCanvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, highlightthickness=0, **kwargs)
        self.configure(bg="#1A1A1A")
        self.data = np.zeros(100)
        self.is_running = True
        self.draw_loop()

    def update_data(self, new_data):
        if len(new_data) > 0:
            # Simple RMS-based peak for visualization, boosted arbitrarily to look visible
            peak = np.abs(new_data).max() * 5.0
            peak = min(peak, 1.0) # Clamp to 1.0 MAX
            self.data = np.roll(self.data, -1)
            self.data[-1] = peak

    def draw_loop(self):
        if not self.is_running:
            return
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
        
    def stop(self):
        self.is_running = False


class LabFrame(ctk.CTkFrame):
    """The Testing Laboratory containing the audio visualizer and recording toggles."""
    
    def __init__(self, master, callbacks):
        super().__init__(master, fg_color="#121214", corner_radius=12, border_width=1, border_color="#1C1C1E")
        self.callbacks = callbacks # on_record, on_replay, on_realtime_test
        
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(side="left", padx=32, pady=16)
        
        ctk.CTkLabel(info_frame, text="TESTING LABORATORY", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), text_color="#A0A0A0").pack(anchor="w")
        ctk.CTkLabel(info_frame, text="Run local or realtime 60s tests to verify audio quality.", font=ctk.CTkFont(family="Segoe UI", size=11), text_color="#808080").pack(anchor="w")

        controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        controls_frame.pack(side="right", padx=32, pady=16)

        self.local_rec_btn = ctk.CTkButton(controls_frame, text="TEST LOCAL (5s)", width=120, height=28, corner_radius=6, font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), command=self._trigger_record)
        self.local_rec_btn.pack(side="left", padx=(0, 6))

        self.realtime_test_btn = ctk.CTkButton(controls_frame, text="REALTIME TEST (60s)", width=130, height=28, corner_radius=6, fg_color="#4B4B4B", hover_color="#5C5C5C", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), command=self._trigger_realtime_test)
        self.realtime_test_btn.pack(side="left", padx=(0, 6))

        self.local_replay_btn = ctk.CTkButton(controls_frame, text="REPLAY", width=80, height=28, corner_radius=6, fg_color="#1C1C1E", hover_color="#2C2C2E", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), command=self._trigger_replay)
        # Don't pack it initially
        # self.local_replay_btn.pack(side="left", padx=(0, 3))

        # Audio Visualizer (Dual Channel)
        vis_frame = ctk.CTkFrame(self, fg_color="transparent", height=78)
        vis_frame.pack(side="right", fill="x", expand=True, padx=(8, 16), pady=8)
        vis_frame.pack_propagate(False)
        
        self.visualizer = WaveformVisualizer(vis_frame, width=280, height=36)
        self.visualizer.pack(fill="x", expand=True, pady=(0, 4))
        
        self.out_visualizer = WaveformVisualizer(vis_frame, width=280, height=36)
        self.out_visualizer.pack(fill="x", expand=True)
        
    def _trigger_record(self):
        if 'on_record' in self.callbacks:
            self.callbacks['on_record'](self.local_rec_btn, self.realtime_test_btn, self.local_replay_btn)
            
    def _trigger_realtime_test(self):
        if 'on_realtime_test' in self.callbacks:
            self.callbacks['on_realtime_test'](self.local_rec_btn, self.realtime_test_btn, self.local_replay_btn)
            
    def _trigger_replay(self):
        if 'on_replay' in self.callbacks:
            self.callbacks['on_replay'](self.local_replay_btn)
