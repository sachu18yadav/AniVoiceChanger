import os
from utils.session_manager import SessionManager

class AppState:
    """Centralized state management for Voice Chameleon."""
    
    def __init__(self):
        self.session_data = SessionManager.load_session()
        
        # Core Settings
        self.backend_mode = self.session_data.get("backend_mode", "Local DSP")
        self.profile = self.session_data.get("last_mode", "standard_female")
        self.pitch = self.session_data.get("last_pitch", 12)
        
        # PTT Logic
        self.ptt_enabled = self.session_data.get("ptt_enabled", False)
        self.ptt_hotkey = self.session_data.get("ptt_hotkey", "<alt>")
        self.ptt_active = False
        
        # Audio Engine State
        self.local_test_active = False
        self.local_record_active = False
        self.passthrough_active = False
        
        # Component References (to be set by AppRouter)
        self.local_engine = None
        self.w_okada_engine = None
        
        # Caching
        self.available_models = self._scan_models()
        self.input_devices = []
        
        # Callbacks for UI updates
        self._listeners = []

    def _scan_models(self):
        """Scans the models/ directory once at startup."""
        models = {}
        model_dir = "models"
        if os.path.exists(model_dir):
            for m_dir in os.listdir(model_dir):
                full_path = os.path.join(model_dir, m_dir)
                if os.path.isdir(full_path):
                    has_pth = any(f.endswith('.pth') for f in os.listdir(full_path))
                    if has_pth:
                        label = f"{m_dir.replace('_', ' ').title()} (AI)"
                        models[m_dir] = {"pitch": 12, "label": label} # Default pitch unless specified elsewhere
        return models

    def subscribe(self, callback):
        """Register a callback to be notified of state changes."""
        self._listeners.append(callback)
        
    def _notify(self, key, value):
        for listener in self._listeners:
            listener(key, value)

    def update(self, key, value):
        """Update a state value and notify listeners."""
        if hasattr(self, key):
            setattr(self, key, value)
            self._notify(key, value)
            
    def save(self):
        """Persist current relevant state to session data."""
        state_dict = {
            "backend_mode": self.backend_mode,
            "last_mode": self.profile,
            "last_pitch": self.pitch,
            "ptt_enabled": self.ptt_enabled,
            "ptt_hotkey": self.ptt_hotkey
        }
        SessionManager.save_session(state_dict)
