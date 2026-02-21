import json
import os
from typing import Dict, Any

class SessionManager:
    CONFIG_FILE = "session_config.json"

    @classmethod
    def save_session(cls, state: Dict[str, Any]):
        """Save current app state to a config file."""
        try:
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"Error saving session: {e}")

    @classmethod
    def load_session(cls) -> Dict[str, Any]:
        """Load last saved app state."""
        if not os.path.exists(cls.CONFIG_FILE):
            return {}
        
        try:
            with open(cls.CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session: {e}")
            return {}

    @classmethod
    def get_default_state(cls) -> Dict[str, Any]:
        """Return the default safe state for the application."""
        return {
            "last_input_device": None,
            "last_output_device": None,
            "last_profile": "female_dsp",
            "last_mode": "female",
            "is_active": False,  # Always start OFF for safety
            "advanced_expanded": False,
            "latency_preset": "Balanced",
            "use_noise_gate": False,
            "use_echo": False,
            "elevenlabs_enabled": False,
            "elevenlabs_api_key": "sk_cf8e0e9100def3309702b6d55050aca7366a56038b150a86",
            "elevenlabs_voice_id": "uCNfGgx20cVUPpqSApMp",
            "elevenlabs_hotkey": "<alt>"
        }
