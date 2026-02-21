import requests
import json
import threading
from dataclasses import dataclass
from typing import Optional, Any
import requests.exceptions

@dataclass
class VoiceProfile:
    name: str
    display_name: str
    # Settings tuned empirically per character
    pitch_semitones: int = 12
    index_ratio: float = 0.6
    pitch_algorithm: str = "crepe"
    chunk_size: int = 384
    tran_k: int = 300
    crossfade: int = 300
    
    # Internal w-okada slot/model mapping if known, or file paths if we need to load them programmatically
    # The current w-okada API usually lets you switch between loaded slots, or update settings.
    slot: int = 0
    pth_path: Optional[str] = None
    index_path: Optional[str] = None

class WOkadaEngineClient:
    """HTTP Client to interact with the local w-okada voice-changer."""
    def __init__(self, host="127.0.0.1", port=18888):
        self.base_url = f"http://{host}:{port}"
        self._lock = threading.Lock()
        self._current_profile = None
    
    def is_alive(self) -> bool:
        """Check if the w-okada server is running and responding."""
        try:
            resp = requests.get(f"{self.base_url}/test", timeout=2) # w-okada usually has this or root
            return resp.status_code == 200 or resp.status_code == 404
        except requests.exceptions.RequestException:
            return False

    def get_info(self) -> dict:
        """Fetch current settings from w-okada."""
        try:
            resp = requests.get(f"{self.base_url}/info", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {}
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch w-okada info: {e}")
            return {}

    def update_settings(self, key: str, val: Any) -> bool:
        """Update a single specific setting in w-okada."""
        try:
            payload = {key: val}
            resp = requests.post(f"{self.base_url}/update_settings", json=payload, timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Failed to update w-okada setting {key}: {e}")
            return False

    def switch_voice(self, profile: VoiceProfile) -> bool:
        """
        Swap character settings dynamically.
        w-okada API allows changing 'tran' (pitch), 'f0Detector' (algorithm),
        'chunk' (chunk size), 'extraConvertSize', etc.
        """
        print(f"Switching Voice Profile to: {profile.display_name}")
        with self._lock:
            # For w-okada MMVCServerSIO HTTP API:
            # Key names can vary by version, typical keys:
            # - tran: target pitch
            # - f0Detector: crepe/rmvpe/fcpe
            # - chunk: 384
            # - indexRatio: float
            settings = {
                "tran": profile.pitch_semitones,
                "f0Detector": profile.pitch_algorithm,
                "chunk": profile.chunk_size,
                "indexRatio": profile.index_ratio
            }
            # Many APIs require a full state push or individual setting updates.
            # Here we loop individual updates to ensure compatibility with /update_settings if required
            # Or send them as a block depending on w-okada API specs.
            
            # W-Okada API typically handles block updates:
            try:
                # Assuming /update_settings accepts partial JSON state updates
                resp = requests.post(f"{self.base_url}/update_settings", json=settings, timeout=5)
                success = (resp.status_code == 200)
                
                # If slot changing is required and API supports "modelSlotIndex"
                if success and resp:
                     requests.post(f"{self.base_url}/update_settings", json={"modelSlotIndex": profile.slot}, timeout=5)

                if success:
                    self._current_profile = profile
                return success
            except requests.exceptions.RequestException as e:
                print(f"Failed to hot-swap profile {profile.name}: {e}")
                return False

# Example Pre-configured Characters
TSUKIYOMI = VoiceProfile(
    name="tsukiyomi",
    display_name="Tsukiyomi-chan",
    pitch_semitones=22,
    index_ratio=0.6,
    pitch_algorithm="crepe",
    chunk_size=384,
    slot=0
)

STANDARD_FEMALE = VoiceProfile(
    name="standard_female",
    display_name="Standard Anime Female",
    pitch_semitones=12,
    index_ratio=0.5,
    pitch_algorithm="crepe",
    chunk_size=384,
    slot=1
)

CHIHIRO = VoiceProfile(
    name="chihiro",
    display_name="Chihiro Fujisaki",
    pitch_semitones=12,
    index_ratio=0.6,
    pitch_algorithm="crepe",
    chunk_size=384,
    slot=2,
    pth_path="models/chihiro/chihiro.pth",
    index_path="models/chihiro/chihiro.index"
)

FOAMY = VoiceProfile(
    name="foamy",
    display_name="Foamy the Squirrel",
    pitch_semitones=0,
    index_ratio=0.7,
    pitch_algorithm="crepe",
    chunk_size=384,
    slot=3,
    pth_path="models/foamy/foamy.pth",
    index_path="models/foamy/foamy.index"
)
