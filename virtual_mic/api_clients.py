import io
import requests
import soundfile as sf
import threading

class ElevenLabsClient:
    """Handles external API calls to ElevenLabs Speech-to-Speech."""
    
    def __init__(self, api_key="sk_cf8e0e9100def3309702b6d55050aca7366a56038b150a86"):
        self.api_key = api_key
        # Default voice ID (e.g., standard female/anime)
        self.default_voice_id = "uCNfGgx20cVUPpqSApMp"
        self.base_url = "https://api.elevenlabs.io/v1/speech-to-speech"

    def process_audio(self, audio_data, sample_rate, voice_id=None, on_success=None, on_error=None):
        """Processes audio asynchronously and triggers callbacks upon completion."""
        vid = voice_id or self.default_voice_id
        
        def _run():
            try:
                wav_io = io.BytesIO()
                sf.write(wav_io, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                wav_io.seek(0)
                
                url = f"{self.base_url}/{vid}"
                headers = {"xi-api-key": self.api_key}
                data = {
                    "model_id": "eleven_multilingual_sts_v2", 
                    "voice_settings": '{"stability": 0.7, "similarity_boost": 0.75}'
                }
                files = {"audio": ("audio.wav", wav_io, "audio/wav")}
                
                resp = requests.post(url, headers=headers, data=data, files=files)
                
                if resp.status_code == 200:
                    out_io = io.BytesIO(resp.content)
                    out_data, out_sr = sf.read(out_io)
                    if on_success:
                        on_success(out_data, out_sr)
                else:
                    if on_error:
                        on_error(f"API Error {resp.status_code}: {resp.text}")
            except Exception as e:
                if on_error:
                    on_error(str(e))
                    
        threading.Thread(target=_run, daemon=True).start()
