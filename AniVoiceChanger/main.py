"""
Anime Voice Changer — main.py
Hold a key, speak, release, hear the character.
Press number keys (1-9) to switch voices at runtime.

Supports two modes (set in .env):
  MODE=dsp  — Local DSP pitch shifting via Pedalboard (no AI, instant)
  MODE=rvc  — Real RVC voice conversion using AI models (~1.5s processing)
"""
import os
import sys
import json
import threading
from pathlib import Path
from time import sleep, time

import keyboard
import numpy as np
import sounddevice as sd
import librosa
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "virtual_mic" / "models"
load_dotenv(BASE_DIR / ".env")

MODE            = os.getenv("MODE", "rvc").strip().lower()
PITCH           = int(os.getenv("PITCH", "12"))
PITCH_ALGO      = os.getenv("PITCH_ALGO", "pm").strip()
RECORD_KEY      = os.getenv("RECORD_KEY", "v").strip()

INPUT_DEVICE    = os.getenv("INPUT_DEVICE_INDEX", "").strip()
OUTPUT_DEVICE   = os.getenv("OUTPUT_DEVICE_INDEX", "").strip()
SPEAKER_DEVICE  = os.getenv("SPEAKER_DEVICE_INDEX", "").strip()

INPUT_DEVICE    = int(INPUT_DEVICE) if INPUT_DEVICE else None
OUTPUT_DEVICE   = int(OUTPUT_DEVICE) if OUTPUT_DEVICE else None
SPEAKER_DEVICE  = int(SPEAKER_DEVICE) if SPEAKER_DEVICE else None

AUDIO_DIR       = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
OUTPUT_PATH     = str(AUDIO_DIR / "output.wav")

SAMPLE_RATE     = 48000

# ── Voice Profiles ──────────────────────────────────────────

class VoiceProfile:
    def __init__(self, name, pitch=12, pth_path=None, index_path=None):
        self.name = name
        self.pitch = pitch
        self.pth_path = pth_path
        self.index_path = index_path
        self.rvc_model = None      # Loaded PyTorch model
        self.rvc_info = None       # Model metadata dict
        self.rvc_index = None      # Faiss index for feature retrieval
        self.rvc_big_npy = None    # Reconstructed index vectors

    def __repr__(self):
        return f"{self.name} (pitch +{self.pitch})"

def discover_voices():
    voices = []
    if not MODELS_DIR.exists():
        return voices

    for folder in sorted(MODELS_DIR.iterdir()):
        if not folder.is_dir():
            continue

        pth_file = None
        index_file = None
        metadata = {}

        for f in folder.iterdir():
            if f.suffix == '.pth':
                pth_file = f
            elif f.suffix == '.index':
                index_file = f
            elif f.name == 'metadata.json':
                try:
                    metadata = json.loads(f.read_text())
                except:
                    pass

        if pth_file is None:
            continue

        name = metadata.get("name") or metadata.get("title", folder.name.replace("_", " ").title())
        pitch = metadata.get("pitch", PITCH)

        voices.append(VoiceProfile(
            name=name, pitch=pitch,
            pth_path=str(pth_file),
            index_path=str(index_file) if index_file else None
        ))

    return voices

# ── Auto-detect devices ────────────────────────────────────

def find_device(keyword, kind="input"):
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        ch_key = 'max_input_channels' if kind == "input" else 'max_output_channels'
        if d[ch_key] > 0 and keyword.lower() in d['name'].lower():
            return i
    return None

def auto_detect_devices():
    global INPUT_DEVICE, OUTPUT_DEVICE, SPEAKER_DEVICE
    if INPUT_DEVICE is None:
        INPUT_DEVICE = find_device("realtek", "input") or find_device("microphone", "input") or sd.default.device[0]
    if OUTPUT_DEVICE is None:
        OUTPUT_DEVICE = find_device("CABLE Input", "output")
    if SPEAKER_DEVICE is None:
        SPEAKER_DEVICE = sd.default.device[1]

# ── DSP Engine (fallback) ──────────────────────────────────

def load_dsp_engine(pitch):
    from pedalboard import Pedalboard, PitchShift
    return Pedalboard([PitchShift(semitones=pitch)])

def dsp_convert(board, audio_data, sr):
    return board(audio_data.reshape(1, -1), sr, reset=True).flatten()

# ── Audio I/O ───────────────────────────────────────────────

def record_while_held(device_index, key):
    frames = []
    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(device=device_index, samplerate=SAMPLE_RATE,
                            channels=1, dtype='float32', callback=callback)
    stream.start()
    while keyboard.is_pressed(key):
        sleep(0.01)
    stream.stop()
    stream.close()
    return np.concatenate(frames).flatten() if frames else None

def save_wav(audio, path, sr):
    from scipy.io import wavfile
    if audio.dtype in (np.float32, np.float64):
        audio = (audio * 32767).astype(np.int16)
    wavfile.write(path, sr, audio)

def play_audio(audio, device_id, sr):
    if device_id is None:
        return
    try:
        out = audio.reshape(-1, 1).astype(np.float32)
        with sd.OutputStream(device=device_id, samplerate=sr, channels=1) as stream:
            stream.write(out)
    except Exception as e:
        print(f"  Playback error on device {device_id}: {e}")

# ── Main Loop ───────────────────────────────────────────────

def print_voice_menu(voices, active_idx):
    mode_label = "RVC AI" if MODE == "rvc" else "DSP"
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │   AVAILABLE VOICES ({mode_label})            │")
    print(f"  ├─────────────────────────────────────────┤")
    for i, v in enumerate(voices):
        marker = " ◀ ACTIVE" if i == active_idx else ""
        status = "■" if v.rvc_model is not None else "□"
        print(f"  │  [{i+1}] {status} {v.name:<22} +{v.pitch:>2} ST{marker:<9} │")
    print(f"  └─────────────────────────────────────────┘")
    print(f"  ■ = model loaded, □ = not loaded yet")
    print(f"  Press 1-{len(voices)} to switch voice.\n")

def main():
    print("=" * 50)
    print("  ANIME VOICE CHANGER")
    print("=" * 50)

    # Discover voices
    voices = discover_voices()
    if not voices:
        voices = [VoiceProfile("Default DSP", pitch=PITCH)]
        print("\n  No models found. Using DSP pitch shift.\n")

    active_voice_idx = 0
    active_voice = voices[0]

    auto_detect_devices()

    dev = sd.query_devices()
    in_name = dev[INPUT_DEVICE]['name'] if INPUT_DEVICE is not None else "default"
    print(f"\n  Mic:      [{INPUT_DEVICE}] {in_name}")
    if OUTPUT_DEVICE is not None:
        print(f"  VB-Cable: [{OUTPUT_DEVICE}] {dev[OUTPUT_DEVICE]['name']}")
    else:
        print(f"  VB-Cable: NOT FOUND")
    if SPEAKER_DEVICE is not None:
        print(f"  Speaker:  [{SPEAKER_DEVICE}] {dev[SPEAKER_DEVICE]['name']}")
    print(f"  Mode:     {MODE.upper()}")
    print(f"  Key:      Hold '{RECORD_KEY}' to talk")

    # Load RVC inference engine if in RVC mode
    rvc_engine = None
    dsp_board = None

    if MODE == "rvc":
        try:
            from rvc_infer import load_hubert, load_rvc_model, load_index, infer as rvc_infer_fn
            rvc_engine = {"load_hubert": load_hubert, "load_model": load_rvc_model, "load_index": load_index, "infer": rvc_infer_fn}
            print("\n  RVC inference engine loaded.")

            # Pre-load HuBERT (downloads ~360MB first time)
            load_hubert()

            # Load the active voice model + index
            if active_voice.pth_path:
                model, info = load_rvc_model(active_voice.pth_path)
                active_voice.rvc_model = model
                active_voice.rvc_info = info
                if active_voice.index_path:
                    idx, npy = load_index(active_voice.index_path)
                    active_voice.rvc_index = idx
                    active_voice.rvc_big_npy = npy
        except Exception as e:
            print(f"\n  RVC engine failed: {e}")
            import traceback
            traceback.print_exc()
            print("  Falling back to DSP mode.")
            rvc_engine = None
            dsp_board = load_dsp_engine(active_voice.pitch)
    else:
        dsp_board = load_dsp_engine(active_voice.pitch)

    print_voice_menu(voices, active_voice_idx)

    def reload_voice():
        nonlocal dsp_board
        if rvc_engine and active_voice.pth_path:
            if active_voice.rvc_model is None:
                model, info = rvc_engine["load_model"](active_voice.pth_path)
                active_voice.rvc_model = model
                active_voice.rvc_info = info
                if active_voice.index_path and active_voice.rvc_index is None:
                    idx, npy = rvc_engine["load_index"](active_voice.index_path)
                    active_voice.rvc_index = idx
                    active_voice.rvc_big_npy = npy
        else:
            dsp_board = load_dsp_engine(active_voice.pitch)

    # Number key hotkeys
    def make_switch_fn(idx):
        def switch(_=None):
            nonlocal active_voice_idx, active_voice
            if idx < len(voices) and idx != active_voice_idx:
                active_voice_idx = idx
                active_voice = voices[idx]
                print(f"\n  ★ Switched to: {active_voice.name} (pitch +{active_voice.pitch})")
                reload_voice()
        return switch

    for i in range(min(9, len(voices))):
        keyboard.on_press_key(str(i + 1), make_switch_fn(i), suppress=False)

    print(f"Active: {active_voice.name} (+{active_voice.pitch} ST)")
    print(f"Ready! Hold '{RECORD_KEY}' to speak. Ctrl+C to exit.\n")

    try:
        while True:
            keyboard.wait(RECORD_KEY)

            print(f"[REC] Recording as {active_voice.name}... (release to stop)")
            audio = record_while_held(INPUT_DEVICE, RECORD_KEY)

            if audio is None or len(audio) < SAMPLE_RATE * 0.3:
                print("  Too short, skipped.")
                continue

            duration = len(audio) / SAMPLE_RATE
            print(f"  Recorded {duration:.1f}s, converting...")

            start = time()

            if rvc_engine and active_voice.rvc_model is not None:
                # Real RVC inference (with index retrieval if available)
                converted, out_sr = rvc_engine["infer"](
                    audio, SAMPLE_RATE,
                    active_voice.rvc_model, active_voice.rvc_info,
                    f0_up_key=active_voice.pitch,
                    index=active_voice.rvc_index,
                    big_npy=active_voice.rvc_big_npy,
                    index_rate=0.75
                )
                if out_sr != SAMPLE_RATE:
                    converted = librosa.resample(converted, orig_sr=out_sr, target_sr=SAMPLE_RATE)
                    out_sr = SAMPLE_RATE
            else:
                # DSP fallback
                converted = dsp_convert(dsp_board, audio, SAMPLE_RATE)
                out_sr = SAMPLE_RATE

            elapsed = time() - start
            print(f"  Converted in {elapsed:.2f}s")

            save_wav(converted, OUTPUT_PATH, out_sr)

            threads = []
            if OUTPUT_DEVICE is not None:
                threads.append(threading.Thread(target=play_audio, args=(converted, OUTPUT_DEVICE, out_sr)))
            threads.append(threading.Thread(target=play_audio, args=(converted, SPEAKER_DEVICE, out_sr)))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            print("  Done.\n")

    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
