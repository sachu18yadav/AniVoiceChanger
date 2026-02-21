# Voice Chameleon (DOTA 2 Voice over tool)

Two voice changer versions: **Basic** (simple, lightweight) and **Advanced** (full GUI).

---

## Quick Start

### 1. Install Dependencies (one-time)
Double-click `install_deps.bat` or run:
```
pip install -r virtual_mic/requirements.txt
```

### 2. Run Basic Version
Double-click `run_basic.bat` or:
```
cd virtual_mic
python basic_voice_changer.py
```
- Hold **V** key → speak → release → hear converted voice
- Edit `basic_voice_changer.py` to change the hotkey or effect

### 3. Run Advanced Version (GUI)
Double-click `run_advanced.bat` or:
```
cd virtual_mic
python gui.py
```
- Select your **microphone** from the Input Source dropdown
- Choose a **Voice Profile** (Tsukiyomi, Standard Anime, etc.)
- Adjust the **Pitch** slider
- Click **RECORD 5S TEST** → speak for 5 seconds → hear the result
- Click **REPLAY** to re-listen
- Click **LOCAL REALTIME** to test live processing

---

## How to Test Live Voice Changing

### Method 1: Record & Test (Recommended First)
1. Run `run_advanced.bat`
2. Select your Realtek mic from the dropdown
3. Select "Tsukiyomi-chan" profile, set pitch to 12
4. Click **RECORD 5S TEST**, speak, wait for playback

### Method 2: Live Realtime (In-Game)
1. Run `run_advanced.bat`
2. Select your mic, choose a profile
3. Click **LOCAL REALTIME ON**
4. Your voice is now being processed live through the DSP engine
5. In your game/app, set the audio input to **CABLE Output (VB-Audio Virtual Cable)**
6. Click **LOCAL REALTIME OFF** to stop

### Method 3: Push-to-Talk (Basic)
1. Run `run_basic.bat`
2. Hold **V**, speak, release
3. Converted audio plays to speakers + VB-Cable automatically

---

## Project Structure

```
DOTA 2 Voice over tool/
├── run_basic.bat          # Launch basic version
├── run_advanced.bat       # Launch GUI version
├── install_deps.bat       # Install all dependencies
├── README.md
├── virtual_mic/           # Core engine (shared by both versions)
│   ├── basic_voice_changer.py   # Basic push-to-talk
│   ├── gui.py                   # Advanced GUI
│   ├── virtual_mic.py           # Voice changer engine
│   ├── effects.py               # DSP effects (pitch shift, anime voice)
│   ├── w_okada_client.py        # W-Okada backend client (advanced only)
│   ├── ai_engine/               # AI model inference
│   ├── models/                  # Voice model files (.pth, .index)
│   ├── utils/                   # Session manager, system checker
│   └── requirements.txt
└── voice-changer/         # W-Okada server (advanced backend, optional)
```

## Notes
- **Basic version** needs only: `sounddevice`, `numpy`, `pedalboard`, `keyboard`
- **Advanced GUI** additionally needs: `customtkinter`, `pystray`, `Pillow`, `psutil`
- The **w-okada server** is only needed if you want to use the AI model slots (Chihiro, Foamy) via the remote backend. The local DSP engine works without it.
