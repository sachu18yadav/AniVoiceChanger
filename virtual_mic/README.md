# Virtual Microphone Project

## Overview
This project implements a virtual microphone on Windows using **VB‑Cable** and Python. It captures audio from your physical microphone, applies real‑time effects (e.g., pitch shifting), and streams the processed audio to a virtual audio device that any application (Discord, Zoom, games, etc.) can use as its input.

## Prerequisites
1. **VB‑Cable** (virtual audio driver)
   - Download from https://vb-audio.com/Cable/
   - Run the installer (`setup_x64.exe` for 64‑bit Windows) and restart.
   - Verify that **"CABLE Input"** and **"CABLE Output"** appear in **Sound Settings**.
2. **Python 3.9+**
3. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure
```
virtual_mic/
├─ requirements.txt          # Python dependencies
├─ effects.py                # Audio effect functions (pitch shift, etc.)
├─ virtual_mic.py            # Main pipeline script
├─ README.md                 # This file
└─ test_pipeline.py          # Basic sanity‑check test
```

## Usage
Run the main script with optional arguments:
```bash
python virtual_mic.py --semitones 3 --buffer-size 1024
```
- `--semitones`: Pitch shift amount (positive = higher pitch). Default 0 (no shift).
- `--buffer-size`: Number of samples per audio block. Default 1024 (~21 ms at 48 kHz).

The script will:
1. Open the default microphone for input.
2. Detect the VB‑Cable output device (named *CABLE Input* or *CABLE Output*).
3. Apply the selected effect(s).
4. Stream the processed audio to the virtual cable.
5. Print status messages; press **Ctrl+C** to stop.

## Configuring Applications
After the script is running:
- Open **Discord** → *User Settings* → *Voice & Video*.
- Set **Input Device** to **"CABLE Output"** (or **"CABLE Input"** depending on driver naming).
- Disable Discord's built‑in noise suppression and echo cancellation for best results.

## Testing
Run the provided test to ensure the audio pipeline can start without errors:
```bash
python -m unittest test_pipeline.py
```
The test opens an input and output stream using the same device detection logic and then closes them.

## Extending the Project
- Implement additional effects in `effects.py` (e.g., robot, helium).
- Integrate AI voice conversion models (RVC, etc.) by replacing the stub `voice_convert`.
- Add a GUI or system‑tray icon for hot‑key toggling.
- Package as a Windows executable with PyInstaller for distribution.

## License
MIT – feel free to modify and share.
