# Real-Time Anime Voice Changer — Implementation Plan

## What This Is

A real-time voice conversion system that takes microphone input and outputs a female anime voice to any application (Discord, Zoom, OBS). Built on the w-okada/voice-changer runtime using RVC V2 models.

---

## The Stack

```
Microphone → w-okada/voice-changer → VB-Cable → Discord / Zoom / OBS
```

Three components. That is the entire working system.

---

## Phase 1 — Working System (Right Now)

### How It Works Under the Hood

RVC (Retrieval-based Voice Conversion) decomposes your voice into three things — phonetic content, pitch (F0), and speaker identity — then reassembles them using a target character's neural model. The key difference from a simple pitch shifter is the `.index` file: a k-NN feature database of the target character's actual voice. Rather than generating a new voice, RVC retrieves the closest matching features from that database. This is why it sounds like a specific character rather than a generic pitch-shifted voice.

The audio pipeline processes overlapping frames (chunks) of audio continuously. Each chunk is processed and then cross-faded with the next to avoid audible clicks at boundaries. Larger chunks = better quality but higher latency. Smaller chunks = lower latency but audio artifacts.

---

### Installation Order

**Step 1 — VB-Cable (virtual audio driver)**
Install before anything else. Applications need to see the virtual device at boot time.
- Download: https://vb-audio.com/Cable/
- After install, verify it appears in Windows Sound settings as a playback and recording device.

**Step 2 — w-okada voice-changer**
- Repository: https://github.com/w-okada/voice-changer
- Download the Windows build from GitHub Releases (prefer this over Google Drive links).
- Extract the folder, double-click `start_http.bat`.
- First boot downloads model weights from Hugging Face — keep network available.
- When Windows Defender Firewall prompts, allow access (the tool runs a local HTTP server).
- If no browser window opens, re-run `start_http.bat`.
- In the browser UI: select **RVC** → click **Start**.

**Step 3 — Model files**
Place your character model files here:
```
[voice-changer install folder]/
├── weights/
│   └── your_character.pth        ← required
└── logs/
    └── your_character/
        └── total_fea.npy         ← use this if you have it, improves character similarity
```
In the UI: Edit → Select → navigate to the `.pth` file → Upload.

---

### Settings (Empirically Tested)

These values were validated through live testing. Start here before adjusting anything.

| Parameter | Value | Reason |
|---|---|---|
| Chunk | 384 | Best quality/latency balance — roughly 1s total latency |
| Extra | 4096 | Inference headroom between chunks |
| Crossfade | 300 | Eliminates chunk boundary artifacts |
| Tran K | **300 (max)** | Single biggest fix for choppy/stuttering output |
| Pitch Algorithm | Crepe, threshold 100 | Best tracking for soft or whispered input |
| GPU | Select explicitly | Do not leave on auto — can silently fall back to CPU |
| Pitch Tune | Start at **+12**, dial up to **+20–23** for anime archetype | Calibrate by ear per model |

Hit **Save Settings** after configuring. Settings persist per voice profile.

---

### Audio Routing

| Location | Setting |
|---|---|
| w-okada Input | Your physical microphone |
| w-okada Output | VB-Cable Input |
| Discord/Zoom Microphone | VB-Cable Output |

---

### Pitch Calibration Guide

- **+12 semitones** — general female quality, natural
- **+20–23 semitones** — anime archetype (exaggerated, high-pitched character voices)
- The right value is character-specific and must be found by ear — no table reliably predicts it
- Held vowels (e.g. "eeee") will wobble at chunk boundaries regardless of pitch — this is normal and disappears in natural speech

---

### Known Limitations

- ~1 second end-to-end latency at chunk 384 — the trade-off for clean audio
- Constant held syllables produce pitch wobble — inaudible during normal speech
- Models trained on Japanese audio perform better in Japanese than English
- The index file upload UI has had bugs in some releases — if it fails, place files in the directory structure above and let the tool auto-discover them

---

## Content Encoder Note (For Custom Model Training)

When training your own character model, RVC V2 uses **ContentVec** as its content encoder by default. ContentVec is preferable to plain HuBERT because it applies pitch and formant perturbations during training, making its content representations independent of the source speaker's vocal tract. In practice this means: when you speak into the mic, the system extracts your *words* cleanly without carrying your voice's resonance into the conversion. This is why the output sounds like the character rather than a pitch-shifted version of you.

If you train a custom model:
- Use 10–20 minutes of clean character audio minimum
- Remove music, background noise, and reverb first (UVR / Ultimate Vocal Remover is the standard tool for this)
- Normalize loudness and slice into segments under 30 seconds before training

---

## Phase 2 — Multiple Voices (Future)

When you are ready to support multiple characters and switch between them:

- w-okada exposes a **local HTTP API** — use this to drive voice switching programmatically rather than through the UI
- Store each character's settings as a config object (pitch semitones, chunk size, index ratio, model paths)
- On voice switch: load new `.pth` + `.npy`, update settings via API, call `torch.cuda.empty_cache()` after unloading the old model to reclaim VRAM
- The audio stream does not need to stop during a model swap — there will be a brief silent gap (~200ms on SSD) which is acceptable

**Index Ratio** (not exposed prominently in the UI but important):
- 0.6–0.75 = stronger character similarity, may artifact if your voice is very different from the training voice
- 0.3–0.4 = more natural prosody, less character-specific

---

## Phase 3 — Any-to-Any Zero-Shot (Future)

The long-term direction: instead of a trained `.pth` per character, the user provides a 5–10 second voice clip as a prompt. A speaker encoder (ECAPA-TDNN) extracts an embedding from the clip, which conditions a latent flow matching model to synthesize in that voice — no fine-tuning required.

**Do not build this now.** Evaluate candidate frameworks (OpenVoice V2, CosyVoice, Seed-VC) when Phase 2 is stable. The field is moving fast — the best option at Phase 3 time likely does not exist yet in its final form.

The only Phase 3 decision to make now is architectural: keep the audio transport layer (w-okada or equivalent) decoupled from the inference model. As long as the model is swappable behind an interface, Phase 3 is a model swap, not a rewrite.

---

## Latency Reality

| Setup | Expected Latency |
|---|---|
| Consumer GPU, WDM drivers, chunk 384 | 800ms – 1200ms |
| Consumer GPU, WASAPI, chunk 192 | 300ms – 500ms (some audio artifacts) |
| NVIDIA GPU, ASIO drivers, TensorRT FP16 | 80ms – 150ms (stretch goal) |
| CPU only | 1500ms+ (not recommended for real use) |

The 60–80ms figure cited in some literature requires ASIO drivers, a dedicated NVIDIA GPU, TensorRT optimization, and no competing workloads. Treat it as a ceiling, not a baseline.

---

## Quick Reference Checklist

- [ ] VB-Cable installed and visible in Windows Sound settings
- [ ] w-okada voice-changer extracted and `start_http.bat` runs without error
- [ ] Firewall access granted
- [ ] `.pth` model file placed in `weights/` folder
- [ ] `.npy` index file placed in `logs/[model_name]/` folder
- [ ] Chunk = 384, Extra = 4096, Crossfade = 300, Tran K = 300
- [ ] Crepe pitch algorithm selected, threshold = 100
- [ ] GPU selected explicitly
- [ ] Pitch tuned by ear (start +12, go higher for anime archetype)
- [ ] Settings saved in UI
- [ ] VB-Cable Output set as microphone in Discord/Zoom
- [ ] Audio confirmed working end-to-end