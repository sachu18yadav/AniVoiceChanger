"""
Lightweight RVC Inference Engine
Uses the standalone HuBERT model that loads RVC's actual checkpoint.
CPU-only, thread-limited for gaming-friendly performance.
"""
import os
import traceback
import numpy as np
from pathlib import Path
from time import time
from scipy import signal

import torch
import torch.nn.functional as F
import librosa

# Limit CPU threads for gaming friendliness
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

MODELS_DIR = Path(__file__).resolve().parent.parent / "virtual_mic" / "models"
HUBERT_CACHE = Path(__file__).resolve().parent / "hubert_cache"
HUBERT_CACHE.mkdir(exist_ok=True)

# High-pass filter to remove DC offset (matches real RVC pipeline)
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

SR = 16000
WINDOW = 160
X_PAD = 1


# ═══════════════════════════════════════════════════════════
# HuBERT (standalone model loading RVC's actual checkpoint)
# ═══════════════════════════════════════════════════════════

_hubert_model = None
_hubert_loaded = False


def load_hubert():
    """Load HuBERT from RVC's hubert_base.pt checkpoint."""
    global _hubert_model, _hubert_proj, _hubert_loaded
    if _hubert_loaded:
        return _hubert_model, _hubert_proj

    from torchaudio_remapper import load_rvc_hubert_to_torchaudio

    checkpoint_path = HUBERT_CACHE / "hubert_base.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"hubert_base.pt not found at {checkpoint_path}. "
            "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
        )

    _hubert_model, _hubert_proj = load_rvc_hubert_to_torchaudio(str(checkpoint_path))
    _hubert_loaded = True
    return _hubert_model, _hubert_proj


def extract_hubert_features(model, proj, audio_16k, version="v2"):
    """
    Extract HuBERT features matching the real RVC pipeline.
    """
    waveform = torch.from_numpy(audio_16k).float().unsqueeze(0)  # (1, T)
    
    with torch.no_grad():
        if version == "v2":
            # Torchaudio outputs features up to layer 12
            layer_outputs, _ = model.extract_features(waveform)
            feats = layer_outputs[-1]  # 768-dim
        else:
            # v1 uses layer 9
            layer_outputs, _ = model.extract_features(waveform, num_layers=9)
            feats = proj(layer_outputs[-1])  # 768 → 256

    # Interpolate 2x to match F0 frame count (exact RVC behavior)
    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    return feats


# ═══════════════════════════════════════════════════════════
# Index Retrieval
# ═══════════════════════════════════════════════════════════

def load_index(index_path):
    if index_path is None or not Path(index_path).exists():
        return None, None
    try:
        import faiss
        index = faiss.read_index(str(index_path))
        big_npy = index.reconstruct_n(0, index.ntotal)
        print(f"  ✓ Index: {Path(index_path).name} ({index.ntotal} vectors)")
        return index, big_npy
    except Exception as e:
        print(f"  Index load failed: {e}")
        return None, None


def apply_index(feats, index, big_npy, index_rate=0.75):
    if index is None or big_npy is None or index_rate == 0:
        return feats
    npy = feats[0].cpu().numpy().astype("float32")
    score, ix = index.search(npy, k=8)
    weight = np.square(1 / (score + 1e-6))
    weight /= weight.sum(axis=1, keepdims=True)
    retrieved = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
    feats = torch.from_numpy(retrieved).unsqueeze(0).float() * index_rate + (1 - index_rate) * feats
    return feats


# ═══════════════════════════════════════════════════════════
# F0 Pitch Extraction (pyworld harvest)
# ═══════════════════════════════════════════════════════════

def get_f0(audio_16k, p_len, f0_up_key=0):
    import pyworld
    from scipy.signal import medfilt

    f0_min, f0_max = 50, 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    audio_double = audio_16k.astype(np.double)
    f0, t = pyworld.harvest(audio_double, fs=SR, f0_ceil=f0_max, f0_floor=f0_min,
                            frame_period=WINDOW / SR * 1000)
    f0 = pyworld.stonemask(audio_double, f0, t, SR)
    f0 = medfilt(f0.astype(np.float64), 3)

    if len(f0) < p_len: f0 = np.pad(f0, (0, p_len - len(f0)))
    elif len(f0) > p_len: f0 = f0[:p_len]

    f0 *= pow(2, f0_up_key / 12)
    f0bak = f0.copy().astype(np.float32)

    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    return f0_coarse, f0bak


# ═══════════════════════════════════════════════════════════
# RVC Model Loader
# ═══════════════════════════════════════════════════════════

def load_rvc_model(model_path):
    from infer_pack.models import (
        SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono,
    )
    model_path = Path(model_path)
    if not model_path.exists(): return None, None

    print(f"  Loading RVC model: {model_path.name}...")
    cpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    config = cpt["config"]
    version = cpt.get("version", "v1")
    f0_flag = cpt.get("f0", 1)
    tgt_sr = config[-1]
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]

    try:
        if version == "v1":
            net_g = SynthesizerTrnMs256NSFsid(*config, is_half=False) if f0_flag else SynthesizerTrnMs256NSFsid_nono(*config)
        else:
            net_g = SynthesizerTrnMs768NSFsid(*config, is_half=False) if f0_flag else SynthesizerTrnMs768NSFsid_nono(*config)
        if hasattr(net_g, 'enc_q'): del net_g.enc_q
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.eval().float()
        print(f"  ✓ Model: {model_path.stem} (v{version}, sr={tgt_sr}, f0={'yes' if f0_flag else 'no'})")
    except Exception as e:
        print(f"  Failed: {e}"); traceback.print_exc(); return None, None

    return net_g, {"tgt_sr": tgt_sr, "version": version, "f0": f0_flag, "config": config}


# ═══════════════════════════════════════════════════════════
# Full Inference Pipeline
# ═══════════════════════════════════════════════════════════

def infer(audio_np, sr, model, model_info, f0_up_key=0, index=None, big_npy=None, index_rate=0.4, rms_mix_rate=0.8):
    t0 = time()
    tgt_sr = model_info["tgt_sr"]
    version = model_info["version"]

    # Resample to 16kHz
    audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000) if sr != 16000 else audio_np.copy()
    audio_16k = signal.filtfilt(bh, ah, audio_16k).astype(np.float32)

    # Pad
    t_pad = SR * X_PAD
    t_pad_tgt = tgt_sr * X_PAD
    audio_padded = np.pad(audio_16k, (t_pad, t_pad), mode="reflect")

    # F0
    p_len = audio_padded.shape[0] // WINDOW
    f0_coarse, f0_raw = get_f0(audio_padded, p_len, f0_up_key)
    t1 = time()

    # HuBERT features
    hubert, proj = load_hubert()
    feats = extract_hubert_features(hubert, proj, audio_padded, version)
    t2 = time()

    # Index retrieval
    if index is not None:
        feats = apply_index(feats, index, big_npy, index_rate)

    # Align
    if feats.shape[1] < p_len:
        p_len = feats.shape[1]; f0_coarse = f0_coarse[:p_len]; f0_raw = f0_raw[:p_len]
    elif feats.shape[1] > p_len:
        feats = feats[:, :p_len, :]

    # Tensors
    phone_lengths = torch.tensor([p_len]).long()
    sid = torch.tensor([0]).long()
    pitch = torch.from_numpy(f0_coarse).long().unsqueeze(0) if model_info["f0"] else None
    pitchf = torch.from_numpy(f0_raw).float().unsqueeze(0) if model_info["f0"] else None

    # Synthesis
    with torch.no_grad():
        if pitch is not None:
            audio_out = model.infer(feats, phone_lengths, pitch, pitchf, sid)[0][0, 0]
        else:
            audio_out = model.infer(feats, phone_lengths, sid)[0][0, 0]
    t3 = time()

    # Evaluate raw output
    result = audio_out.data.cpu().float().numpy()

    # RMS ENVELOPE MATCHING (Fixes robotic unclear noises during silence/consonants)
    if rms_mix_rate > 0:
        # Resample padded input to target sr for accurate envelope calculation
        audio_padded_tgt = librosa.resample(audio_padded, orig_sr=SR, target_sr=tgt_sr)
        
        # Match lengths in case of rounding
        min_len = min(len(audio_padded_tgt), len(result))
        audio_padded_tgt = audio_padded_tgt[:min_len]
        result = result[:min_len]

        # Calculate RMS using small windows (e.g. 5ms)
        win_len = tgt_sr // 200
        rms_in = librosa.feature.rms(y=audio_padded_tgt, frame_length=win_len*2, hop_length=win_len)[0]
        rms_out = librosa.feature.rms(y=result, frame_length=win_len*2, hop_length=win_len)[0]
        
        # Interpolate RMS curves up to full sample length
        rms_in_interp = np.interp(np.arange(len(result)), np.linspace(0, len(result), len(rms_in)), rms_in)
        rms_out_interp = np.interp(np.arange(len(result)), np.linspace(0, len(result), len(rms_out)), rms_out)
        
        rms_out_interp = np.maximum(rms_out_interp, 1e-6) # Prevent div by zero
        
        # Calculate dynamic scaling ratio based on original envelope
        ratio = rms_in_interp / rms_out_interp
        
        # Smooth the ratio to avoid distortion clicks
        from scipy.ndimage import gaussian_filter1d
        ratio = gaussian_filter1d(ratio, sigma=tgt_sr//200)
        
        # Mix the normalized envelope with the raw output
        result_rms = result * ratio
        result = result_rms * rms_mix_rate + result * (1 - rms_mix_rate)

    # Trim padding and audio scaling
    if t_pad_tgt > 0 and len(result) > t_pad_tgt * 2:
        result = result[t_pad_tgt:-t_pad_tgt]
        
    audio_max = np.abs(result).max() / 0.99
    if audio_max > 1: result = result / audio_max

    print(f"    F0={t1-t0:.1f}s HuBERT={t2-t1:.1f}s Synth={t3-t2:.1f}s Total={t3-t0:.1f}s ({len(result)} @ {tgt_sr}Hz)")
    return result, tgt_sr
