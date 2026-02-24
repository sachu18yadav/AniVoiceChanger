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
try:
    import onnxruntime as ort
except ImportError:
    ort = None

device = "cuda" if torch.cuda.is_available() else "cpu"
is_half = device == "cuda"  # Use half precision on GPU for massive speed boost

if device == "cpu":
    # Balanced CPU threads for better performance (2 was too slow for realtime AI testing)
    import multiprocessing
    cores = max(2, min(6, multiprocessing.cpu_count() - 2))
    torch.set_num_threads(cores)
    torch.set_num_interop_threads(cores)

MODELS_DIR = Path(__file__).resolve().parent.parent / "virtual_mic" / "models"
HUBERT_CACHE = Path(__file__).resolve().parent / "hubert_cache"
HUBERT_CACHE.mkdir(exist_ok=True)

# High-pass filter to remove DC offset (matches real RVC pipeline)
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

SR = 16000
WINDOW = 160
X_PAD = 0.05  # 50ms of padding to absorb the 20ms convolution starvation without bloating payload
_COMPILED_MODEL_CACHE = {}  # Maps model_path -> compiled net_g


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
    _hubert_model = _hubert_model.to(device)
    if is_half: _hubert_model = _hubert_model.half()
    _hubert_model.eval()
    
    _hubert_proj = _hubert_proj.to(device)
    if is_half: _hubert_proj = _hubert_proj.half()
    _hubert_proj.eval()
    
    _hubert_loaded = True
    return _hubert_model, _hubert_proj


def extract_hubert_features(model, proj, audio_16k, version="v2"):
    """
    Extract HuBERT features matching the real RVC pipeline.
    """
    waveform = torch.from_numpy(audio_16k).float().unsqueeze(0).to(device)  # (1, T)
    if is_half: waveform = waveform.half()
    
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
    
    retrieved_t = torch.from_numpy(retrieved).unsqueeze(0).to(device)
    if is_half: retrieved_t = retrieved_t.half()
    else: retrieved_t = retrieved_t.float()
        
    feats = retrieved_t * index_rate + (1 - index_rate) * feats
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
    # frame_period=20ms — Must remain 20.0ms. Altering this breaks PyTorch tensor synchronization and destroys inference quality.
    f0, t = pyworld.dio(audio_double, fs=SR, f0_ceil=f0_max, f0_floor=f0_min, frame_period=20.0)
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
            net_g = SynthesizerTrnMs256NSFsid(*config, is_half=is_half) if f0_flag else SynthesizerTrnMs256NSFsid_nono(*config)
        else:
            net_g = SynthesizerTrnMs768NSFsid(*config, is_half=is_half) if f0_flag else SynthesizerTrnMs768NSFsid_nono(*config)
        if hasattr(net_g, 'enc_q'): del net_g.enc_q
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g = net_g.to(device)
        if is_half: net_g = net_g.half()
        else: net_g = net_g.float()
        net_g.eval()
        
        
        # ONNX Export & Initialization
        model_key = str(model_path)
        onnx_path = model_path.with_suffix('.onnx')
        
        if ort is not None:
            if not onnx_path.exists():
                print(f"  Exporting {model_path.name} to ONNX for 5x faster CPU inference...")
                try:
                    export_onnx(net_g, model_info={"f0": f0_flag}, export_path=str(onnx_path))
                    print(f"  ✓ Exported ONNX to {onnx_path.name}")
                except Exception as e:
                    print(f"  \u26a0\ufe0f Failed to export ONNX: {e}")
            
            if onnx_path.exists():
                if model_key not in _COMPILED_MODEL_CACHE:
                    print("  Loading ONNX Runtime session...")
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
                    
                    sess_options = ort.SessionOptions()
                    if device == "cpu":
                        import multiprocessing
                        sys_cores = max(2, min(6, multiprocessing.cpu_count() - 2))
                        sess_options.intra_op_num_threads = sys_cores
                        sess_options.inter_op_num_threads = sys_cores
                        
                    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
                    _COMPILED_MODEL_CACHE[model_key] = session
                    print(f"  ✓ Model (ONNX): {onnx_path.name}")
                return _COMPILED_MODEL_CACHE[model_key], {"tgt_sr": tgt_sr, "version": version, "f0": f0_flag, "config": config, "is_onnx": True}
                
        # PyTorch Fallback (torch.compile)
        if model_key not in _COMPILED_MODEL_CACHE:
            try:
                net_g = torch.compile(net_g, mode="reduce-overhead")
                _COMPILED_MODEL_CACHE[model_key] = net_g
                print(f"  ✓ Model: {model_path.stem} (v{version}, sr={tgt_sr}, f0={'yes' if f0_flag else 'no'}) [compiled]")
            except Exception:
                _COMPILED_MODEL_CACHE[model_key] = net_g
                print(f"  ✓ Model: {model_path.stem} (v{version}, sr={tgt_sr}, f0={'yes' if f0_flag else 'no'})")
        else:
            net_g = _COMPILED_MODEL_CACHE[model_key]
            print(f"  ✓ Model: {model_path.stem} (cached+compiled)")
    except Exception as e:
        print(f"  Failed: {e}"); traceback.print_exc(); return None, None
    return net_g, {"tgt_sr": tgt_sr, "version": version, "f0": f0_flag, "config": config, "is_onnx": False}

def export_onnx(model, model_info, export_path):
    """Export RVC synthesizer to ONNX for ultra-low latency inference."""
    import torch
    model.eval()
    
    if hasattr(model, "remove_weight_norm"):
        try:
            model.remove_weight_norm()
        except Exception:
            pass
            
    # Dummy inputs for tracing based on RVC input shapes
    # feats: (1, 50, 256 for v1 or 768 for v2) -> let's say 256 or 768
    # We use dynamic axes so length doesn't matter, but channel size does.
    # However, to be safe, we just inspect the model's expected feat channel
    feat_dim = 256 if not hasattr(model, 'emb_g') or getattr(model, 'emb_g').weight.shape[1] == 256 else 768
    # Actually, RVC v2 is 768. We can check by version but it's easier to just use the model weights.
    try:
        feat_dim = model.dec.in_channels
    except:
        feat_dim = 768 # Default v2
        
    # Since our streaming architecture strictly processes 28800 samples (600ms) at a time,
    # the corresponding `p_len` in RVC is always exactly 60 frames. 
    # Baking a static 60-frame size into the ONNX graph allows PyTorch
    # to bypass Dynamo shape constraint violations.
    dummy_feats = torch.randn(1, 60, feat_dim).to(device)
    if is_half: dummy_feats = dummy_feats.half()
    
    dummy_p_len = torch.tensor([60], dtype=torch.long).to(device)
    dummy_sid = torch.tensor([0], dtype=torch.long).to(device)
    
    inputs = (dummy_feats, dummy_p_len, dummy_sid)
    input_names = ["feats", "p_len", "sid"]
    
    if model_info["f0"]:
        dummy_pitch = torch.ones(1, 60, dtype=torch.long).to(device)
        dummy_pitchf = torch.randn(1, 60).to(device)
        if is_half: dummy_pitchf = dummy_pitchf.half()
        inputs = (dummy_feats, dummy_p_len, dummy_pitch, dummy_pitchf, dummy_sid)
        input_names = ["feats", "p_len", "pitch", "pitchf", "sid"]

    # Direct method hooking: mock the training 'forward' method with
    # the voice conversion 'infer' method to evade PyTorch's signature checker.
    # This prevents the 'missing y_lengths' training crash.
    original_forward = model.forward
    model.forward = model.infer
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs,
                export_path,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                input_names=input_names,
                output_names=["audio"],
                opset_version=17,
            )
    finally:
        # Restore just in case the model is reused for torch.compile
        model.forward = original_forward


# ═══════════════════════════════════════════════════════════
# Full Inference Pipeline
# ═══════════════════════════════════════════════════════════

def infer(audio_np, sr, model, model_info, f0_up_key=0, index=None, big_npy=None, index_rate=0.75, rms_mix_rate=0.2):
    t0 = time()
    tgt_sr = model_info["tgt_sr"]
    version = model_info["version"]

    # Resample to 16kHz
    audio_16k = librosa.resample(audio_np, orig_sr=sr, target_sr=16000) if sr != 16000 else audio_np.copy()
    
    # Bypassing the extreme 16kHz high-pass filter for voice clarity
    # audio_16k = signal.filtfilt(bh, ah, audio_16k).astype(np.float32)
    audio_16k = audio_16k.astype(np.float32)

    # Pad
    t_pad = int(SR * X_PAD)
    t_pad_tgt = int(tgt_sr * X_PAD)
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
    phone_lengths = torch.tensor([p_len]).long().to(device)
    sid = torch.tensor([0]).long().to(device)
    pitch = torch.from_numpy(f0_coarse).long().unsqueeze(0).to(device) if model_info["f0"] else None
    
    pitchf = None
    if model_info["f0"]:
        pitchf = torch.from_numpy(f0_raw).unsqueeze(0).to(device)
        if is_half: pitchf = pitchf.half()
        else: pitchf = pitchf.float()

    # --- ONNX Static Shape Guard ---
    # Our generated ONNX graph is strictly baked to `60` frames to bypass
    # PyTorch Dynamo shape validation errors. If HuBERT returned slightly more or less 
    # frames due to STFT float overlaps (e.g. 59 or 61), we must normalize them to 60.
    if model_info.get("is_onnx", False):
        target_len = 60
        phone_lengths = torch.tensor([target_len]).long().to(device)
        
        if feats.shape[1] > target_len:
            feats = feats[:, :target_len, :]
        elif feats.shape[1] < target_len:
            pad_amt = target_len - feats.shape[1]
            feats = torch.nn.functional.pad(feats, (0, 0, 0, pad_amt))
            
        if pitch is not None:
            if pitch.shape[1] > target_len:
                pitch = pitch[:, :target_len]
                pitchf = pitchf[:, :target_len]
            elif pitch.shape[1] < target_len:
                pad_amt = target_len - pitch.shape[1]
                pitch = torch.nn.functional.pad(pitch, (0, pad_amt))
                pitchf = torch.nn.functional.pad(pitchf, (0, pad_amt))

    # Synthesis
    with torch.no_grad():
        t2_5 = time()
        if model_info.get("is_onnx", False):
            # ONNX Inference
            ort_inputs = {
                "feats": feats.cpu().numpy(),
                "p_len": phone_lengths.cpu().numpy(),
                "sid": sid.cpu().numpy()
            }
            if model_info["f0"]:
                ort_inputs["pitch"] = pitch.cpu().numpy()
                ort_inputs["pitchf"] = pitchf.cpu().numpy()
            
            try:
                audio_out = model.run(None, ort_inputs)[0][0, 0]
                audio_out = torch.from_numpy(audio_out)
            except Exception as e:
                # If ONNX shape somehow still mismatches, fallback to avoid total silence
                print(f"[ONNX Error] {e} - shapes feats:{ort_inputs['feats'].shape} pitch:{ort_inputs['pitch'].shape}")
                return None
        else:
            # PyTorch Inference
            if pitch is not None:
                audio_out = model.infer(feats, phone_lengths, pitch, pitchf, sid)[0][0, 0]
            else:
                audio_out = model.infer(feats, phone_lengths, sid)[0][0, 0]
    t3 = time()

    # Evaluate raw output
    result = audio_out.data.cpu().float().numpy()

    # The ONNX model natively drops ~2 edge frames (20ms) inside its convolutions.
    # We must NEVER use arbitrary back-slicing (:-t_pad_tgt) because the missing frames
    # cause it to slice deeply into the user's actual audio, deleting 20ms of real speech!
    # Instead, we strip the exact front padding, and take EXACTLY the mathematical duration we need.
    expected_len = int(np.round(len(audio_np) * tgt_sr / sr))
    
    if len(result) > t_pad_tgt:
        # Strip exact front padding, and grab only the guaranteed audio duration
        start_idx = int(t_pad_tgt)
        result = result[start_idx : start_idx + expected_len]
        
    # Strictly pad the deficit (if the model catastrophically failed) 
    # instead of doing a trailing crop that deletes the middle of words
    if len(result) < expected_len:
        result = np.pad(result, (0, expected_len - len(result)))

    # RMS ENVELOPE MATCHING
    if rms_mix_rate > 0:
        win_len = SR // 200
        rms_in = librosa.feature.rms(y=audio_np, frame_length=win_len*2, hop_length=win_len)[0]
        rms_out = librosa.feature.rms(y=result, frame_length=win_len*2, hop_length=win_len)[0]
        
        rms_in_interp = np.interp(np.arange(len(result)), np.linspace(0, len(result), len(rms_in)), rms_in)
        rms_out_interp = np.interp(np.arange(len(result)), np.linspace(0, len(result), len(rms_out)), rms_out)
        
        rms_in_interp = np.maximum(rms_in_interp, 1e-5)
        rms_out_interp = np.maximum(rms_out_interp, 1e-4) # Slightly higher floor for output to prevent massive multipliers
        
        scale = rms_in_interp / rms_out_interp
        # CRITICAL FIX: Limit the maximum scale multiplier to 2.5x to prevent
        # near-silent background room noise from exploding into deafening roar spikes!
        scale = np.clip(scale, 0.0, 2.5) 
        
        scale = scale * rms_mix_rate + (1.0 - rms_mix_rate)
        result = result * scale
        
        # Apply scaling
        result = result * scale
        
    # Trim padding and audio scaling
        
    audio_max = np.abs(result).max() / 0.99
    if audio_max > 1: result = result / audio_max

    print(f"    F0={t1-t0:.1f}s HuBERT={t2-t1:.1f}s Synth={t3-t2:.1f}s Total={t3-t0:.1f}s ({len(result)} @ {tgt_sr}Hz)")
    return result, tgt_sr
