"""
Standalone HuBERT model for loading RVC's hubert_base.pt
Uses fairseq-compatible naming so weights load directly.
No fairseq dependency required.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ═══════════════════════════════════════════
# CNN Feature Extractor
# ═══════════════════════════════════════════

class ConvLayerBlock0(nn.Module):
    """Layer 0: Conv + Dropout + GroupNorm, indices 0, 1, 2"""
    def __init__(self):
        super().__init__()
        self._modules["0"] = nn.Conv1d(1, 512, 10, 5, bias=True)
        self._modules["1"] = nn.Dropout(0)
        self._modules["2"] = nn.GroupNorm(1, 512, affine=True)

    def forward(self, x):
        x = self._modules["0"](x)
        x = self._modules["1"](x)
        x = F.gelu(self._modules["2"](x))
        return x


class ConvLayerBlockN(nn.Module):
    """Layers 1-6: Conv only (no bias, no norm in checkpoint)"""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self._modules["0"] = nn.Conv1d(512, 512, kernel_size, stride, bias=False)

    def forward(self, x):
        x = F.gelu(self._modules["0"](x))
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 7 conv layers matching fairseq naming:
        # conv_layers.0: Conv1d(1,512,10,5) + Dropout + GroupNorm
        # conv_layers.1-4: Conv1d(512,512,3,2)
        # conv_layers.5-6: Conv1d(512,512,2,2)
        self.conv_layers = nn.ModuleList([
            ConvLayerBlock0(),
            ConvLayerBlockN(3, 2),
            ConvLayerBlockN(3, 2),
            ConvLayerBlockN(3, 2),
            ConvLayerBlockN(3, 2),
            ConvLayerBlockN(2, 2),
            ConvLayerBlockN(2, 2),
        ])

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T)
        for layer in self.conv_layers:
            x = layer(x)
        return x  # (B, 512, T')


# ═══════════════════════════════════════════
# Transformer Components
# ═══════════════════════════════════════════

class SelfAttention(nn.Module):
    """fairseq naming: self_attn.{k,q,v,out}_proj"""
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    """fairseq naming: self_attn, self_attn_layer_norm, fc1, fc2, final_layer_norm"""
    def __init__(self, embed_dim=768, ffn_dim=3072, num_heads=12):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None):
        # Post-norm architecture (used by fairseq hubert_base)
        residual = x
        x = self.self_attn(x, padding_mask)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class Encoder(nn.Module):
    """fairseq naming: pos_conv.0 (weight_norm), layer_norm, layers.X"""
    def __init__(self, embed_dim=768, num_layers=12, ffn_dim=3072, num_heads=12):
        super().__init__()
        # pos_conv: plain conv (weights will be loaded manually to bypass weight_norm incompatibility)
        self.pos_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 128, padding=64, groups=16),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, ffn_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x, padding_mask=None, output_layer=12):
        # Positional encoding
        pos = self.pos_conv(x.transpose(1, 2))
        pos = pos[:, :, :x.size(1)]
        x = x + pos.transpose(1, 2)
        x = self.layer_norm(x)

        # Transformer layers
        all_layers = []
        for i, layer in enumerate(self.layers):
            x = layer(x, padding_mask)
            all_layers.append(x)
            if output_layer is not None and i + 1 == output_layer:
                break
        return all_layers


# ═══════════════════════════════════════════
# Full HuBERT Model
# ═══════════════════════════════════════════

class HuBERTModel(nn.Module):
    def __init__(self, embed_dim=768, num_layers=12, ffn_dim=3072, num_heads=12):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.layer_norm = nn.LayerNorm(512)  # Normalizes CNN output (512-dim)
        self.post_extract_proj = nn.Linear(512, embed_dim)
        self.mask_emb = nn.Parameter(torch.zeros(embed_dim))
        self.encoder = Encoder(embed_dim, num_layers, ffn_dim, num_heads)
        self.final_proj = nn.Linear(embed_dim, 256)

    def extract_features(self, source, padding_mask=None, output_layer=12):
        """Extract features matching fairseq's interface."""
        features = self.feature_extractor(source)    # (B, 512, T')
        features = features.transpose(1, 2)          # (B, T', 512)
        features = self.layer_norm(features)         # Norm 512-dim
        features = self.post_extract_proj(features)  # (B, T', 768)
        layer_outputs = self.encoder(features, padding_mask, output_layer)
        return layer_outputs, None


def load_hubert_from_checkpoint(checkpoint_path):
    """Load HuBERT from RVC's hubert_base.pt."""
    from fairseq_stubs import install_stubs
    install_stubs()

    cpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    weights = cpt["model"]

    # Filter out keys we don't need
    weights = {k: v for k, v in weights.items() if k != "label_embs_concat"}

    # Reconstruct pos_conv weight from weight_norm decomposition
    # fairseq stores: pos_conv.0.weight_g [1,1,128] and pos_conv.0.weight_v [768,48,128]
    if "encoder.pos_conv.0.weight_g" in weights and "encoder.pos_conv.0.weight_v" in weights:
        weight_g = weights.pop("encoder.pos_conv.0.weight_g")  # [1, 1, 128]
        weight_v = weights.pop("encoder.pos_conv.0.weight_v")  # [768, 48, 128]
        # Reconstruct: weight = weight_g * weight_v / ||weight_v||
        norm = torch.norm(weight_v, dim=(0, 1), keepdim=True)  # [1, 1, 128]
        full_weight = weight_g * weight_v / norm
        weights["encoder.pos_conv.0.weight"] = full_weight

    model = HuBERTModel()
    missing, unexpected = model.load_state_dict(weights, strict=False)

    loaded = len(weights) - len(unexpected)
    total = len(weights)
    print(f"  HuBERT: {loaded}/{total} weights loaded")
    if unexpected:
        print(f"  Skipped: {unexpected[:3]}...")

    model.eval()
    return model
