import sys; sys.path.insert(0, '.')
import torch
import torchaudio
from torch.nn.utils.parametrize import remove_parametrizations

def load_rvc_hubert_to_torchaudio(checkpoint_path):
    # 1. Load torchaudio model
    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model()
    
    # 2. Remove weight_norm parametrization from pos_conv so we can load plain reconstructed weight
    try:
        remove_parametrizations(model.encoder.transformer.pos_conv_embed.conv, "weight")
    except Exception:
        pass

    # 3. Load fairseq weights
    from fairseq_stubs import install_stubs; install_stubs()
    cpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    fs_weights = cpt['model']
    fs_weights = {k: v for k, v in fs_weights.items() if k != "label_embs_concat"}

    # 4. Reconstruct pos_conv weight
    if "encoder.pos_conv.0.weight_g" in fs_weights:
        wg = fs_weights.pop("encoder.pos_conv.0.weight_g")
        wv = fs_weights.pop("encoder.pos_conv.0.weight_v")
        norm = torch.norm(wv, dim=(0, 1), keepdim=True)
        fs_weights["encoder.pos_conv.0.weight"] = wg * wv / norm

    # 5. Map keys
    new_state = {}
    for k, v in fs_weights.items():
        nk = k
        if k == "feature_extractor.conv_layers.0.2.weight": nk = "feature_extractor.conv_layers.0.layer_norm.weight"
        elif k == "feature_extractor.conv_layers.0.2.bias": nk = "feature_extractor.conv_layers.0.layer_norm.bias"
        elif k == "layer_norm.weight": nk = "encoder.feature_projection.layer_norm.weight"
        elif k == "layer_norm.bias": nk = "encoder.feature_projection.layer_norm.bias"
        elif k == "post_extract_proj.weight": nk = "encoder.feature_projection.projection.weight"
        elif k == "post_extract_proj.bias": nk = "encoder.feature_projection.projection.bias"
        elif k == "encoder.layer_norm.weight": nk = "encoder.transformer.layer_norm.weight"
        elif k == "encoder.layer_norm.bias": nk = "encoder.transformer.layer_norm.bias"
        elif k == "encoder.pos_conv.0.bias": nk = "encoder.transformer.pos_conv_embed.conv.bias"
        elif k == "encoder.pos_conv.0.weight": nk = "encoder.transformer.pos_conv_embed.conv.weight"
        elif k.startswith("feature_extractor.conv_layers."):
            # e.g. feature_extractor.conv_layers.1.0.weight -> feature_extractor.conv_layers.1.conv.weight
            nk = k.replace(".0.weight", ".conv.weight").replace(".0.bias", ".conv.bias")
        elif k.startswith("encoder.layers."):
            nk = k.replace("encoder.layers.", "encoder.transformer.layers.")
            nk = nk.replace(".self_attn.", ".attention.")
            nk = nk.replace(".self_attn_layer_norm.", ".layer_norm.")
            nk = nk.replace(".fc1.", ".feed_forward.intermediate_dense.")
            nk = nk.replace(".fc2.", ".feed_forward.output_dense.")
        
        new_state[nk] = v

    # 6. Load mapped weights
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    
    loaded = len(new_state) - len(unexpected)
    print(f"  Torchaudio HuBERT mapped: {loaded}/{len(new_state)} weights")
    
    # 7. Extract final_proj for v1 models (torchaudio doesn't have this layer built-in)
    import torch.nn as nn
    final_proj = None
    if "final_proj.weight" in cpt["model"]:
        w = cpt["model"]["final_proj.weight"]
        b = cpt["model"]["final_proj.bias"]
        final_proj = nn.Linear(w.shape[1], w.shape[0])
        final_proj.weight.data = w
        final_proj.bias.data = b
        final_proj.eval()

    model.eval()
    return model, final_proj

if __name__ == "__main__":
    m = load_rvc_hubert_to_torchaudio("hubert_cache/hubert_base.pt")
    # Test random input - torchaudio expects 2D (batch, time)
    x = torch.randn(1, 16000)
    out, _ = m.extract_features(x)
    f = out[-1]
    print(f"Random output shape: {f.shape}, mean: {f.mean():.4f}")
