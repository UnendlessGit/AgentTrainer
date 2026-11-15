"""Dev-only parity reference: runs Pixel2Play's own PyTorch modules on fixed inputs.

Usage: python reference.py <open-p2p checkout> <checkpoint.ckpt> <out.safetensors> [steps]
Writes inputs (frames, actions, text) and P2P's logits so the MLX port can be compared.
"""
import sys
import numpy as np
import torch

p2p_dir, ckpt_path, out_path = sys.argv[1:4]
T = int(sys.argv[4]) if len(sys.argv) > 4 else 6
sys.path.insert(0, p2p_dir)

import torchvision  # noqa: E402

_effnet = torchvision.models.efficientnet_b0
torchvision.models.efficientnet_b0 = lambda dropout=0.2, weights=None: _effnet(dropout=dropout, weights=None)

from elefant.im_tokenizer.config import ImageTokenizerConfig, ConvTokenizerConfig  # noqa: E402
from elefant.im_tokenizer.conv_tokenizer import ConvTokenizer  # noqa: E402
from elefant.policy_model.policy_transformer import PolicyCausalTransformer, PolicyCausalTransformerConfig  # noqa: E402
from elefant.policy_model.action_decoder import ActionDecoderConfig  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
D = sd["bc_transformer.text_embed_mlp.weight"].shape[0]
L = len({k.split(".")[3] for k in sd if k.startswith("bc_transformer._transformer.transformer_layers.")})

tok = ConvTokenizer(ImageTokenizerConfig(type="conv", conv_tokenizer_config=ConvTokenizerConfig(num_tokens=1)), 192, 192, D)
cfg = PolicyCausalTransformerConfig(
    embed_dim=D, n_steps=T, n_transformer_layers=L, n_q_head=16, n_kv_head=16, mask_block_size=128,
    n_thinking_tokens=1, attention_history_len=[T] * L, model_type="dense",
    action_decoder=ActionDecoderConfig(embed_dim=D, n_action_tokens=9, input_action_token_dim=D),
    n_kv_sink_tokens=0, n_action_tokens=8, text_token_size=1, text_tokenizer_embed_dim=768, dropout=0.1,
)
policy = PolicyCausalTransformer(cfg, tok)
policy.load_state_dict({k[len("bc_transformer."):]: v for k, v in sd.items() if k.startswith("bc_transformer.")}, strict=True)
emb = {n: torch.nn.Embedding.from_pretrained(sd[f"{n}.weight"].float()) for n in
       ["key_action_embedding", "mouse_button_embedding", "mouse_delta_x_embedding", "mouse_delta_y_embedding"]}
heads = {}
for n in ["keyboard_out_logits", "mouse_button_out_logits", "mouse_delta_x_out_logits", "mouse_delta_y_out_logits"]:
    w = sd[f"{n}.weight"]
    lin = torch.nn.Linear(w.shape[1], w.shape[0])
    lin.load_state_dict({"weight": w, "bias": sd[f"{n}.bias"]})
    heads[n] = lin
policy = policy.float().eval()

g = torch.Generator().manual_seed(1234)
# Smooth-ish synthetic frames (low-frequency noise upsampled) so the vision stem sees structure.
low = torch.randint(0, 256, (1, T, 3, 12, 12), generator=g).float()
frames = torch.nn.functional.interpolate(low.view(T, 3, 12, 12), size=(192, 192), mode="bilinear", align_corners=False)
frames = frames.clamp(0, 255).round().to(torch.uint8).view(1, T, 3, 192, 192)
keys = torch.randint(0, 20, (1, T, 4), generator=g)
btn = torch.randint(0, 4, (1, T, 2), generator=g)
dx = torch.randint(0, 23, (1, T, 1), generator=g)
dy = torch.randint(0, 17, (1, T, 1), generator=g)
text = torch.randn(1, T, 768, generator=g) * 0.05
present = torch.ones(1, T, dtype=torch.bool)
present[:, : T // 2] = False
text[:, : T // 2] = 0

captured = {}
policy._transformer.transformer_layers[0].register_forward_pre_hook(lambda m, inp: (captured.setdefault("seq_in", inp[0]), None)[1])
policy._transformer.transformer_layers[-1].register_forward_hook(lambda m, inp, out: captured.update(seq_out=out[0]))
policy._transformer.transformer_layers[0].register_forward_hook(lambda m, inp, out: (captured.setdefault("layer0", out[0]), None)[1])
with torch.no_grad():
    a_in = torch.cat([emb["key_action_embedding"](keys), emb["mouse_button_embedding"](btn),
                      emb["mouse_delta_x_embedding"](dx), emb["mouse_delta_y_embedding"](dy)], dim=2)
    img_tokens = tok(frames.float() / 255.0)
    out, a_out_tokens, *_ = policy(frames.float() / 255.0, a_in, text.view(1, T, 1, 768))
    logits = {
        "keys": heads["keyboard_out_logits"](out[:, :, 0:4]),
        "pointer": heads["mouse_button_out_logits"](out[:, :, 4:6]),
        "x": heads["mouse_delta_x_out_logits"](out[:, :, 6:7]),
        "y": heads["mouse_delta_y_out_logits"](out[:, :, 7:8]),
    }

save_file({
    "frames": frames.permute(0, 1, 3, 4, 2).contiguous(),  # NTHWC for MLX
    "actions": torch.cat([keys, btn, dx, dy], dim=2).int().contiguous(),
    "text": text.contiguous(), "present": present.to(torch.uint8),
    "img_tokens": img_tokens.reshape(T, D).contiguous(), "a_out": a_out_tokens.reshape(T, D).contiguous(),
    **{f"logits_{k}": v.contiguous() for k, v in logits.items()},
    "seq_in": captured["seq_in"][0].contiguous(), "seq_out": captured["seq_out"][0].contiguous(),
    "layer0": captured["layer0"][0].contiguous(),
}, out_path)
print(f"wrote {out_path}: T={T} D={D} layers={L} keys0={logits['keys'][0, 0, 0, :5].tolist()}")
