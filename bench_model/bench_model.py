#!/usr/bin/env python3
"""Generate a random-weight HF model with real model shapes (bf16 safetensors).

    python3 tools/bench_model.py OUT_DIR [qwen05b|qwen15b|llama1b]

Requires torch, transformers
"""

import os
import sys

import torch
import transformers as tf

SHAPES = {
    "qwen05b": (tf.Qwen2Config, tf.Qwen2ForCausalLM, dict(
        vocab_size=151936, hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
        num_attention_heads=14, num_key_value_heads=2, tie_word_embeddings=True, rope_theta=1e6,
        max_position_embeddings=32768, rms_norm_eps=1e-6)),
    "qwen15b": (tf.Qwen2Config, tf.Qwen2ForCausalLM, dict(
        vocab_size=151936, hidden_size=1536, intermediate_size=8960, num_hidden_layers=28,
        num_attention_heads=12, num_key_value_heads=2, tie_word_embeddings=True, rope_theta=1e6,
        max_position_embeddings=32768, rms_norm_eps=1e-6)),
    "llama1b": (tf.LlamaConfig, tf.LlamaForCausalLM, dict(
        vocab_size=128256, hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=32, num_key_value_heads=8, tie_word_embeddings=True, rope_theta=500000.0,
        max_position_embeddings=131072, rms_norm_eps=1e-5,
        rope_scaling=dict(rope_type="llama3", factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0,
                          original_max_position_embeddings=8192))),
}

out = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "qwen05b"
Cfg, Mdl, kw = SHAPES[name]
cfg = Cfg(**kw)
with torch.device("meta"):
    pass
torch.set_default_dtype(torch.bfloat16)
m = Mdl(cfg)
with torch.no_grad():
    for n, p in m.named_parameters():
        if "norm" in n:
            p.fill_(1.0)
        else:
            p.normal_(0, 0.02)
os.makedirs(out, exist_ok=True)
m.save_pretrained(out, safe_serialization=True)
print("params:", sum(p.numel() for p in m.parameters()) / 1e6, "M ->", out)
