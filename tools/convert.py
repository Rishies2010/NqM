#!/usr/bin/env python3
"""Convert a HF model dir to a .nqm file (see docs/FORMAT.md).

    python3 tools/convert.py MODEL_DIR OUT.nqm [--quant q8|q4|f32] [--group 64]

Supports Llama-architecture models (Llama 3.x, SmolLM2, ...), Qwen2 / Qwen2.5 / Qwen2.5-Coder,
and Qwen3, with a byte-level BPE tokenizer.json.
Requires numpy.
"""

import argparse
import glob
import json
import math
import os
import struct
import sys

import numpy as np

MAGIC = 0x314D514E
VERSION = 1
HEADER_BYTES = 512

QT = {"f32": 0, "q8": 1, "q4": 2}


PAT_LLAMA3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
PAT_QWEN2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
PRETOK_NAMES = {"gpt2": 0, "llama3": 1, "qwen2": 2, "gpt2digits": 3}


def die(msg):
    print(f"convert.py: error: {msg}", file=sys.stderr)
    sys.exit(1)




class SafeTensors:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.off = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.header.pop("__metadata__", None)

    def get(self, name):
        info = self.header[name]
        s, e = info["data_offsets"]
        raw = np.asarray(self.mm[self.off + s:self.off + e])
        dt = info["dtype"]
        if dt == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        elif dt == "F32":
            arr = raw.view(np.float32)
        else:
            die(f"tensor {name} has unsupported dtype {dt} (already quantised checkpoint?)")
        return arr.reshape(info["shape"])


class Weights:
    def __init__(self, model_dir):
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            die(f"no *.safetensors files in {model_dir}")
        self.where = {}
        self.files = []
        for p in files:
            st = SafeTensors(p)
            self.files.append(st)
            for k in st.header:
                self.where[k] = st

    def __contains__(self, k):
        return k in self.where

    def shape(self, k):
        return tuple(self.where[k].header[k]["shape"])

    def get(self, k):
        if k not in self.where:
            die(f"tensor {k} not found in checkpoint")
        return self.where[k].get(k)




def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def detect_pretok(tj, override):
    if override != "auto":
        return PRETOK_NAMES[override]
    pt = tj.get("pre_tokenizer")

    def flat(p):
        if p is None:
            return []
        if p["type"] == "Sequence":
            return [x for q in p["pretokenizers"] for x in flat(q)]
        return [p]

    parts = flat(pt)
    types = [p["type"] for p in parts]
    if types == ["ByteLevel"] and parts[0].get("use_regex", True) and not parts[0].get("add_prefix_space", False):
        return 0
    if (types == ["Digits", "ByteLevel"] and parts[0].get("individual_digits")
            and parts[1].get("use_regex", True) and not parts[1].get("add_prefix_space", False)):
        return 3
    if types == ["Split", "ByteLevel"] and not parts[1].get("use_regex", True) and not parts[1].get("add_prefix_space", False):
        sp = parts[0]
        pat = (sp.get("pattern") or {}).get("Regex")
        if sp.get("behavior") == "Isolated" and not sp.get("invert", False):
            if pat == PAT_LLAMA3:
                return 1
            if pat == PAT_QWEN2:
                return 2
            die("unknown pre-tokenizer regex:\n  " + str(pat) +
                "\nIf it matches one of gpt2/llama3/qwen2 in spirit, force it with --pretok.")
    die("unsupported pre_tokenizer layout: " + json.dumps(pt)[:300] + "\n(only byte-level BPE tokenizers are supported)")


def find_bos(tj):
    stack = [tj.get("post_processor")]
    while stack:
        p = stack.pop()
        if not p:
            continue
        if p.get("type") == "TemplateProcessing":
            for item in p.get("single", []):
                if "SpecialToken" in item:
                    return p["special_tokens"][item["SpecialToken"]["id"]]["ids"][0]
        if p.get("type") == "Sequence":
            stack.extend(p.get("processors", []))
    return -1


def build_tokenizer(model_dir, vocab_size, cfg, pretok_override):
    path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(path):
        die("tokenizer.json not found (SentencePiece-only tokenizers are not supported)")
    tj = json.load(open(path, encoding="utf-8"))
    m = tj["model"]
    if m["type"] != "BPE":
        die(f"tokenizer model type {m['type']} unsupported (need BPE)")
    norm = tj.get("normalizer")
    if norm is not None and norm.get("type") != "NFC":
        die("tokenizer normalizer other than NFC is not supported: " + json.dumps(norm)[:200])
    if norm is not None:
        print("note: tokenizer uses NFC normalisation; nqm does not normalise, so input that is not already NFC "
              "may tokenize differently from Hugging Face (still valid text).")
    pretok = detect_pretok(tj, pretok_override)
    dec = {v: k for k, v in bytes_to_unicode().items()}
    vocab = m["vocab"]
    toks = [b""] * vocab_size
    for s, i in vocab.items():
        if i >= vocab_size:
            die(f"token id {i} >= model vocab_size {vocab_size}")
        try:
            toks[i] = bytes(dec[c] for c in s)
        except KeyError:
            die(f"token {s!r} is not byte-level BPE (SentencePiece-style tokenizer?)")
    added = {}
    for a in tj.get("added_tokens", []):
        if a["id"] >= vocab_size:
            continue
        toks[a["id"]] = a["content"].encode("utf-8")
        added[a["content"]] = a["id"]
    merges = []
    for r in m["merges"]:
        a, b = r.split(" ") if isinstance(r, str) else r
        try:
            merges.append((vocab[a], vocab[b], vocab[a + b]))
        except KeyError:
            die(f"merge {a!r} + {b!r} refers to a token missing from the vocab")
    ignore_merges = bool(m.get("ignore_merges", False))

    def tid(name):
        return vocab.get(name, added.get(name))

    
    chat, tpl, chat_end = 0, [-1] * 4, None
    if tid("<|start_header_id|>") is not None and tid("<|eot_id|>") is not None and tid("<|end_header_id|>") is not None:
        chat, tpl, chat_end = 1, [tid("<|start_header_id|>"), tid("<|end_header_id|>"), tid("<|eot_id|>"), -1], tid("<|eot_id|>")
    elif tid("<|im_start|>") is not None and tid("<|im_end|>") is not None:
        chat, tpl, chat_end = 2, [tid("<|im_start|>"), tid("<|im_end|>"), -1, -1], tid("<|im_end|>")
    stops = []
    if chat_end is not None:
        stops.append(chat_end)
    gen = {}
    gp = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gp):
        gen = json.load(open(gp))
    for src in (cfg.get("eos_token_id"), gen.get("eos_token_id")):
        for e in (src if isinstance(src, list) else [src]):
            if e is not None and e not in stops:
                stops.append(e)
    stops = (stops + [-1] * 4)[:4]
    return dict(toks=toks, merges=merges, pretok=pretok, ignore_merges=ignore_merges,
                bos=find_bos(tj), stops=stops, chat=chat, tpl=tpl)




def choose_group(requested, dims):
    g = requested
    while g >= 8:
        if all(d % g == 0 for d in dims):
            return g
        g //= 2
    die(f"no group size >= 8 divides all dimensions {dims}")


def q8_block(w, gs):
    g = w.reshape(w.shape[0], -1, gs)
    scale = np.abs(g).max(axis=2) / 127.0
    safe = np.where(scale == 0, 1.0, scale)
    q = np.rint(g / safe[:, :, None]).clip(-127, 127).astype(np.int8)
    return q.reshape(w.shape).tobytes(), scale.astype("<f4")


def q4_block(w, gs):
    g = w.reshape(w.shape[0], -1, gs)
    scale = np.abs(g).max(axis=2) / 7.0
    safe = np.where(scale == 0, 1.0, scale)
    q = (np.rint(g / safe[:, :, None]).clip(-7, 7).astype(np.int16) + 8).astype(np.uint8).reshape(w.shape)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed.tobytes(), scale.astype("<f4")


class Out:
    def __init__(self, f):
        self.f = f

    def pad(self, a=64):
        self.f.write(b"\0" * ((-self.f.tell()) % a))

    def vec(self, arr):
        self.f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
        self.pad()

    def matrix(self, get, rows, cols, qt, gs):
        """get() returns a float32 (rows, cols) array. Written in row blocks to bound memory."""
        w = get()
        if w.shape != (rows, cols):
            die(f"unexpected tensor shape {w.shape}, expected {(rows, cols)}")
        w = np.ascontiguousarray(w, dtype=np.float32)
        if qt == "f32":
            self.f.write(w.astype("<f4").tobytes())
            self.pad()
            return
        scales = []
        step = max(1, (64 << 20) // (cols * 4))
        for r0 in range(0, rows, step):
            blk = w[r0:r0 + step]
            data, sc = (q8_block if qt == "q8" else q4_block)(blk, gs)
            self.f.write(data)
            scales.append(sc)
        self.pad()
        self.f.write(np.concatenate(scales).astype("<f4").tobytes())
        self.pad()




def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--quant", choices=["f32", "q8", "q4"], default="q8", help="attention + FFN weights")
    ap.add_argument("--embed-quant", choices=["f32", "q8"], default=None, help="embedding/lm_head (default q8; f32 if --quant f32)")
    ap.add_argument("--group", type=int, default=None, help="group size for body weights (default 64 for q8, 32 for q4)")
    ap.add_argument("--embed-group", type=int, default=64)
    ap.add_argument("--pretok", choices=["auto"] + list(PRETOK_NAMES), default="auto")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    mt = cfg.get("model_type")
    if mt not in ("llama", "qwen2", "qwen3"):
        die(f"model_type {mt!r} is not supported (llama, qwen2, qwen3 only)")
    if cfg.get("hidden_act", "silu") != "silu":
        die(f"hidden_act {cfg.get('hidden_act')} unsupported")
    if cfg.get("use_sliding_window"):
        die("sliding-window attention is not supported")
    W = Weights(args.model_dir)

    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    hidden = cfg["intermediate_size"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
    vocab_rows, dim = W.shape("model.embed_tokens.weight")
    if dim != cfg["hidden_size"]:
        die("embedding width != hidden_size")
    qd, kd = n_heads * head_dim, n_kv * head_dim
    tied = "lm_head.weight" not in W
    bias = "model.layers.0.self_attn.q_proj.bias" in W
    qkn = "model.layers.0.self_attn.q_norm.weight" in W
    for bad in ("self_attn.o_proj.bias", "mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias"):
        if f"model.layers.0.{bad}" in W:
            die(f"tensor {bad} present: attention_bias / mlp_bias on output projections is not supported")

    rp = cfg.get("rope_parameters") or {}
    theta = float(cfg.get("rope_theta", rp.get("rope_theta", 10000.0)))
    rs = cfg.get("rope_scaling") or (rp if rp.get("rope_type") not in (None, "default") else None)
    rope_kind, rope_f = 0, (1.0, 1.0, 4.0, 8192)
    if rs:
        rt = rs.get("rope_type") or rs.get("type")
        if rt == "llama3":
            rope_kind = 1
            rope_f = (float(rs["factor"]), float(rs["low_freq_factor"]), float(rs["high_freq_factor"]),
                      int(rs["original_max_position_embeddings"]))
        elif rt not in (None, "default"):
            die(f"rope scaling type {rt!r} unsupported (llama3 only)")

    qbody = args.quant
    qembed = args.embed_quant or ("f32" if qbody == "f32" else "q8")
    gs_body = choose_group(args.group or (32 if qbody == "q4" else 64), [dim, qd, hidden])
    gs_embed = choose_group(args.embed_group, [dim])
    if qbody == "f32":
        gs_body = 0
    if qembed == "f32":
        gs_embed = 0

    tk = build_tokenizer(args.model_dir, vocab_rows, cfg, args.pretok)
    name = (args.name or os.path.basename(os.path.normpath(args.model_dir)))[:31]

    print(f"{name}: {mt}, {n_layers} layers, dim {dim}, heads {n_heads}/{n_kv} head_dim {head_dim}, hidden {hidden}, "
          f"vocab {vocab_rows}, tied={tied}, bias={bias}, qk_norm={qkn}, rope_theta={theta}, rope_scaling={'llama3' if rope_kind else 'none'}")
    print(f"tokenizer: pretok={tk['pretok']} merges={len(tk['merges'])} ignore_merges={tk['ignore_merges']} "
          f"bos={tk['bos']} stops={tk['stops']} chat={tk['chat']}")
    print(f"quant: body {qbody}(g{gs_body}) embed {qembed}(g{gs_embed})")

    with open(args.out, "wb") as f:
        o = Out(f)
        f.write(b"\0" * HEADER_BYTES)
        
        off_tok = f.tell()
        lens = np.array([len(t) for t in tk["toks"]], dtype="<u4")
        blob = b"".join(tk["toks"])
        f.write(lens.tobytes())
        f.write(blob)
        o.pad(4)
        if tk["merges"]:
            f.write(np.array(tk["merges"], dtype="<u4").tobytes())
        o.pad(64)
        off_w = f.tell()
        
        o.matrix(lambda: W.get("model.embed_tokens.weight"), vocab_rows, dim, qembed, gs_embed)
        for l in range(n_layers):
            p = f"model.layers.{l}."
            o.vec(W.get(p + "input_layernorm.weight"))
            o.matrix(lambda: W.get(p + "self_attn.q_proj.weight"), qd, dim, qbody, gs_body)
            o.matrix(lambda: W.get(p + "self_attn.k_proj.weight"), kd, dim, qbody, gs_body)
            o.matrix(lambda: W.get(p + "self_attn.v_proj.weight"), kd, dim, qbody, gs_body)
            o.matrix(lambda: W.get(p + "self_attn.o_proj.weight"), dim, qd, qbody, gs_body)
            if bias:
                for n in "qkv":
                    o.vec(W.get(p + f"self_attn.{n}_proj.bias"))
            if qkn:
                o.vec(W.get(p + "self_attn.q_norm.weight"))
                o.vec(W.get(p + "self_attn.k_norm.weight"))
            o.vec(W.get(p + "post_attention_layernorm.weight"))
            o.matrix(lambda: W.get(p + "mlp.gate_proj.weight"), hidden, dim, qbody, gs_body)
            o.matrix(lambda: W.get(p + "mlp.up_proj.weight"), hidden, dim, qbody, gs_body)
            o.matrix(lambda: W.get(p + "mlp.down_proj.weight"), dim, hidden, qbody, gs_body)
            print(f"\r  layer {l + 1}/{n_layers}", end="", flush=True)
        print()
        o.vec(W.get("model.norm.weight"))
        if not tied:
            o.matrix(lambda: W.get("lm_head.weight"), vocab_rows, dim, qembed, gs_embed)
        size = f.tell()

        
        h = bytearray(HEADER_BYTES)

        def sw(i, v):
            struct.pack_into("<I", h, 4 * i, v & 0xFFFFFFFF)

        def sf(i, v):
            struct.pack_into("<f", h, 4 * i, v)

        sw(0, MAGIC); sw(1, VERSION); sw(2, HEADER_BYTES)
        sw(3, (1 if tied else 0) | (2 if bias else 0) | (4 if qkn else 0))
        sw(4, dim); sw(5, hidden); sw(6, n_layers); sw(7, n_heads); sw(8, n_kv); sw(9, head_dim)
        sw(10, vocab_rows); sw(11, int(cfg.get("max_position_embeddings", 2048)))
        sf(12, theta); sf(13, float(cfg.get("rms_norm_eps", 1e-5)))
        sw(14, rope_kind); sf(15, rope_f[0]); sf(16, rope_f[1]); sf(17, rope_f[2]); sw(18, rope_f[3])
        sw(19, QT[qbody]); sw(20, QT[qembed]); sw(21, gs_body); sw(22, gs_embed)
        sw(23, tk["pretok"]); sw(24, 1 if tk["ignore_merges"] else 0)
        sw(25, len(tk["merges"])); sw(26, len(blob))
        sw(27, tk["bos"])
        for i in range(4):
            sw(28 + i, tk["stops"][i])
            sw(33 + i, tk["tpl"][i])
        sw(32, tk["chat"])
        for base, val in ((38, off_tok), (40, off_w), (42, size)):
            sw(base, val & 0xFFFFFFFF); sw(base + 1, val >> 32)
        h[176:176 + len(name.encode())] = name.encode()
        f.seek(0)
        f.write(h)
    print(f"wrote {args.out}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
