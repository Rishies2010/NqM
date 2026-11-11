# NQM1 File Format

## File layout

NQM1 files contain a 512-byte header, followed by the tokenizer and weights sections. All integer fields are little-endian.

The tokenizer and weights sections are aligned to 64-byte boundaries. Their offsets and the total file size are stored in the header.

```text
Offset                   Section
--------------------------------------
0                   Header (512 bytes)
off_tokenizer           Tokenizer
off_weights              Weights
```

## Header

The header consists of 128 32-bit words.

| Word | Field | Description |
|---:|---|---|
| 0 | `magic` | `0x314D514E` (`NQM1`) |
| 1 | `version` | Format version (`1`) |
| 2 | `header_bytes` | Header size (`512`) |
| 3 | `flags` | Bit 0: tied embeddings; bit 1: QKV bias; bit 2: QK normalization |
| 4 | `dim` | Model width |
| 5 | `hidden_dim` | FFN width |
| 6 | `n_layers` | Number of layers |
| 7 | `n_heads` | Number of query heads |
| 8 | `n_kv_heads` | Number of key/value heads |
| 9 | `head_dim` | Attention head dimension |
| 10 | `vocab_size` | Vocabulary size |
| 11 | `max_seq_len` | Maximum sequence length |
| 12 | `rope_theta` | RoPE base frequency (`f32`) |
| 13 | `norm_eps` | Normalization epsilon (`f32`) |
| 14 | `rope_scaling` | `0`: none; `1`: Llama 3 |
| 15 | `rope_factor` | RoPE scaling factor (`f32`) |
| 16 | `rope_low_freq_factor` | Low-frequency scaling factor (`f32`) |
| 17 | `rope_high_freq_factor` | High-frequency scaling factor (`f32`) |
| 18 | `rope_orig_ctx` | Original context length (`original_max_position_embeddings`) |
| 19 | `quant_body` | `0`: F32; `1`: Q8; `2`: Q4 |
| 20 | `quant_embed` | `0`: F32; `1`: Q8 |
| 21 | `group_size_body` | Quantization group size for body matrices |
| 22 | `group_size_embed` | Quantization group size for embeddings and output head |
| 23 | `pretok_kind` | `0`: GPT-2; `1`: Llama 3; `2`: Qwen2; `3`: GPT-2 with split digits |
| 24 | `tok_flags` | Bit 0: `ignore_merges` |
| 25 | `n_merges` | Number of BPE merges |
| 26 | `blob_bytes` | Tokenizer blob size in bytes |
| 27 | `bos_id` | BOS token ID (`-1`: none) |
| 28–31 | `stop_ids[4]` | Stop token IDs (`-1`: unused) |
| 32 | `chat_kind` | `0`: none; `1`: Llama 3; `2`: ChatML |
| 33–36 | `tpl_ids[4]` | Template token IDs |
| 37 | Reserved | Zero |
| 38–39 | `off_tokenizer` | Tokenizer offset (`u64`) |
| 40–41 | `off_weights` | Weights offset (`u64`) |
| 42–43 | `file_size` | Total file size (`u64`) |
| 44–51 | `name` | Model name (32 bytes, NUL-padded) |
| 52–127 | Reserved | Zero |

The `tpl_ids` fields contain the following IDs:

- Llama 3: `start_header`, `end_header`, `eot`.
- ChatML: `im_start`, `im_end`.

## Tokenizer

The tokenizer section uses byte-level BPE.

```c
u32 token_len[vocab_size];
u8  blob[blob_bytes];
u32 merges[n_merges][3];
```

`token_len` contains the byte length of each token. A length of zero indicates an unused token ID.

`blob` contains the concatenated token bytes. Token data is stored as raw bytes, with the GPT-2 bytes-to-Unicode mapping already reversed by the converter.

Each merge entry contains three token IDs:

```text
left_id
right_id
merged_id
```

The array order defines BPE merge rank.

Special tokens are stored as literal text.

The tokenizer blob is padded to a 4-byte boundary. The merge table is padded to a 64-byte boundary.

## Weights

Weights are stored in a fixed order. Matrices have no names or individual headers in the file.

### Matrix representation

A matrix is defined by its dimensions, quantization type, and group size.

**Matrix data:**

| Type | Storage |
|---|---|
| F32 | `rows × cols × 4` bytes |
| Q8 | `rows × cols` bytes (`i8`) |
| Q4 | `rows × cols / 2` bytes |

Matrix data is padded to a 64-byte boundary.

**Quantization scales:**

Q8 and Q4 matrices store an additional array of F32 scales:

```c
f32 scales[rows * (cols / group_size)];
```

The scale array is also padded to a 64-byte boundary. F32 matrices do not contain scales.

### Quantization

**Q8**

Weights are stored as signed 8-bit integers. Each group uses one scale, calculated from its absolute maximum:

```text
scale = absmax / 127
w = i8 * scale
```

Groups are formed along each matrix row.

**Q4**

Two weights are packed into each byte:

- Low nibble: even column.
- High nibble: odd column.

Each stored nibble represents `q + 8`, where `q` is in the range `[-7, 7]`.

```text
scale = absmax / 7
w = q * scale
```

Groups are formed along each matrix row.

### Vectors

Norms and biases are stored as F32 arrays. Each vector is padded to a 64-byte boundary.

### Tensor order

The weights section uses the following tensor order:

```text
embed             matrix(vocab_size, dim)

for each layer:
    attn_norm     f32[dim]

    wq            matrix(n_heads * head_dim, dim)
    wk            matrix(n_kv_heads * head_dim, dim)
    wv            matrix(n_kv_heads * head_dim, dim)
    wo            matrix(dim, n_heads * head_dim)

    [bq, bk, bv]  f32 vectors (if flags.bias)
    [q_norm]      f32[head_dim] (if flags.qk_norm)
    [k_norm]      f32[head_dim] (if flags.qk_norm)

    ffn_norm      f32[dim]

    w_gate        matrix(hidden_dim, dim)
    w_up          matrix(hidden_dim, dim)
    w_down        matrix(dim, hidden_dim)

final_norm        f32[dim]
lm_head           matrix(vocab_size, dim) (if embeddings are untied)
```

The embedding matrix and `lm_head` use `quant_embed`. All other matrices use `quant_body`.

When embeddings are tied, `lm_head` is omitted.

## RoPE

NQM1 uses the Hugging Face rotate-half layout for RoPE. Each head is divided into two halves, with corresponding elements at indices `i` and `i + head_dim / 2` forming a rotation pair.

Hugging Face weights using this layout require no permutation during conversion.