# NqM

Tiny LLM inference for stripped / bare environments, written in **RUST**.
Fully CPU-based inference, and only requires `no_std` + `alloc`.
The engine requires a heap alloc, libm, file access (R/W), and text input/output.

### Supported LLM models:

NqM supports models using the following architectures:
- Llama 3.x
- SmolLM2
- Qwen2 and Qwen2.5
- Qwen2.5-Coder
- Qwen3

Models around 1.5B parameter range are within its intended scope. Larger models are also possible, subject to CPU performance and available memory. They are converted and stored in a custom .`nqm` file containing the configuration, tokenizer, and weights.

Supported weight formats are INT8, INT4, and F32. (The format is documented in docs/FORMAT.md.)

### External dependencies:

- libc: process startup, memory allocation, file I/O (malloc, open, read, write)
- libm: mathematical functions (Implement expf, sqrtf, sinf, cosf, powf)

## Quick start

Fetch a Hugging Face model directory (config.json, tokenizer.json, *.safetensors)
**Example :** `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`...
Here, **`Qwen2.5-1.5B-Instruct`** is taken as example (not shipped with repo).

Build:
```bash
cargo build --release
# optional, faster on build machine (not portable):
# RUSTFLAGS="-C target-cpu=native" cargo build --release

python3 tools/convert.py ./Qwen2.5-1.5B-Instruct qwen1.5b.nqm --quant q8
python3 tools/convert.py ./Qwen2.5-1.5B-Instruct qwen1.5b-q4.nqm --quant q4
```

Run:

```bash
./target/release/nqm-run qwen1.5b.nqm --chat --system "You are a concise assistant."
./target/release/nqm-run qwen1.5b.nqm --ask "Write a C function that reverses a string" -t 0.2
./target/release/nqm-run qwen1.5b.nqm -p "def fib(n):" -n 120 -t 0 # raw completion
```

Estimated sizes (formula: q8/g64 = 1.06 B per weight, q4/g32 = 0.63 B per weight, embeddings always q8):
KV cache is `2 * layers * ctx * kv_heads * head_dim * 4` bytes (Qwen2.5-1.5B at 2048 ctx: ~117 MB).

## What it does not do (as of now)

* Only byte-level BPE tokenizers (GPT-2 / Llama 3 / Qwen / SmolLM2). Not SentencePiece (Llama 2, TinyLlama,
  Mistral), Gemma/Phi. The converter refuses unknown tokenizers.
* No Unicode NFC normalisation : non-NFC input tokenizes differently from HF but still decodes to same text.
* No sliding-window attention, no MoE, no vision, no rope scaling other than `llama3`.
* Single thread, prompt processed one token at a time, context is a fixed KV cache (no shifting).
* Little-endian targets only.

Porting to another OS: [docs/PORTING.md](docs/PORTING.md). Format: [docs/FORMAT.md](docs/FORMAT.md).
Development notes: [docs/DEVLOG.md](docs/DEVLOG.md).

## Roadmap

1. int8 activations + integer dot products (VNNI / `pmaddubsw` / NEON `sdot`), the f32 path stays as the oracle
2. threads behind a tiny `parallel_for` hook in the platform layer
3. batched prefill (matrix-matrix) so long prompts are not 1 token at a time
4. f16 KV cache, sliding context
5. SentencePiece tokenizer (Llama 2 / TinyLlama / Mistral)

## License

<a href="https://opensource.org/license/mit">
  <img align="right" height="72" alt="MIT License" src="https://universoabierto.org/wp-content/uploads/2017/06/mit_license_logo_by_excaliburzero-d9ur2lg.png" />
</a>

NqM is licensed under the **MIT License**.

The full license text is available [here](https://opensource.org/license/mit) and is included in the [`LICENSE`](LICENSE) file in the repository.
