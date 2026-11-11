# Porting

The inference library is `no_std` and requires a global allocator, `libm` (or equivalent math implementations), and a 64-byte-aligned model buffer.

The `nqm-run` frontend uses libc for file I/O, memory allocation and output. Its OS-specific code is isolated in `src/sys.rs`.

For systems without a compatible libc, the inference library can be integrated directly using a custom entry point and platform-specific I/O.

```bash
cargo build --release --target <target>
```

Additional requirements:

- `panic = "abort"`.
- Little-endian target architecture.
- Memory for the model file, KV cache and runtime allocations.