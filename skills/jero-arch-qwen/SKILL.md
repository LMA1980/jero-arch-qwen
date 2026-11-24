---
name: jero-arch-qwen
description: Qwen 2/3/3.5 MoE, Qwen3-Embedding, and DeepSeek2 Multi-Head Latent Attention (MLA) architecture recipes (`QwenRecipe`, `Qwen3MoeRecipe`, `QwenEmbeddingRecipe`, `DeepSeek2MlaRecipe`) for the SpacemiT K3 platform.
---

# `jero-arch-qwen` Architecture Crate

Provides Qwen 2 / 3 / 3.5 MoE (`QwenRecipe`, `Qwen3MoeRecipe`), dense embedding (`QwenEmbeddingRecipe`), DeepSeek2 MLA (`DeepSeek2MlaRecipe`), and `soft_prompt_embeds` support for K3 inference.

## Crate Contract & Dependencies

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (zero-copy `ModelWeights`, `ExecutionContext`, `ArchRecipe`, `K3Error`) and optional `jero-kernels-rvv` (enabled by default via `with-kernels`).
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
```
