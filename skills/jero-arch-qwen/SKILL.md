---
name: jero-arch-qwen
description: Qwen 2/3/3.5 MoE, Qwen3-Embedding, and DeepSeek2 Multi-Head Latent Attention (MLA) architecture recipes (`QwenRecipe`, `Qwen2Recipe`, `Qwen3Recipe`, `Qwen3MoeRecipe`, `Qwen35Recipe`, `Qwen35MoeRecipe`, `QwenEmbeddingRecipe`, `DeepSeek2Recipe`, `DeepSeek2MlaRecipe`) for the SpacemiT K3 platform.
---

# `jero-arch-qwen` Architecture Crate

Provides modular `ArchRecipe` implementations and shared GQA / SwiGLU / MoE / MLA building blocks across:

- **`qwen2` (`QwenRecipe`, `Qwen2Recipe`, `Qwen3Recipe`)**: Strict ADR-023 separation (`qwen2` QKV bias vs `qwen3` per-head QK-norm without QKV bias), tied `output.weight` -> `token_embd.weight` / `model.embed_tokens.weight` aliasing, padded LM-head row slicing, and Layer 0 `soft_prompt_embeds` injection for `Qwen2-VL` / `Qwen2.5-VL` prefixes.
- **`qwen3_moe` (`Qwen3MoeRecipe`, `dispatch_qwen3_moe_ffn`)**: Fine-grained sparse top-`k` MoE routing with deterministic ascending tie-breaking (`select_top_k_experts`) over 3D stacked or per-expert tensors.
- **`deepseek2_mla` (`DeepSeek2MlaConfig`, `DeepSeek2Recipe`, `DeepSeek2MlaRecipe`, `run_mla_absorbed_attention`)**: Multi-Head Latent Attention (`MLA`) matrix absorption over compressed `[c_kv, k_rope]` latent caches (`q_lora_rank: None` and `Some(rank)`), optional `exp_probs_b.bias` routing, and shared + routed experts.
- **`qwen35` (`Qwen35Recipe`, `Qwen35MoeRecipe`)**: 3:1 interleaved `Gated-DeltaNet` linear recurrence and full `QK-norm` GQA layers with dense or shared + routed MoE FFN.
- **`embedding` (`QwenEmbeddingRecipe`, `"qwen3-embedding"`)**: Causal `0..num_layers` transformer execution without LM-head allocation (`requires_kv_cache() == false`), pooling via `EmbeddingPooling::LastToken` (or `ctx.embedding_pooling()`) + optional Matryoshka truncation + L2 normalization via `jero_kernels_rvv::pool_and_l2_normalize_f32` directly into `ctx.embedding_output_mut()`.
- **`common` (`DenseAttnLayerSpec`, `matvec_any`, `rmsnorm_standard`, `run_dense_attention_sublayer_scaled`, `run_dense_swiglu_ffn`, `is_supported_quant`, `validate_qwen_revision_from_probe`)**: Shared fail-closed tensor hydration supporting both `GgufFile` and `SafeTensorsFile` (`ModelWeights`), with zero synthetic fallbacks per ADR-045.

## Crate Contract & Dependencies

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (zero-copy `ModelWeights`, `SafeTensorsFile`, `GgufFile`, `ExecutionContext`, `ArchRecipe`, `K3Error`), `rayon` (`1.12`), and optional `jero-kernels-rvv` (enabled by default via `with-kernels`).
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo check --target riscv64gc-unknown-linux-gnu --all-targets
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
cargo agents plugin validate Symposium.toml --no-check-crates
```
