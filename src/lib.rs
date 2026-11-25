//! `jero-arch-qwen`: Alibaba Qwen (`qwen2`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`,
//! `qwen3-embedding`) and `DeepSeek` (`deepseek2`) execution recipes for the `SpacemiT` K3 platform.
//!
//! Enforces strict ADR-023 separation between [`Qwen2Recipe`] (`attn_{q,k,v}.bias`) and
//! [`Qwen3Recipe`] / [`Qwen3MoeRecipe`] / [`QwenEmbeddingRecipe`] (per-head
//! `attn_{q,k}_norm.weight` without QKV biases), ADR-045 fail-closed weight validation across both
//! `GgufFile` and `SafeTensorsFile`, Layer 0 multimodal `soft_prompt_embeds` conditioning,
//! `DeepSeek2` Multi-Head Latent Attention (`MLA`) matrix absorption, 3:1 hybrid `Gated-DeltaNet` +
//! full `GQA` attention, and dense causal embeddings (`LastToken` / `Mean` / `Cls` pooling +
//! optional Matryoshka truncation + L2 normalization).

#![deny(unsafe_op_in_unsafe_fn)]

pub mod common;
pub mod deepseek2_mla;
pub mod embedding;
mod kernels_compat;
pub mod qwen2;
pub mod qwen35;
pub mod qwen3_moe;

#[cfg(test)]
mod tests;

pub use common::{
    DenseAttnLayerSpec, add_bias_if_present, embed_token, even_head_dim, is_supported_quant,
    matvec_any, project_lm_head, rmsnorm_standard, run_decode_with,
    run_dense_attention_sublayer_scaled, run_dense_swiglu_ffn, run_prefill_with,
    run_prefill_with_hidden, select_top_k_experts, validate_qwen_revision_from_probe,
};
pub use deepseek2_mla::{
    DeepSeek2MlaConfig, DeepSeek2MlaRecipe, DeepSeek2Recipe, run_mla_absorbed_attention,
};
pub use embedding::QwenEmbeddingRecipe;
pub use qwen2::{Qwen2Recipe, Qwen3Recipe, QwenRecipe};
pub use qwen3_moe::{Qwen3MoeRecipe, dispatch_qwen3_moe_ffn};
pub use qwen35::{Qwen35MoeRecipe, Qwen35Recipe};

/// Canonical crate identifier for this architecture recipe package.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Canonical SPDX license identifier for this architecture recipe package.
pub const CRATE_LICENSE: &str = "LGPL-3.0-only WITH LGPL-3.0-linking-exception";

/// Returns the crate description configured in `Cargo.toml`.
#[must_use]
pub const fn crate_description() -> &'static str {
    env!("CARGO_PKG_DESCRIPTION")
}
