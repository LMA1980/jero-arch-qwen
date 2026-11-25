//! Alibaba `Qwen2` / `Qwen2.5` (`qwen2`, `Qwen2-VL`/`Qwen2.5-VL`) and `Qwen3` (`qwen3`) dense
//! architecture recipes.
//!
//! Enforces strict ADR-023 separation between `Qwen2Recipe` (`attn_{q,k,v}.bias`) and
//! `Qwen3Recipe` (`attn_{q,k}_norm.weight` without QKV biases), supports Layer 0 multimodal
//! `soft_prompt_embeds` conditioning, and fails closed on missing or dimension-mismatched
//! weights (ADR-045).

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelMetadata, ModelWeights, QuantType};

use crate::common::{
    DenseAttnLayerSpec, embed_token, execute_standard_token_with_ffn, is_supported_quant,
    run_decode_with, run_dense_swiglu_ffn, run_prefill_with_hidden,
    validate_model_quant_and_adr023,
};

/// Configuration for `Qwen2Recipe` (`qwen2`, `qwen2.5`, and `Qwen2-VL`/`Qwen2.5-VL` dense checkpoints).
#[derive(Debug, Clone)]
pub struct Qwen2Recipe {
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count (`GQA`).
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Standard `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`1_000_000.0` for `Qwen2.5`).
    pub rope_theta: f32,
}

/// Canonical alias for [`Qwen2Recipe`] covering `qwen2`, `qwen2.5`, and `Qwen2-VL`/`Qwen2.5-VL`.
pub type QwenRecipe = Qwen2Recipe;

impl Default for Qwen2Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen2Recipe {
    /// Creates a baseline `qwen2` recipe with standard `Qwen2.5-1.5B` defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 28,
            hidden_dim: 1536,
            ffn_dim: 8960,
            n_heads_q: 12,
            n_heads_kv: 2,
            head_dim: 128,
            vocab_size: 151_936,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    const fn attn_spec(&self) -> DenseAttnLayerSpec {
        DenseAttnLayerSpec {
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads_q: self.n_heads_q,
            n_heads_kv: self.n_heads_kv,
            head_dim: self.head_dim,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            use_qkv_bias: true,
            use_qk_norm: false,
        }
    }
}

impl ArchRecipe for Qwen2Recipe {
    fn architecture_id(&self) -> &'static str {
        "qwen2"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        validate_model_quant_and_adr023("qwen2", |q| self.supports_quant(q), metadata, weights)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let attn = self.attn_spec();
        run_prefill_with_hidden(
            ctx,
            "qwen2",
            self.hidden_dim,
            weights,
            |hidden, pos, seq_len, kv, logits| {
                execute_standard_token_with_ffn(
                    "qwen2",
                    self.num_layers,
                    attn,
                    hidden,
                    pos,
                    seq_len,
                    kv,
                    logits,
                    weights,
                    |layer, norm_in| {
                        run_dense_swiglu_ffn(layer, self.hidden_dim, self.ffn_dim, norm_in, weights)
                    },
                )
            },
        )
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let attn = self.attn_spec();
        run_decode_with(ctx, "qwen2", |token_id, pos, kv, logits| {
            let hidden = embed_token(token_id, self.hidden_dim, weights)?;
            execute_standard_token_with_ffn(
                "qwen2",
                self.num_layers,
                attn,
                hidden,
                pos,
                pos + 1,
                kv,
                logits,
                weights,
                |layer, norm_in| {
                    run_dense_swiglu_ffn(layer, self.hidden_dim, self.ffn_dim, norm_in, weights)
                },
            )
        })
    }
}

/// Configuration for `Qwen3Recipe` (`qwen3` dense checkpoints with per-head QK-norm).
#[derive(Debug, Clone)]
pub struct Qwen3Recipe {
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count (`GQA`).
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Standard `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`1_000_000.0`).
    pub rope_theta: f32,
}

impl Default for Qwen3Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3Recipe {
    /// Creates a baseline `qwen3` recipe with `Qwen3-1.7B` defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 28,
            hidden_dim: 2048,
            ffn_dim: 6144,
            n_heads_q: 16,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 151_936,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    pub(crate) const fn attn_spec(&self) -> DenseAttnLayerSpec {
        DenseAttnLayerSpec {
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads_q: self.n_heads_q,
            n_heads_kv: self.n_heads_kv,
            head_dim: self.head_dim,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            use_qkv_bias: false,
            use_qk_norm: true,
        }
    }
}

impl ArchRecipe for Qwen3Recipe {
    fn architecture_id(&self) -> &'static str {
        "qwen3"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        validate_model_quant_and_adr023("qwen3", |q| self.supports_quant(q), metadata, weights)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let attn = self.attn_spec();
        run_prefill_with_hidden(
            ctx,
            "qwen3",
            self.hidden_dim,
            weights,
            |hidden, pos, seq_len, kv, logits| {
                execute_standard_token_with_ffn(
                    "qwen3",
                    self.num_layers,
                    attn,
                    hidden,
                    pos,
                    seq_len,
                    kv,
                    logits,
                    weights,
                    |layer, norm_in| {
                        run_dense_swiglu_ffn(layer, self.hidden_dim, self.ffn_dim, norm_in, weights)
                    },
                )
            },
        )
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let attn = self.attn_spec();
        run_decode_with(ctx, "qwen3", |token_id, pos, kv, logits| {
            let hidden = embed_token(token_id, self.hidden_dim, weights)?;
            execute_standard_token_with_ffn(
                "qwen3",
                self.num_layers,
                attn,
                hidden,
                pos,
                pos + 1,
                kv,
                logits,
                weights,
                |layer, norm_in| {
                    run_dense_swiglu_ffn(layer, self.hidden_dim, self.ffn_dim, norm_in, weights)
                },
            )
        })
    }
}
