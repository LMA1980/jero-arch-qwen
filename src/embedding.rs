//! Dense causal text & multimodal embedding recipe for `Qwen3-Embedding` (`"qwen3-embedding"`).
//!
//! Executes causal `QK-norm` `GQA` transformer layers `0..num_layers`, applies final `output_norm`,
//! extracts the final hidden sequence states, and pools + L2-normalizes directly into
//! [`ExecutionContext::embedding_output_mut`] (defaulting to [`EmbeddingPooling::LastToken`] with
//! optional Matryoshka truncation) without allocating or projecting LM-head logits.

use jero_core::{
    ArchRecipe, EmbeddingPooling, ExecutionContext, K3Error, ModelMetadata, ModelWeights, QuantType,
};

use crate::common::{
    bounds_err, collect_layer0_sequence_hidden, ensure_all_finite, even_head_dim,
    is_supported_quant, read_required_1d_f32, rmsnorm_standard, run_dense_attention_sublayer,
    run_dense_swiglu_ffn, validate_model_quant_and_adr023, validate_model_weights_adr023,
};
use crate::kernels_compat::pool_and_l2_normalize;
use crate::qwen2::Qwen3Recipe;

/// Configuration for `QwenEmbeddingRecipe` (`"qwen3-embedding"`).
#[derive(Debug, Clone)]
pub struct QwenEmbeddingRecipe {
    /// Base `Qwen3` dense causal attention configuration (`QK-norm`, no QKV bias).
    pub base: Qwen3Recipe,
    /// Pooling mode applied when [`ExecutionContext::embedding_pooling`] is
    /// [`EmbeddingPooling::None`] (defaults to [`EmbeddingPooling::LastToken`]).
    pub default_pooling: EmbeddingPooling,
}

impl Default for QwenEmbeddingRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl QwenEmbeddingRecipe {
    /// Creates a baseline `qwen3-embedding` recipe matching `Qwen3-Embedding-0.6B` defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            base: Qwen3Recipe {
                num_layers: 28,
                hidden_dim: 1024,
                ffn_dim: 3072,
                n_heads_q: 16,
                n_heads_kv: 8,
                head_dim: 128,
                vocab_size: 151_669,
                rms_eps: 1e-6,
                rope_theta: 1_000_000.0,
            },
            default_pooling: EmbeddingPooling::LastToken,
        }
    }

    /// Wraps an explicit [`Qwen3Recipe`] backbone as a `qwen3-embedding` recipe using
    /// [`EmbeddingPooling::LastToken`].
    #[must_use]
    pub const fn with_base(base: Qwen3Recipe) -> Self {
        Self {
            base,
            default_pooling: EmbeddingPooling::LastToken,
        }
    }

    fn validate_recipe_config(&self) -> Result<(), K3Error> {
        let b = &self.base;
        if b.num_layers == 0
            || b.hidden_dim == 0
            || b.ffn_dim == 0
            || b.n_heads_q == 0
            || b.n_heads_kv == 0
            || b.head_dim < 2
            || b.n_heads_q % b.n_heads_kv != 0
            || !b.rms_eps.is_finite()
            || b.rms_eps <= 0.0
            || !b.rope_theta.is_finite()
            || b.rope_theta <= 0.0
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "qwen3-embedding".to_owned(),
                reason: "QwenEmbeddingRecipe contains invalid zero, non-divisible, or non-finite geometry".to_owned(),
            });
        }
        Ok(())
    }

    fn resolve_pooling_and_dim(
        &self,
        ctx: &ExecutionContext<'_>,
    ) -> Result<(u8, Option<usize>, usize), K3Error> {
        let hidden_dim = self.base.hidden_dim;
        let out_slice = ctx.embedding_output().ok_or_else(|| {
            bounds_err(
                "embedding_output",
                0,
                hidden_dim.saturating_mul(size_of::<f32>()),
                0,
                "QwenEmbeddingRecipe requires ExecutionContext::with_embedding_output",
            )
        })?;

        let pooling = match ctx.embedding_pooling() {
            EmbeddingPooling::None => self.default_pooling,
            explicit => explicit,
        };
        let mode_code = pooling
            .mode_code()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "qwen3-embedding".to_owned(),
                reason: "QwenEmbeddingRecipe requires Mean, Cls, or LastToken pooling".to_owned(),
            })?;

        let matryoshka_dim = ctx.matryoshka_dim();
        let effective_dim = match matryoshka_dim {
            None => hidden_dim,
            Some(d) if d == 0 || d > hidden_dim => {
                return Err(bounds_err(
                    "embed_matryoshka_dim",
                    0,
                    d.saturating_mul(size_of::<f32>()),
                    hidden_dim.saturating_mul(size_of::<f32>()),
                    format!("matryoshka_dim ({d}) must lie within 1..={hidden_dim}"),
                ));
            }
            Some(d) => d,
        };

        if out_slice.len() < effective_dim {
            return Err(bounds_err(
                "embedding_output",
                0,
                effective_dim.saturating_mul(size_of::<f32>()),
                size_of_val(out_slice),
                format!(
                    "embedding_output len {} is smaller than required effective_dim {effective_dim}",
                    out_slice.len()
                ),
            ));
        }

        Ok((mode_code, matryoshka_dim, effective_dim))
    }

    fn run_embedding_layers(
        &self,
        ctx: &mut ExecutionContext<'_>,
        hiddens: &mut [Vec<f32>],
        start_pos: usize,
        seq_end: usize,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let hidden_dim = self.base.hidden_dim;
        let num_layers = self.base.num_layers;
        let eps = self.base.rms_eps;
        let attn = self.base.attn_spec();
        let kv_dim = self.base.n_heads_kv * even_head_dim(self.base.head_dim);
        let total_new = hiddens.len();

        let use_caller_kv = !ctx.kv_cache().is_empty();
        let caller_layer_stride = if use_caller_kv {
            let stride = ctx.kv_cache().len() / num_layers;
            let capacity_slots = stride / (2 * kv_dim).max(1);
            if capacity_slots < seq_end {
                return Err(bounds_err(
                    "kv_cache",
                    0,
                    num_layers.saturating_mul(2 * seq_end * kv_dim * size_of::<f32>()),
                    size_of_val(ctx.kv_cache()),
                    format!(
                        "caller KV cache capacity ({capacity_slots} slots/layer) < seq_end ({seq_end})"
                    ),
                ));
            }
            stride
        } else {
            if start_pos > 0 {
                return Err(bounds_err(
                    "kv_cache",
                    0,
                    num_layers.saturating_mul(2 * seq_end * kv_dim * size_of::<f32>()),
                    0,
                    format!(
                        "cannot resume causal embedding at start_position={start_pos} with an empty kv_cache"
                    ),
                ));
            }
            0
        };

        let mut transient_layer_kv = if use_caller_kv {
            Vec::new()
        } else {
            vec![0.0_f32; 2 * total_new * kv_dim]
        };

        for layer in 0..num_layers {
            let layer_kv = if use_caller_kv {
                let kv = ctx.kv_cache_mut();
                &mut kv[layer * caller_layer_stride..(layer + 1) * caller_layer_stride]
            } else {
                transient_layer_kv.fill(0.0);
                &mut transient_layer_kv[..]
            };

            let ffn_norm_w = read_required_1d_f32(
                weights,
                &format!("blk.{layer}.ffn_norm.weight"),
                &[&format!(
                    "model.layers.{layer}.post_attention_layernorm.weight"
                )],
                hidden_dim,
                "qwen3-embedding",
            )?;

            for (step_idx, hidden) in hiddens.iter_mut().enumerate() {
                run_dense_attention_sublayer(
                    attn,
                    layer,
                    start_pos + step_idx,
                    seq_end,
                    hidden,
                    layer_kv,
                    weights,
                )?;
                let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
                rmsnorm_standard(&mut norm_ffn_in, hidden, &ffn_norm_w, eps)?;
                let ffn_out = run_dense_swiglu_ffn(
                    layer,
                    hidden_dim,
                    self.base.ffn_dim,
                    &norm_ffn_in,
                    weights,
                )?;
                for (h, &r) in hidden.iter_mut().zip(ffn_out.iter()) {
                    *h += r;
                }
            }
        }
        Ok(())
    }
}

impl ArchRecipe for QwenEmbeddingRecipe {
    fn architecture_id(&self) -> &'static str {
        "qwen3-embedding"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn requires_kv_cache(&self) -> bool {
        false
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
        self.validate_recipe_config()?;
        validate_model_weights_adr023("qwen3", weights)?;

        let hidden_dim = self.base.hidden_dim;
        let eps = self.base.rms_eps;
        let (mode_code, matryoshka_dim, effective_dim) = self.resolve_pooling_and_dim(ctx)?;
        let (mut hiddens, start_pos, seq_end) =
            collect_layer0_sequence_hidden(ctx, "qwen3-embedding", hidden_dim, weights)?;
        let total_new = hiddens.len();

        self.run_embedding_layers(ctx, &mut hiddens, start_pos, seq_end, weights)?;

        let out_norm_w = read_required_1d_f32(
            weights,
            "output_norm.weight",
            &["model.norm.weight"],
            hidden_dim,
            "qwen3-embedding",
        )?;
        let mut hidden_seq = vec![0.0_f32; total_new * hidden_dim];
        for (step_idx, hidden) in hiddens.iter().enumerate() {
            let dst = &mut hidden_seq[step_idx * hidden_dim..(step_idx + 1) * hidden_dim];
            rmsnorm_standard(dst, hidden, &out_norm_w, eps)?;
        }
        ensure_all_finite("qwen3_embedding_hidden_seq", &hidden_seq)?;

        let emb_out = ctx.embedding_output_mut().ok_or_else(|| {
            bounds_err(
                "embedding_output",
                0,
                effective_dim.saturating_mul(size_of::<f32>()),
                0,
                "embedding_output missing on ExecutionContext",
            )
        })?;
        if emb_out.len() > effective_dim {
            emb_out[effective_dim..].fill(0.0);
        }
        pool_and_l2_normalize(
            &mut emb_out[..effective_dim],
            &hidden_seq,
            total_new,
            hidden_dim,
            mode_code,
            matryoshka_dim,
        )?;

        ctx.advance_position(total_new)?;
        Ok(())
    }

    fn decode_step(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        Err(K3Error::CapabilityMismatch {
            architecture: "qwen3-embedding".to_owned(),
            reason:
                "qwen3-embedding is a single-pass embedding recipe and does not support decode_step"
                    .to_owned(),
        })
    }
}
