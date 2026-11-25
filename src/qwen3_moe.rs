//! Sparse Mixture-of-Experts `Qwen3MoE` (`qwen3moe`) recipe and fail-closed top-`k` router/expert
//! dispatch shared with `DeepSeek2` and `Qwen3.5 MoE`.

use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, ModelMetadata, ModelWeights, QuantType, TensorView,
};

use crate::common::{
    bounds_err, embed_token, ensure_quant_supported, execute_standard_token_with_ffn,
    is_supported_quant, matvec_any, read_required_1d_f32, resolve_tensor, run_decode_with,
    run_dense_swiglu_ffn, run_prefill_with_hidden, select_top_k_experts,
    validate_model_quant_and_adr023,
};
use crate::kernels_compat::swiglu;
use crate::qwen2::Qwen3Recipe;

/// Sparse fine-grained Mixture-of-Experts recipe for `qwen3moe` (e.g. `Qwen3-30B-A3B`).
#[derive(Debug, Clone)]
pub struct Qwen3MoeRecipe {
    /// Base `Qwen3` dense attention configuration (`QK-norm`, no QKV bias).
    pub base: Qwen3Recipe,
    /// Total number of routed experts (`128` for `Qwen3-30B-A3B`).
    pub num_experts: usize,
    /// Number of active experts routed per token (`8` for `Qwen3-30B-A3B`).
    pub num_experts_per_tok: usize,
    /// Per-expert `SwiGLU` intermediate dimension (`768` for `Qwen3-30B-A3B`).
    pub expert_ffn_dim: usize,
}

impl Default for Qwen3MoeRecipe {
    fn default() -> Self {
        Self::new_30b_a3b()
    }
}

impl Qwen3MoeRecipe {
    /// Creates a baseline `qwen3moe` recipe matching `Qwen3-30B-A3B`.
    #[must_use]
    pub const fn new() -> Self {
        Self::new_30b_a3b()
    }

    /// Creates the canonical `Qwen3-30B-A3B` profile (`128` experts, top-`8` active).
    #[must_use]
    pub const fn new_30b_a3b() -> Self {
        Self {
            base: Qwen3Recipe {
                num_layers: 48,
                hidden_dim: 2048,
                ffn_dim: 6144,
                n_heads_q: 32,
                n_heads_kv: 4,
                head_dim: 128,
                vocab_size: 151_936,
                rms_eps: 1e-6,
                rope_theta: 1_000_000.0,
            },
            num_experts: 128,
            num_experts_per_tok: 8,
            expert_ffn_dim: 768,
        }
    }
}

impl ArchRecipe for Qwen3MoeRecipe {
    fn architecture_id(&self) -> &'static str {
        "qwen3moe"
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
        validate_model_quant_and_adr023("qwen3moe", |q| self.supports_quant(q), metadata, weights)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let attn = self.base.attn_spec();
        let hidden_dim = self.base.hidden_dim;
        run_prefill_with_hidden(
            ctx,
            "qwen3moe",
            hidden_dim,
            weights,
            |hidden, pos, seq_len, kv, logits| {
                execute_standard_token_with_ffn(
                    "qwen3moe",
                    self.base.num_layers,
                    attn,
                    hidden,
                    pos,
                    seq_len,
                    kv,
                    logits,
                    weights,
                    |layer, norm_in| {
                        let mut m_out = vec![0.0_f32; hidden_dim];
                        dispatch_qwen3_moe_ffn(
                            &mut m_out,
                            norm_in,
                            layer,
                            self.num_experts,
                            self.num_experts_per_tok,
                            self.expert_ffn_dim,
                            false,
                            weights,
                        )?;
                        Ok(m_out)
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
        let attn = self.base.attn_spec();
        let hidden_dim = self.base.hidden_dim;
        run_decode_with(ctx, "qwen3moe", |token_id, pos, kv, logits| {
            let hidden = embed_token(token_id, hidden_dim, weights)?;
            execute_standard_token_with_ffn(
                "qwen3moe",
                self.base.num_layers,
                attn,
                hidden,
                pos,
                pos + 1,
                kv,
                logits,
                weights,
                |layer, norm_in| {
                    let mut m_out = vec![0.0_f32; hidden_dim];
                    dispatch_qwen3_moe_ffn(
                        &mut m_out,
                        norm_in,
                        layer,
                        self.num_experts,
                        self.num_experts_per_tok,
                        self.expert_ffn_dim,
                        false,
                        weights,
                    )?;
                    Ok(m_out)
                },
            )
        })
    }
}

fn resolve_required_expert<'a>(
    weights: &'a dyn ModelWeights,
    stacked_name: &str,
    indexed_name: &str,
    hf_indexed_name: &str,
    expert_idx: usize,
    num_experts: usize,
) -> Result<TensorView<'a>, K3Error> {
    if weights.contains_tensor(stacked_name) {
        let stacked = weights.tensor(stacked_name)?;
        if stacked.shape().len() != 3 {
            return Err(bounds_err(
                stacked.name(),
                stacked.offset(),
                stacked.byte_len(),
                stacked.as_bytes().len(),
                format!(
                    "stacked expert tensor '{}' must be 3D, got {:?}",
                    stacked.name(),
                    stacked.shape()
                ),
            ));
        }
        return stacked.slice_expert(expert_idx, num_experts);
    }
    resolve_tensor(weights, indexed_name, &[hf_indexed_name])
}

fn run_single_swiglu_expert(
    expert_gate: &TensorView<'_>,
    expert_up: &TensorView<'_>,
    expert_down: &TensorView<'_>,
    norm_ffn_in: &[f32],
    expert_ffn_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    ensure_quant_supported("qwen3moe", expert_gate)?;
    ensure_quant_supported("qwen3moe", expert_up)?;
    ensure_quant_supported("qwen3moe", expert_down)?;

    let hidden_dim = norm_ffn_in.len();
    let expected_elems = hidden_dim.checked_mul(expert_ffn_dim).ok_or_else(|| {
        bounds_err(
            expert_gate.name(),
            0,
            usize::MAX,
            expert_gate.as_bytes().len(),
            "expert SwiGLU dimension product overflowed usize",
        )
    })?;
    for view in [expert_gate, expert_up, expert_down] {
        if view.num_elements() != expected_elems {
            return Err(bounds_err(
                view.name(),
                view.offset(),
                view.byte_len(),
                view.as_bytes().len(),
                format!(
                    "expert tensor '{}' elements {} != hidden_dim({hidden_dim}) * expert_ffn_dim({expert_ffn_dim})",
                    view.name(),
                    view.num_elements()
                ),
            ));
        }
    }

    let mut gate = vec![0.0_f32; expert_ffn_dim];
    let mut up = vec![0.0_f32; expert_ffn_dim];
    matvec_any(&mut gate, expert_gate, norm_ffn_in)?;
    matvec_any(&mut up, expert_up, norm_ffn_in)?;

    let mut act = vec![0.0_f32; expert_ffn_dim];
    swiglu(&mut act, &gate, &up)?;

    let mut out = vec![0.0_f32; hidden_dim];
    matvec_any(&mut out, expert_down, &act)?;
    Ok(out)
}

fn select_experts_with_optional_bias(
    router_logits: &[f32],
    router_bias: Option<&[f32]>,
    top_k: usize,
) -> Vec<(usize, f32)> {
    let Some(bias) = router_bias else {
        return select_top_k_experts(router_logits, top_k, true);
    };
    let n = router_logits.len();
    if n == 0 || top_k == 0 {
        return Vec::new();
    }
    let biased: Vec<f32> = router_logits
        .iter()
        .enumerate()
        .map(|(idx, &l)| l + bias.get(idx).copied().unwrap_or(0.0))
        .collect();
    let chosen = select_top_k_experts(&biased, top_k, false);
    let max_logit = router_logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = router_logits
        .iter()
        .map(|&l| {
            if l.is_finite() && max_logit.is_finite() {
                (l - max_logit).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum_probs: f32 = probs.iter().sum();
    let mut out: Vec<(usize, f32)> = chosen
        .into_iter()
        .map(|(idx, _)| {
            let p = if sum_probs > 0.0 && sum_probs.is_finite() {
                probs[idx] / sum_probs
            } else {
                0.0
            };
            (idx, p)
        })
        .collect();
    let sel_sum: f32 = out.iter().map(|&(_, w)| w).sum();
    if sel_sum > 0.0 && sel_sum.is_finite() {
        for (_, w) in &mut out {
            *w /= sel_sum;
        }
    } else {
        #[expect(clippy::cast_precision_loss, reason = "top_k fits cleanly in f32")]
        let uniform = 1.0_f32 / (out.len().max(1) as f32);
        for (_, w) in &mut out {
            *w = uniform;
        }
    }
    out
}

fn add_required_shared_expert(
    out: &mut [f32],
    norm_ffn_in: &[f32],
    layer: usize,
    expert_ffn_dim: usize,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let sg = resolve_tensor(
        weights,
        &format!("blk.{layer}.ffn_gate_shexp.weight"),
        &[&format!(
            "model.layers.{layer}.mlp.shared_expert.gate_proj.weight"
        )],
    )?;
    let su = resolve_tensor(
        weights,
        &format!("blk.{layer}.ffn_up_shexp.weight"),
        &[&format!(
            "model.layers.{layer}.mlp.shared_expert.up_proj.weight"
        )],
    )?;
    let sd = resolve_tensor(
        weights,
        &format!("blk.{layer}.ffn_down_shexp.weight"),
        &[&format!(
            "model.layers.{layer}.mlp.shared_expert.down_proj.weight"
        )],
    )?;
    let sh_out = run_single_swiglu_expert(&sg, &su, &sd, norm_ffn_in, expert_ffn_dim)?;
    for (dst, &src) in out.iter_mut().zip(sh_out.iter()) {
        *dst += src;
    }
    Ok(())
}

/// Evaluates a fine-grained sparse `MoE` layer (router gate + top-`k` `SwiGLU` experts) using
/// either 3D stacked tensors (`ffn_*_exps.weight`) or per-expert tensors (`ffn_*_exps.{e}.weight`),
/// plus optional required shared experts (`ffn_*_shexp.weight` when `use_shared_expert` is `true`).
///
/// Fails closed per ADR-045 if `num_experts == 0`, `top_k == 0`, `expert_ffn_dim == 0`,
/// `out.len() != norm_ffn_in.len()`, or any required router, shared-expert, or active expert
/// tensor is missing or dimension-mismatched.
///
/// Returns the selected `(expert_idx, normalized_weight)` pairs.
///
/// # Errors
///
/// Returns [`K3Error`] if router evaluation, expert tensor lookup/slicing, dimension checks,
/// or kernel math fails.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit layer MoE dispatch helper shared by Qwen3Moe, DeepSeek2, and Qwen35Moe"
)]
pub fn dispatch_qwen3_moe_ffn(
    out: &mut [f32],
    norm_ffn_in: &[f32],
    layer: usize,
    num_experts: usize,
    top_k: usize,
    expert_ffn_dim: usize,
    use_shared_expert: bool,
    weights: &dyn ModelWeights,
) -> Result<Vec<(usize, f32)>, K3Error> {
    let hidden_dim = norm_ffn_in.len();
    if hidden_dim == 0 || out.len() != hidden_dim {
        return Err(bounds_err(
            "moe_ffn",
            0,
            size_of_val(out),
            size_of_val(norm_ffn_in),
            format!(
                "MoE hidden_dim ({hidden_dim}) must be non-zero and match out len ({})",
                out.len()
            ),
        ));
    }
    if num_experts == 0 || top_k == 0 || expert_ffn_dim == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "qwen3moe".to_owned(),
            reason: format!(
                "invalid MoE configuration: num_experts={num_experts}, top_k={top_k}, expert_ffn_dim={expert_ffn_dim}"
            ),
        });
    }

    out.fill(0.0);

    if use_shared_expert {
        add_required_shared_expert(out, norm_ffn_in, layer, expert_ffn_dim, weights)?;
    }

    let router_view = resolve_tensor(
        weights,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        &[&format!("model.layers.{layer}.mlp.gate.weight")],
    )?;
    ensure_quant_supported("qwen3moe", &router_view)?;
    let mut router_logits = vec![0.0_f32; num_experts];
    matvec_any(&mut router_logits, &router_view, norm_ffn_in)?;

    let bias_name = format!("blk.{layer}.exp_probs_b.bias");
    let bias_opt = if weights.contains_tensor(&bias_name) {
        Some(read_required_1d_f32(
            weights,
            &bias_name,
            &[],
            num_experts,
            "qwen3moe",
        )?)
    } else {
        None
    };
    let selected = select_experts_with_optional_bias(&router_logits, bias_opt.as_deref(), top_k);

    let g_stack = format!("blk.{layer}.ffn_gate_exps.weight");
    let u_stack = format!("blk.{layer}.ffn_up_exps.weight");
    let d_stack = format!("blk.{layer}.ffn_down_exps.weight");

    for &(expert_idx, gate_weight) in &selected {
        let g_view = resolve_required_expert(
            weights,
            &g_stack,
            &format!("blk.{layer}.ffn_gate_exps.{expert_idx}.weight"),
            &format!("model.layers.{layer}.mlp.experts.{expert_idx}.gate_proj.weight"),
            expert_idx,
            num_experts,
        )?;
        let u_view = resolve_required_expert(
            weights,
            &u_stack,
            &format!("blk.{layer}.ffn_up_exps.{expert_idx}.weight"),
            &format!("model.layers.{layer}.mlp.experts.{expert_idx}.up_proj.weight"),
            expert_idx,
            num_experts,
        )?;
        let d_view = resolve_required_expert(
            weights,
            &d_stack,
            &format!("blk.{layer}.ffn_down_exps.{expert_idx}.weight"),
            &format!("model.layers.{layer}.mlp.experts.{expert_idx}.down_proj.weight"),
            expert_idx,
            num_experts,
        )?;
        let exp_out =
            run_single_swiglu_expert(&g_view, &u_view, &d_view, norm_ffn_in, expert_ffn_dim)?;
        for (dst, &src) in out.iter_mut().zip(exp_out.iter()) {
            *dst += gate_weight * src;
        }
    }

    Ok(selected)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MoeLayerConfig {
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
    pub(crate) expert_ffn_dim: usize,
    pub(crate) use_shared: bool,
}

pub(crate) fn run_layer_moe_or_dense_ffn(
    layer: usize,
    hidden_dim: usize,
    dense_ffn_dim: usize,
    moe: Option<MoeLayerConfig>,
    norm_ffn_in: &[f32],
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let Some(moe_cfg) = moe else {
        return run_dense_swiglu_ffn(layer, hidden_dim, dense_ffn_dim, norm_ffn_in, weights);
    };
    let has_router = weights.contains_tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))
        || weights.contains_tensor(&format!("model.layers.{layer}.mlp.gate.weight"));
    let has_dense = weights.contains_tensor(&format!("blk.{layer}.ffn_gate.weight"))
        || weights.contains_tensor(&format!("model.layers.{layer}.mlp.gate_proj.weight"));
    if has_router || !has_dense {
        let mut out = vec![0.0_f32; hidden_dim];
        dispatch_qwen3_moe_ffn(
            &mut out,
            norm_ffn_in,
            layer,
            moe_cfg.num_experts,
            moe_cfg.top_k,
            moe_cfg.expert_ffn_dim,
            moe_cfg.use_shared,
            weights,
        )?;
        Ok(out)
    } else {
        run_dense_swiglu_ffn(layer, hidden_dim, dense_ffn_dim, norm_ffn_in, weights)
    }
}
