//! `Qwen3.5` (`qwen35`) and `Qwen3.5 MoE` (`qwen35moe`) hybrid 3:1 `Gated-DeltaNet` linear
//! attention + full `QK-norm` `GQA` recipes.

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};

use crate::common::{
    DenseAttnLayerSpec, bounds_err, embed_token, even_head_dim, is_supported_quant,
    project_lm_head, project_required, read_required_1d_f32, rmsnorm_standard, run_decode_with,
    run_dense_attention_sublayer, run_prefill_with_hidden,
};
use crate::kernels_compat::gated_deltanet_step;
use crate::qwen3_moe::{MoeLayerConfig, run_layer_moe_or_dense_ffn};

/// Dense `Qwen3.5` (`qwen35`) recipe with 3:1 interleaved `Gated-DeltaNet` linear attention
/// and full `QK-norm` `GQA` layers.
#[derive(Debug, Clone)]
pub struct Qwen35Recipe {
    /// Transformer block layer count.
    pub num_layers: usize,
    /// Model embedding and hidden dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value head count.
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Standard `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`1_000_000.0`).
    pub rope_theta: f32,
    /// Interval at which full `GQA` layers appear (`4` => 3 linear + 1 full attention).
    pub full_attention_interval: usize,
}

impl Default for Qwen35Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen35Recipe {
    /// Creates a baseline `qwen35` dense hybrid recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 24,
            hidden_dim: 1024,
            ffn_dim: 3584,
            n_heads_q: 8,
            n_heads_kv: 2,
            head_dim: 128,
            vocab_size: 248_320,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            full_attention_interval: 4,
        }
    }

    /// Returns `true` when `layer_idx` executes full `QK-norm` `GQA` instead of `Gated-DeltaNet`.
    #[must_use]
    pub const fn is_full_attention_layer(&self, layer_idx: usize) -> bool {
        let interval = if self.full_attention_interval == 0 {
            4
        } else {
            self.full_attention_interval
        };
        (layer_idx + 1) % interval == 0
    }
}

/// Sparse `Qwen3.5 MoE` (`qwen35moe`) recipe combining hybrid `Gated-DeltaNet` linear attention
/// with shared + top-`k` routed experts.
#[derive(Debug, Clone)]
pub struct Qwen35MoeRecipe {
    /// Base hybrid linear/full attention configuration.
    pub base: Qwen35Recipe,
    /// Total number of routed experts.
    pub num_experts: usize,
    /// Active routed experts per token.
    pub num_experts_per_tok: usize,
    /// Number of always-active shared experts.
    pub num_shared_experts: usize,
    /// Per-expert `SwiGLU` intermediate dimension.
    pub expert_ffn_dim: usize,
}

impl Default for Qwen35MoeRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen35MoeRecipe {
    /// Creates a baseline `qwen35moe` hybrid `MoE` recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            base: Qwen35Recipe::new(),
            num_experts: 128,
            num_experts_per_tok: 8,
            num_shared_experts: 1,
            expert_ffn_dim: 512,
        }
    }

    const fn moe_cfg(&self) -> MoeLayerConfig {
        MoeLayerConfig {
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
            expert_ffn_dim: self.expert_ffn_dim,
            use_shared: self.num_shared_experts > 0,
        }
    }
}

fn sigmoid_clamp(x: f32) -> f32 {
    let clamped = x.clamp(-6.0, 6.0);
    (1.0_f32 / (1.0_f32 + (-clamped).exp())).clamp(0.01, 0.99)
}

fn repeat_kv_for_q_heads(
    kv: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
) -> Vec<f32> {
    if n_heads_kv >= n_heads_q || n_heads_kv == 0 {
        return kv[..n_heads_q * head_dim].to_vec();
    }
    let ratio = (n_heads_q / n_heads_kv).max(1);
    let mut out = Vec::with_capacity(n_heads_q * head_dim);
    for q_h in 0..n_heads_q {
        let kv_h = (q_h / ratio).min(n_heads_kv - 1);
        out.extend_from_slice(&kv[kv_h * head_dim..(kv_h + 1) * head_dim]);
    }
    out
}

fn choose_deltanet_geometry(
    q_dim: usize,
    desired_head_dim: usize,
    max_state_elems: usize,
) -> (usize, usize) {
    let start_d = desired_head_dim.min(q_dim).max(1);
    for d in (1..=start_d).rev() {
        if q_dim % d == 0 && q_dim * d <= max_state_elems {
            return (q_dim / d, d);
        }
    }
    (q_dim.max(1), 1)
}

struct DeltaNetStepInputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    decay_gate: Vec<f32>,
    beta: Vec<f32>,
    eff_heads: usize,
    eff_head_dim: usize,
    state_len: usize,
}

fn project_deltanet_inputs(
    spec: DenseAttnLayerSpec,
    layer: usize,
    norm_hidden: &[f32],
    max_state_elems: usize,
    weights: &dyn ModelWeights,
) -> Result<DeltaNetStepInputs, K3Error> {
    let head_d_even = even_head_dim(spec.head_dim);
    let n_q = spec.n_heads_q;
    let n_kv = spec.n_heads_kv;
    let q_dim = n_q * head_d_even;
    let kv_dim = n_kv * head_d_even;

    let q = project_required(
        weights,
        &format!("blk.{layer}.attn_q.weight"),
        &[&format!("model.layers.{layer}.self_attn.q_proj.weight")],
        norm_hidden,
        q_dim,
        "qwen35",
    )?;
    let k_raw = project_required(
        weights,
        &format!("blk.{layer}.attn_k.weight"),
        &[&format!("model.layers.{layer}.self_attn.k_proj.weight")],
        norm_hidden,
        kv_dim,
        "qwen35",
    )?;
    let v_raw = project_required(
        weights,
        &format!("blk.{layer}.attn_v.weight"),
        &[&format!("model.layers.{layer}.self_attn.v_proj.weight")],
        norm_hidden,
        kv_dim,
        "qwen35",
    )?;
    let k = repeat_kv_for_q_heads(&k_raw, n_q, n_kv, head_d_even);
    let v = repeat_kv_for_q_heads(&v_raw, n_q, n_kv, head_d_even);

    let (eff_heads, eff_head_dim) = choose_deltanet_geometry(q_dim, head_d_even, max_state_elems);
    let state_len = eff_heads * eff_head_dim * eff_head_dim;
    if max_state_elems < state_len {
        return Err(bounds_err(
            "deltanet_state",
            0,
            state_len.saturating_mul(size_of::<f32>()),
            max_state_elems.saturating_mul(size_of::<f32>()),
            format!(
                "DeltaNet layer {layer} state buffer length {max_state_elems} < required {state_len}"
            ),
        ));
    }

    let raw_decay = project_required(
        weights,
        &format!("blk.{layer}.ssm_a.weight"),
        &[&format!("blk.{layer}.ssm_a")],
        norm_hidden,
        eff_heads,
        "qwen35",
    )?;
    let raw_beta = project_required(
        weights,
        &format!("blk.{layer}.ssm_beta.weight"),
        &[&format!("blk.{layer}.ssm_beta")],
        norm_hidden,
        eff_heads,
        "qwen35",
    )?;

    Ok(DeltaNetStepInputs {
        q,
        k,
        v,
        decay_gate: raw_decay.into_iter().map(sigmoid_clamp).collect(),
        beta: raw_beta.into_iter().map(sigmoid_clamp).collect(),
        eff_heads,
        eff_head_dim,
        state_len,
    })
}

fn run_gated_deltanet_sublayer(
    spec: DenseAttnLayerSpec,
    layer: usize,
    pos: usize,
    hidden: &mut [f32],
    layer_kv: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    let attn_norm_w = read_required_1d_f32(
        weights,
        &format!("blk.{layer}.attn_norm.weight"),
        &[&format!("model.layers.{layer}.input_layernorm.weight")],
        hidden_dim,
        "qwen35",
    )?;
    let mut norm_hidden = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut norm_hidden, hidden, &attn_norm_w, spec.rms_eps)?;

    let inputs = project_deltanet_inputs(spec, layer, &norm_hidden, layer_kv.len(), weights)?;
    let mut out = vec![0.0_f32; inputs.q.len()];
    let state = &mut layer_kv[..inputs.state_len];
    if pos == 0 {
        state.fill(0.0);
    }
    gated_deltanet_step(
        state,
        &mut out,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &inputs.decay_gate,
        &inputs.beta,
        inputs.eff_heads,
        inputs.eff_head_dim,
    )?;

    let proj_attn = project_required(
        weights,
        &format!("blk.{layer}.attn_output.weight"),
        &[&format!("model.layers.{layer}.self_attn.o_proj.weight")],
        &out,
        hidden_dim,
        "qwen35",
    )?;
    for (h, &r) in hidden.iter_mut().zip(proj_attn.iter()) {
        *h += r;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "shared Gated-DeltaNet + full-attention Qwen3.5 step supporting Layer-0 soft prompts"
)]
fn execute_qwen35_token_with_hidden(
    arch: &str,
    base: &Qwen35Recipe,
    moe: Option<MoeLayerConfig>,
    mut hidden: Vec<f32>,
    pos: usize,
    seq_len: usize,
    kv_cache: &mut [f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    if base.num_layers == 0
        || base.hidden_dim == 0
        || base.n_heads_q == 0
        || base.n_heads_kv == 0
        || base.head_dim < 2
        || !base.rms_eps.is_finite()
        || base.rms_eps <= 0.0
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: "Qwen35Recipe contains zero or non-finite geometry parameters".to_owned(),
        });
    }
    if hidden.len() != base.hidden_dim {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "layer-0 hidden length {} does not match hidden_dim {}",
                hidden.len(),
                base.hidden_dim
            ),
        });
    }

    let layers = base.num_layers;
    let hidden_dim = base.hidden_dim;
    let eps = base.rms_eps;
    let spec = DenseAttnLayerSpec {
        hidden_dim,
        ffn_dim: base.ffn_dim,
        n_heads_q: base.n_heads_q,
        n_heads_kv: base.n_heads_kv,
        head_dim: base.head_dim,
        rms_eps: eps,
        rope_theta: base.rope_theta,
        use_qkv_bias: false,
        use_qk_norm: true,
    };

    let layer_stride = kv_cache.len() / layers;
    for layer in 0..layers {
        let layer_kv = &mut kv_cache[layer * layer_stride..(layer + 1) * layer_stride];
        let force_deltanet = weights.contains_tensor(&format!("blk.{layer}.ssm_a.weight"))
            || weights.contains_tensor(&format!("blk.{layer}.ssm_a"));
        let is_full = !force_deltanet && base.is_full_attention_layer(layer);
        if is_full {
            run_dense_attention_sublayer(
                spec,
                layer,
                pos,
                seq_len,
                &mut hidden,
                layer_kv,
                weights,
            )?;
        } else {
            run_gated_deltanet_sublayer(spec, layer, pos, &mut hidden, layer_kv, weights)?;
        }

        let ffn_norm_w = read_required_1d_f32(
            weights,
            &format!("blk.{layer}.ffn_norm.weight"),
            &[&format!(
                "model.layers.{layer}.post_attention_layernorm.weight"
            )],
            hidden_dim,
            arch,
        )?;
        let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut norm_ffn_in, &hidden, &ffn_norm_w, eps)?;

        let ffn_out = run_layer_moe_or_dense_ffn(
            layer,
            hidden_dim,
            base.ffn_dim,
            moe,
            &norm_ffn_in,
            weights,
        )?;
        for (h, &r) in hidden.iter_mut().zip(ffn_out.iter()) {
            *h += r;
        }
    }

    let out_norm_w = read_required_1d_f32(
        weights,
        "output_norm.weight",
        &["model.norm.weight"],
        hidden_dim,
        arch,
    )?;
    let mut normed_final = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut normed_final, &hidden, &out_norm_w, eps)?;
    project_lm_head(weights, &normed_final, logits)
}

impl ArchRecipe for Qwen35Recipe {
    fn architecture_id(&self) -> &'static str {
        "qwen35"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_prefill_with_hidden(
            ctx,
            "qwen35",
            self.hidden_dim,
            weights,
            |hidden, pos, seq_len, kv, logits| {
                execute_qwen35_token_with_hidden(
                    "qwen35", self, None, hidden, pos, seq_len, kv, logits, weights,
                )
            },
        )
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, "qwen35", |token_id, pos, kv, logits| {
            let hidden = embed_token(token_id, self.hidden_dim, weights)?;
            execute_qwen35_token_with_hidden(
                "qwen35",
                self,
                None,
                hidden,
                pos,
                pos + 1,
                kv,
                logits,
                weights,
            )
        })
    }
}

impl ArchRecipe for Qwen35MoeRecipe {
    fn architecture_id(&self) -> &'static str {
        "qwen35moe"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let moe = self.moe_cfg();
        run_prefill_with_hidden(
            ctx,
            "qwen35moe",
            self.base.hidden_dim,
            weights,
            |hidden, pos, seq_len, kv, logits| {
                execute_qwen35_token_with_hidden(
                    "qwen35moe",
                    &self.base,
                    Some(moe),
                    hidden,
                    pos,
                    seq_len,
                    kv,
                    logits,
                    weights,
                )
            },
        )
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let moe = self.moe_cfg();
        run_decode_with(ctx, "qwen35moe", |token_id, pos, kv, logits| {
            let hidden = embed_token(token_id, self.base.hidden_dim, weights)?;
            execute_qwen35_token_with_hidden(
                "qwen35moe",
                &self.base,
                Some(moe),
                hidden,
                pos,
                pos + 1,
                kv,
                logits,
                weights,
            )
        })
    }
}
