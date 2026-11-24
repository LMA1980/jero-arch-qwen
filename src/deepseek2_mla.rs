//! `DeepSeek-V2` / `DeepSeek-R1-Distill` (`deepseek2`) Multi-Head Latent Attention (`MLA`)
//! matrix absorption with shared + routed `MoE` execution recipe.

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};

use crate::common::{
    bounds_err, embed_token, even_head_dim, is_supported_quant, project_lm_head, project_required,
    read_required_1d_f32, read_required_matrix_f32, rmsnorm_standard, run_decode_with,
    run_prefill_with_hidden,
};
use crate::kernels_compat::{apply_rope, mla_absorbed_attention};
use crate::qwen3_moe::{MoeLayerConfig, run_layer_moe_or_dense_ffn};

/// Multi-Head Latent Attention (`MLA`) projection geometry for `deepseek2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeek2MlaConfig {
    /// Low-rank compressed KV latent rank (`d_c`, e.g. `512`).
    pub kv_lora_rank: usize,
    /// Optional query low-rank compression dimension (`1536` on full DeepSeek-V2, `None` on Lite).
    pub q_lora_rank: Option<usize>,
    /// Non-rotary per-head query/key dimension (`d_nope`, e.g. `128`).
    pub qk_nope_head_dim: usize,
    /// Decoupled `RoPE` per-head key/query dimension (`d_R`, e.g. `64`).
    pub qk_rope_head_dim: usize,
    /// Per-head value dimension (`d_v`, e.g. `128`).
    pub v_head_dim: usize,
}

impl Default for DeepSeek2MlaConfig {
    fn default() -> Self {
        Self {
            kv_lora_rank: 512,
            q_lora_rank: None,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
        }
    }
}

/// `DeepSeek-V2` / `DeepSeek-R1-Distill` (`deepseek2`) `MLA` + Shared/Routed `MoE` recipe.
#[derive(Debug, Clone)]
pub struct DeepSeek2Recipe {
    /// Transformer block layer count (`27` for DeepSeek-V2-Lite).
    pub num_layers: usize,
    /// Model embedding and hidden dimension (`2048` for DeepSeek-V2-Lite).
    pub hidden_dim: usize,
    /// Dense FFN intermediate dimension (first layer / non-MoE layers).
    pub ffn_dim: usize,
    /// Attention query head count (`16` for DeepSeek-V2-Lite).
    pub n_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Standard `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency (`10_000.0`).
    pub rope_theta: f32,
    /// Multi-Head Latent Attention configuration.
    pub mla: DeepSeek2MlaConfig,
    /// Total routed expert count (`64` for DeepSeek-V2-Lite).
    pub num_experts: usize,
    /// Active routed expert count per token (`6` for DeepSeek-V2-Lite).
    pub num_experts_per_tok: usize,
    /// Number of always-active shared experts (`2` for DeepSeek-V2-Lite).
    pub num_shared_experts: usize,
    /// Per-expert `SwiGLU` intermediate dimension (`1408` for DeepSeek-V2-Lite).
    pub expert_ffn_dim: usize,
}

/// Canonical alias for [`DeepSeek2Recipe`] (`deepseek2` Multi-Head Latent Attention).
pub type DeepSeek2MlaRecipe = DeepSeek2Recipe;

impl Default for DeepSeek2Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepSeek2Recipe {
    /// Creates a baseline `deepseek2` recipe matching `DeepSeek-V2-Lite-16B-A2.4B`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_layers: 27,
            hidden_dim: 2048,
            ffn_dim: 10_944,
            n_heads: 16,
            vocab_size: 102_400,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            mla: DeepSeek2MlaConfig {
                kv_lora_rank: 512,
                q_lora_rank: None,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
            },
            num_experts: 64,
            num_experts_per_tok: 6,
            num_shared_experts: 2,
            expert_ffn_dim: 1408,
        }
    }
}

/// Absorbs `W^{UK}` (`[n_heads, d_nope, kv_lora_rank]`) into query non-rotary projections
/// (`[n_heads * d_nope]`), runs absorbed `MLA` attention directly over compressed latent KV cache
/// entries (`[c_kv, k_rope]`), and projects per-head latent context outputs through absorbed
/// `W^{UV}` (`[n_heads, v_head_dim, kv_lora_rank]`).
///
/// # Errors
///
/// Returns [`K3Error`] if slice lengths do not match the configured `MLA` head dimensions or if
/// latent KV attention fails bounds checks.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit matrix-absorption reference contract for DeepSeek2 MLA verification"
)]
pub fn run_mla_absorbed_attention(
    out_heads: &mut [f32],
    q_nope: &[f32],
    q_rotary: &[f32],
    kv_latent_cache: &[f32],
    k_up_proj: &[f32],
    v_up_proj: &[f32],
    n_heads: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    rope_head_dim: usize,
    v_head_dim: usize,
    seq_len: usize,
) -> Result<(), K3Error> {
    if n_heads == 0
        || kv_lora_rank == 0
        || qk_nope_head_dim == 0
        || rope_head_dim == 0
        || v_head_dim == 0
        || q_nope.len() != n_heads * qk_nope_head_dim
        || q_rotary.len() != n_heads * rope_head_dim
        || out_heads.len() != n_heads * v_head_dim
        || k_up_proj.len() != n_heads * qk_nope_head_dim * kv_lora_rank
        || v_up_proj.len() != n_heads * v_head_dim * kv_lora_rank
    {
        return Err(bounds_err(
            "mla_absorbed",
            0,
            size_of_val(out_heads),
            size_of_val(kv_latent_cache),
            "MLA absorbed attention slice dimensions do not match configuration",
        ));
    }

    let mut q_absorbed = vec![0.0_f32; n_heads * kv_lora_rank];
    for h in 0..n_heads {
        let q_h = &q_nope[h * qk_nope_head_dim..(h + 1) * qk_nope_head_dim];
        let k_up_head = &k_up_proj
            [h * qk_nope_head_dim * kv_lora_rank..(h + 1) * qk_nope_head_dim * kv_lora_rank];
        let dst = &mut q_absorbed[h * kv_lora_rank..(h + 1) * kv_lora_rank];
        for (d, &q_val) in q_h.iter().enumerate() {
            let row = &k_up_head[d * kv_lora_rank..(d + 1) * kv_lora_rank];
            for (r, slot) in dst.iter_mut().enumerate() {
                *slot += q_val * row[r];
            }
        }
    }

    let mut out_latent = vec![0.0_f32; n_heads * kv_lora_rank];
    #[expect(
        clippy::cast_precision_loss,
        reason = "head dim sums fit comfortably in f32"
    )]
    let scale = 1.0_f32 / ((qk_nope_head_dim + rope_head_dim) as f32).sqrt();
    mla_absorbed_attention(
        &mut out_latent,
        &q_absorbed,
        q_rotary,
        kv_latent_cache,
        n_heads,
        kv_lora_rank,
        rope_head_dim,
        seq_len,
        scale,
    )?;

    out_heads.fill(0.0);
    for h in 0..n_heads {
        let lat_h = &out_latent[h * kv_lora_rank..(h + 1) * kv_lora_rank];
        let v_up_head =
            &v_up_proj[h * v_head_dim * kv_lora_rank..(h + 1) * v_head_dim * kv_lora_rank];
        let dst_h = &mut out_heads[h * v_head_dim..(h + 1) * v_head_dim];
        for (v_idx, dst) in dst_h.iter_mut().enumerate() {
            let row = &v_up_head[v_idx * kv_lora_rank..(v_idx + 1) * kv_lora_rank];
            let mut acc = 0.0_f32;
            for (&w, &lat) in row.iter().zip(lat_h.iter()) {
                acc += w * lat;
            }
            *dst = acc;
        }
    }
    Ok(())
}

fn project_mla_queries(
    recipe: &DeepSeek2Recipe,
    layer: usize,
    norm_hidden: &[f32],
    d_nope: usize,
    d_r: usize,
    weights: &dyn ModelWeights,
) -> Result<(Vec<f32>, Vec<f32>), K3Error> {
    let n_heads = recipe.n_heads;
    if let Some(q_rank) = recipe.mla.q_lora_rank {
        if q_rank == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "deepseek2".to_owned(),
                reason: "q_lora_rank cannot be Some(0)".to_owned(),
            });
        }
        let q_a_name = format!("blk.{layer}.attn_q_a.weight");
        if weights.contains_tensor(&q_a_name) {
            let q_latent =
                project_required(weights, &q_a_name, &[], norm_hidden, q_rank, "deepseek2")?;
            let q_norm_w = read_required_1d_f32(
                weights,
                &format!("blk.{layer}.attn_q_a_norm.weight"),
                &[],
                q_rank,
                "deepseek2",
            )?;
            let mut q_normed = vec![0.0_f32; q_rank];
            rmsnorm_standard(&mut q_normed, &q_latent, &q_norm_w, recipe.rms_eps)?;
            let q_full = project_required(
                weights,
                &format!("blk.{layer}.attn_q_b.weight"),
                &[],
                &q_normed,
                n_heads * (d_nope + d_r),
                "deepseek2",
            )?;
            let mut q_nope = Vec::with_capacity(n_heads * d_nope);
            let mut q_rotary = Vec::with_capacity(n_heads * d_r);
            for head in q_full.chunks_exact(d_nope + d_r) {
                q_nope.extend_from_slice(&head[..d_nope]);
                q_rotary.extend_from_slice(&head[d_nope..]);
            }
            return Ok((q_nope, q_rotary));
        }
    }

    let q_nope = project_required(
        weights,
        &format!("blk.{layer}.attn_q.weight"),
        &[],
        norm_hidden,
        n_heads * d_nope,
        "deepseek2",
    )?;
    let q_rotary = project_required(
        weights,
        &format!("blk.{layer}.attn_q_rope.weight"),
        &[],
        norm_hidden,
        n_heads * d_r,
        "deepseek2",
    )?;
    Ok((q_nope, q_rotary))
}

fn project_mla_kv_latent(
    recipe: &DeepSeek2Recipe,
    layer: usize,
    pos: usize,
    seq_len: usize,
    norm_hidden: &[f32],
    weights: &dyn ModelWeights,
) -> Result<(Vec<f32>, Vec<f32>), K3Error> {
    let d_c = recipe.mla.kv_lora_rank;
    let d_r = even_head_dim(recipe.mla.qk_rope_head_dim);
    let latent_dim = d_c + d_r;
    let kv_compressed = project_required(
        weights,
        &format!("blk.{layer}.attn_kv_a_mqa.weight"),
        &[],
        norm_hidden,
        latent_dim,
        "deepseek2",
    )?;
    let kv_a_norm_w = read_required_1d_f32(
        weights,
        &format!("blk.{layer}.attn_kv_a_norm.weight"),
        &[],
        d_c,
        "deepseek2",
    )?;
    let mut c_kv = vec![0.0_f32; d_c];
    rmsnorm_standard(
        &mut c_kv,
        &kv_compressed[..d_c],
        &kv_a_norm_w,
        recipe.rms_eps,
    )?;
    let mut k_rope = kv_compressed[d_c..latent_dim].to_vec();
    apply_rope(
        &mut k_rope,
        d_r,
        pos,
        seq_len.max(pos + 1),
        recipe.rope_theta,
    )?;
    Ok((c_kv, k_rope))
}

fn run_deepseek2_mla_sublayer(
    recipe: &DeepSeek2Recipe,
    layer: usize,
    pos: usize,
    seq_len: usize,
    hidden: &mut [f32],
    layer_kv: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let hidden_dim = recipe.hidden_dim;
    let n_heads = recipe.n_heads;
    let d_c = recipe.mla.kv_lora_rank;
    let d_r = even_head_dim(recipe.mla.qk_rope_head_dim);
    let d_nope = recipe.mla.qk_nope_head_dim;
    let d_v = recipe.mla.v_head_dim;
    let latent_dim = d_c + d_r;

    if layer_kv.len() < latent_dim {
        return Err(bounds_err(
            "mla_kv_cache",
            0,
            latent_dim.saturating_mul(size_of::<f32>()),
            size_of_val(layer_kv),
            format!(
                "DeepSeek2 layer KV cache length {} < latent_dim {latent_dim}",
                layer_kv.len()
            ),
        ));
    }

    let attn_norm_w = read_required_1d_f32(
        weights,
        &format!("blk.{layer}.attn_norm.weight"),
        &[&format!("model.layers.{layer}.input_layernorm.weight")],
        hidden_dim,
        "deepseek2",
    )?;
    let mut norm_hidden = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut norm_hidden, hidden, &attn_norm_w, recipe.rms_eps)?;

    let (q_nope, mut q_rotary) =
        project_mla_queries(recipe, layer, &norm_hidden, d_nope, d_r, weights)?;
    apply_rope(
        &mut q_rotary,
        d_r,
        pos,
        seq_len.max(pos + 1),
        recipe.rope_theta,
    )?;

    let (c_kv, k_rope) = project_mla_kv_latent(recipe, layer, pos, seq_len, &norm_hidden, weights)?;

    let k_up_proj = read_required_matrix_f32(
        weights,
        &format!("blk.{layer}.attn_w_uk.weight"),
        &[],
        n_heads * d_nope * d_c,
        "deepseek2",
    )?;
    let v_up_proj = read_required_matrix_f32(
        weights,
        &format!("blk.{layer}.attn_w_uv.weight"),
        &[],
        n_heads * d_v * d_c,
        "deepseek2",
    )?;

    let capacity_slots = layer_kv.len() / latent_dim;
    let slot = pos % capacity_slots;
    let active_len = (pos + 1).min(capacity_slots);
    let slot_slice = &mut layer_kv[slot * latent_dim..(slot + 1) * latent_dim];
    slot_slice[..d_c].copy_from_slice(&c_kv);
    slot_slice[d_c..].copy_from_slice(&k_rope);

    let mut attn_heads = vec![0.0_f32; n_heads * d_v];
    run_mla_absorbed_attention(
        &mut attn_heads,
        &q_nope,
        &q_rotary,
        &layer_kv[..active_len * latent_dim],
        &k_up_proj,
        &v_up_proj,
        n_heads,
        d_c,
        d_nope,
        d_r,
        d_v,
        active_len,
    )?;

    let proj_attn = project_required(
        weights,
        &format!("blk.{layer}.attn_output.weight"),
        &[&format!("model.layers.{layer}.self_attn.o_proj.weight")],
        &attn_heads,
        hidden_dim,
        "deepseek2",
    )?;
    for (h, &r) in hidden.iter_mut().zip(proj_attn.iter()) {
        *h += r;
    }
    Ok(())
}

fn execute_deepseek2_token_with_hidden(
    recipe: &DeepSeek2Recipe,
    mut hidden: Vec<f32>,
    pos: usize,
    seq_len: usize,
    kv_cache: &mut [f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    if recipe.num_layers == 0
        || recipe.hidden_dim == 0
        || recipe.n_heads == 0
        || recipe.mla.kv_lora_rank == 0
        || recipe.mla.qk_nope_head_dim == 0
        || recipe.mla.qk_rope_head_dim < 2
        || recipe.mla.v_head_dim == 0
        || !recipe.rms_eps.is_finite()
        || recipe.rms_eps <= 0.0
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: "deepseek2".to_owned(),
            reason: "DeepSeek2Recipe contains zero or non-finite geometry parameters".to_owned(),
        });
    }
    if hidden.len() != recipe.hidden_dim {
        return Err(K3Error::CapabilityMismatch {
            architecture: "deepseek2".to_owned(),
            reason: format!(
                "layer-0 hidden length {} does not match hidden_dim {}",
                hidden.len(),
                recipe.hidden_dim
            ),
        });
    }

    let layers = recipe.num_layers;
    let hidden_dim = recipe.hidden_dim;
    let eps = recipe.rms_eps;
    let layer_stride = kv_cache.len() / layers;
    let moe_cfg = Some(MoeLayerConfig {
        num_experts: recipe.num_experts,
        top_k: recipe.num_experts_per_tok,
        expert_ffn_dim: recipe.expert_ffn_dim,
        use_shared: recipe.num_shared_experts > 0,
    });

    for layer in 0..layers {
        let layer_kv = &mut kv_cache[layer * layer_stride..(layer + 1) * layer_stride];
        run_deepseek2_mla_sublayer(recipe, layer, pos, seq_len, &mut hidden, layer_kv, weights)?;

        let ffn_norm_w = read_required_1d_f32(
            weights,
            &format!("blk.{layer}.ffn_norm.weight"),
            &[&format!(
                "model.layers.{layer}.post_attention_layernorm.weight"
            )],
            hidden_dim,
            "deepseek2",
        )?;
        let mut norm_ffn_in = vec![0.0_f32; hidden_dim];
        rmsnorm_standard(&mut norm_ffn_in, &hidden, &ffn_norm_w, eps)?;

        let ffn_out = run_layer_moe_or_dense_ffn(
            layer,
            hidden_dim,
            recipe.ffn_dim,
            moe_cfg,
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
        "deepseek2",
    )?;
    let mut normed_final = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut normed_final, &hidden, &out_norm_w, eps)?;
    project_lm_head(weights, &normed_final, logits)
}

impl ArchRecipe for DeepSeek2Recipe {
    fn architecture_id(&self) -> &'static str {
        "deepseek2"
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
            "deepseek2",
            self.hidden_dim,
            weights,
            |hidden, pos, seq_len, kv, logits| {
                execute_deepseek2_token_with_hidden(self, hidden, pos, seq_len, kv, logits, weights)
            },
        )
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        run_decode_with(ctx, "deepseek2", |tok, pos, kv, logits| {
            let hidden = embed_token(tok, self.hidden_dim, weights)?;
            execute_deepseek2_token_with_hidden(self, hidden, pos, pos + 1, kv, logits, weights)
        })
    }
}
