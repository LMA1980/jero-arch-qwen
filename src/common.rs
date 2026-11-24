//! Shared fail-closed tensor hydration, `matvec_any`, Layer 0 soft-prompt conditioning,
//! and dense attention / `SwiGLU` FFN sublayers for Qwen and `DeepSeek2` recipes.

use jero_core::{ExecutionContext, K3Error, ModelMetadata, ModelWeights, QuantType, TensorView};
#[cfg(not(miri))]
use rayon::prelude::*;

use crate::kernels_compat::{
    apply_rope, dequantize_row, dispatch_non_f32_matvec, dot_product_f32, gqa_causal_attention,
    is_supported_quant as compat_is_supported_quant, rmsnorm, swiglu,
};

/// Returns `true` if `quant` is supported by the Qwen / `DeepSeek2` execution backend.
#[must_use]
pub const fn is_supported_quant(quant: QuantType) -> bool {
    compat_is_supported_quant(quant)
}

/// Clamps and rounds `raw` down to a positive even head dimension (`>= 2`) for `RoPE`.
#[must_use]
pub const fn even_head_dim(raw: usize) -> usize {
    let clamped = if raw < 2 { 2 } else { raw };
    if clamped % 2 == 0 {
        clamped
    } else {
        clamped - 1
    }
}

/// Attention head geometry for grouped-query attention (`GQA`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct GqaHeadDims {
    pub(crate) n_heads_q: usize,
    pub(crate) n_heads_kv: usize,
    pub(crate) head_dim: usize,
}

/// Dense attention and `SwiGLU` feed-forward layer specification shared across recipes.
#[derive(Debug, Clone, Copy)]
pub struct DenseAttnLayerSpec {
    /// Model embedding and residual stream dimension.
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension.
    pub ffn_dim: usize,
    /// Number of query attention heads.
    pub n_heads_q: usize,
    /// Number of key/value attention heads (`GQA`/`MQA`).
    pub n_heads_kv: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Standard `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency.
    pub rope_theta: f32,
    /// Whether `attn_{q,k,v}.bias` tensors are required (`Qwen2`).
    pub use_qkv_bias: bool,
    /// Whether per-head `attn_{q,k}_norm.weight` tensors are required (`Qwen3`, `Qwen3.5`).
    pub use_qk_norm: bool,
}

pub(crate) fn bounds_err(
    tensor_name: &str,
    offset: usize,
    byte_len: usize,
    buffer_len: usize,
    reason: impl Into<String>,
) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: tensor_name.to_owned(),
        offset,
        byte_len,
        buffer_len,
        reason: reason.into(),
    }
}

pub(crate) fn resolve_tensor<'a>(
    weights: &'a dyn ModelWeights,
    primary: &str,
    aliases: &[&str],
) -> Result<TensorView<'a>, K3Error> {
    if weights.contains_tensor(primary) {
        return weights.tensor(primary);
    }
    for &alias in aliases {
        if weights.contains_tensor(alias) {
            return weights.tensor(alias);
        }
    }
    weights.tensor(primary)
}

pub(crate) fn ensure_quant_supported(
    architecture: &str,
    view: &TensorView<'_>,
) -> Result<(), K3Error> {
    if is_supported_quant(view.quant()) {
        Ok(())
    } else {
        Err(K3Error::unsupported_tensor_quant(
            architecture,
            view.name(),
            view.quant(),
        ))
    }
}

pub(crate) fn ensure_all_finite(tensor_name: &str, data: &[f32]) -> Result<(), K3Error> {
    for (idx, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(bounds_err(
                tensor_name,
                idx.saturating_mul(size_of::<f32>()),
                size_of::<f32>(),
                size_of_val(data),
                format!("non-finite f32 value ({val}) in '{tensor_name}' at index {idx}"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn decode_view_elements_f32(
    view: &TensorView<'_>,
    expected_len: usize,
    architecture: &str,
) -> Result<Vec<f32>, K3Error> {
    ensure_quant_supported(architecture, view)?;
    if expected_len == 0 || view.num_elements() != expected_len {
        return Err(bounds_err(
            view.name(),
            view.offset(),
            view.byte_len(),
            view.as_bytes().len(),
            format!(
                "tensor '{}' element count {} != expected {expected_len}",
                view.name(),
                view.num_elements()
            ),
        ));
    }

    let mut out = vec![0.0_f32; expected_len];
    if view.quant() == QuantType::F32 {
        let raw = view.as_bytes();
        let req = expected_len.saturating_mul(size_of::<f32>());
        if raw.len() < req {
            return Err(bounds_err(
                view.name(),
                view.offset(),
                req,
                raw.len(),
                format!(
                    "F32 tensor '{}' byte length {} < required {req}",
                    view.name(),
                    raw.len()
                ),
            ));
        }
        for (dst, chunk) in out.iter_mut().zip(raw.chunks_exact(size_of::<f32>())) {
            *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    } else if expected_len % view.quant().block_elements() == 0 {
        dequantize_row(view, 0, &mut out)?;
    } else {
        return Err(bounds_err(
            view.name(),
            view.offset(),
            view.byte_len(),
            view.as_bytes().len(),
            format!(
                "tensor '{}' length {expected_len} is not divisible by {} block size {}",
                view.name(),
                view.quant(),
                view.quant().block_elements()
            ),
        ));
    }

    ensure_all_finite(view.name(), &out)?;
    Ok(out)
}

pub(crate) fn read_required_1d_f32(
    weights: &dyn ModelWeights,
    primary: &str,
    aliases: &[&str],
    expected_len: usize,
    architecture: &str,
) -> Result<Vec<f32>, K3Error> {
    if expected_len == 0 {
        return Err(bounds_err(
            primary,
            0,
            0,
            0,
            format!("expected non-zero 1D length for '{primary}'"),
        ));
    }
    let view = resolve_tensor(weights, primary, aliases)?;
    let shape = view.shape();
    let valid_1d =
        shape == [expected_len] || shape == [1, expected_len] || shape == [expected_len, 1];
    if !valid_1d {
        return Err(bounds_err(
            view.name(),
            view.offset(),
            view.byte_len(),
            view.as_bytes().len(),
            format!(
                "tensor '{}' shape {shape:?} does not match expected 1D length {expected_len}",
                view.name()
            ),
        ));
    }
    decode_view_elements_f32(&view, expected_len, architecture)
}

pub(crate) fn read_required_matrix_f32(
    weights: &dyn ModelWeights,
    primary: &str,
    aliases: &[&str],
    expected_len: usize,
    architecture: &str,
) -> Result<Vec<f32>, K3Error> {
    if expected_len == 0 {
        return Err(bounds_err(
            primary,
            0,
            0,
            0,
            format!("expected non-zero matrix length for '{primary}'"),
        ));
    }
    let view = resolve_tensor(weights, primary, aliases)?;
    ensure_quant_supported(architecture, &view)?;
    if view.num_elements() != expected_len {
        return Err(bounds_err(
            view.name(),
            view.offset(),
            view.byte_len(),
            view.as_bytes().len(),
            format!(
                "tensor '{}' element count {} != expected {expected_len}",
                view.name(),
                view.num_elements()
            ),
        ));
    }
    if view.quant() == QuantType::F32 || expected_len % view.quant().block_elements() == 0 {
        return decode_view_elements_f32(&view, expected_len, architecture);
    }

    let shape = view.shape();
    let row_len = match shape {
        [cols, ..] if *cols > 0 && expected_len % *cols == 0 => *cols,
        _ => expected_len,
    };
    let row_count = expected_len / row_len;
    let mut out = vec![0.0_f32; expected_len];
    for r in 0..row_count {
        dequantize_row(&view, r, &mut out[r * row_len..(r + 1) * row_len])?;
    }
    ensure_all_finite(view.name(), &out)?;
    Ok(out)
}

fn validate_matvec_dimensions(
    weights: &TensorView<'_>,
    out_len: usize,
    in_len: usize,
) -> Result<(), K3Error> {
    if out_len == 0 || in_len == 0 {
        return Err(bounds_err(
            weights.name(),
            weights.offset(),
            weights.byte_len(),
            weights.as_bytes().len(),
            "matvec output and input slices cannot be empty",
        ));
    }
    let expected = out_len.checked_mul(in_len).ok_or_else(|| {
        bounds_err(
            weights.name(),
            weights.offset(),
            usize::MAX,
            weights.as_bytes().len(),
            "matvec dimension product overflowed usize",
        )
    })?;
    let shape = weights.shape();
    if shape.len() == 2 {
        let (d0, d1) = (shape[0], shape[1]);
        if !((d0 == in_len && d1 == out_len) || (d0 == out_len && d1 == in_len)) {
            return Err(bounds_err(
                weights.name(),
                weights.offset(),
                weights.byte_len(),
                weights.as_bytes().len(),
                format!(
                    "tensor '{}' shape [{d0}, {d1}] does not match out_len={out_len}, in_len={in_len}",
                    weights.name()
                ),
            ));
        }
    } else if weights.num_elements() != expected {
        return Err(bounds_err(
            weights.name(),
            weights.offset(),
            weights.byte_len(),
            weights.as_bytes().len(),
            format!(
                "tensor '{}' elements {} != out({out_len}) * in({in_len})",
                weights.name(),
                weights.num_elements()
            ),
        ));
    }
    Ok(())
}

fn matvec_f32(out: &mut [f32], weights: &TensorView<'_>, input: &[f32]) -> Result<(), K3Error> {
    let rows = out.len();
    let cols = input.len();
    let expected_elems = rows.checked_mul(cols).ok_or_else(|| {
        bounds_err(
            weights.name(),
            weights.offset(),
            usize::MAX,
            weights.as_bytes().len(),
            "F32 matvec dimension overflowed usize",
        )
    })?;
    if weights.num_elements() != expected_elems {
        return Err(bounds_err(
            weights.name(),
            weights.offset(),
            weights.byte_len(),
            weights.as_bytes().len(),
            format!(
                "F32 tensor elements {} != out({rows}) * in({cols})",
                weights.num_elements()
            ),
        ));
    }
    let raw = weights.as_bytes();
    let required_bytes = expected_elems
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| {
            bounds_err(
                weights.name(),
                weights.offset(),
                usize::MAX,
                raw.len(),
                "F32 matvec byte length overflowed usize",
            )
        })?;
    if raw.len() < required_bytes {
        return Err(bounds_err(
            weights.name(),
            weights.offset(),
            required_bytes,
            raw.len(),
            format!(
                "F32 buffer length {} < required {required_bytes}",
                raw.len()
            ),
        ));
    }

    let mut chunk_buf = [0.0_f32; 256];
    let stride_bytes = size_of_val(input);
    for (r, out_slot) in out.iter_mut().enumerate() {
        let row_bytes = &raw[r * stride_bytes..(r + 1) * stride_bytes];
        let mut row_sum = 0.0_f32;
        let mut col = 0;
        while col < cols {
            let n = (cols - col).min(chunk_buf.len());
            for (i, dst) in chunk_buf.iter_mut().enumerate().take(n) {
                let off = (col + i) * size_of::<f32>();
                *dst = f32::from_le_bytes([
                    row_bytes[off],
                    row_bytes[off + 1],
                    row_bytes[off + 2],
                    row_bytes[off + 3],
                ]);
            }
            row_sum += dot_product_f32(&chunk_buf[..n], &input[col..col + n]);
            col += n;
        }
        *out_slot = row_sum;
    }
    Ok(())
}

fn matvec_any_slice(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    if weights.quant() == QuantType::F32 {
        matvec_f32(out, weights, input)
    } else {
        dispatch_non_f32_matvec(out, weights, input)
    }
}

/// Dispatches matrix-vector multiplication (`out = weights * input`) across all supported
/// quantization formats, fanning out independent output row slices across Rayon threads when
/// beneficial.
///
/// Supports both GGUF `[in_dim, out_dim]` and `SafeTensors` `[out_dim, in_dim]` 2D layouts.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if slices are empty, dimensions mismatch, or backing
/// bytes are truncated, and [`K3Error::CapabilityMismatch`] on unsupported quantization formats.
pub fn matvec_any(out: &mut [f32], weights: &TensorView<'_>, input: &[f32]) -> Result<(), K3Error> {
    validate_matvec_dimensions(weights, out.len(), input.len())?;

    #[cfg(not(miri))]
    if out.len() >= 64 && weights.shape().len() == 2 && weights.quant().scale_trailer_bytes() == 0 {
        let n_threads = rayon::current_num_threads();
        if n_threads > 1 {
            let chunk_rows = out.len().div_ceil(n_threads).clamp(16, 2048);
            let shape = weights.shape();
            if shape[1] == out.len() && shape[0] == input.len() {
                out.par_chunks_mut(chunk_rows).enumerate().try_for_each(
                    |(chunk_idx, out_chunk)| {
                        let row_start = chunk_idx * chunk_rows;
                        let chunk_view = weights.slice_rows(row_start, out_chunk.len())?;
                        matvec_any_slice(out_chunk, &chunk_view, input)
                    },
                )?;
                return Ok(());
            }
            if shape[0] == out.len()
                && shape[1] == input.len()
                && input.len() % weights.quant().block_elements() == 0
            {
                let row_bytes = (input.len() / weights.quant().block_elements())
                    * weights.quant().block_bytes();
                let raw = weights.as_bytes();
                out.par_chunks_mut(chunk_rows).enumerate().try_for_each(
                    |(chunk_idx, out_chunk)| {
                        let row_start = chunk_idx * chunk_rows;
                        let chunk_shape = [out_chunk.len(), input.len()];
                        let chunk_view = TensorView::from_bytes(
                            weights.name(),
                            weights.quant(),
                            &chunk_shape,
                            row_start * row_bytes,
                            raw,
                        )?;
                        matvec_any_slice(out_chunk, &chunk_view, input)
                    },
                )?;
                return Ok(());
            }
        }
    }

    matvec_any_slice(out, weights, input)
}

pub(crate) fn project_required(
    weights: &dyn ModelWeights,
    primary: &str,
    aliases: &[&str],
    input: &[f32],
    out_len: usize,
    architecture: &str,
) -> Result<Vec<f32>, K3Error> {
    let view = resolve_tensor(weights, primary, aliases)?;
    ensure_quant_supported(architecture, &view)?;
    let mut out = vec![0.0_f32; out_len];
    matvec_any(&mut out, &view, input)?;
    Ok(out)
}

/// Selects the top-`k` experts deterministically from `router_logits` with ascending index
/// tie-breaking, returning `(expert_idx, gating_weight)` pairs.
///
/// When `normalize` is `true`, the selected top-`k` weights are renormalized to sum to `1.0`.
/// When `normalize` is `false`, raw full-router softmax probabilities are returned.
#[must_use]
pub fn select_top_k_experts(
    router_logits: &[f32],
    top_k: usize,
    normalize: bool,
) -> Vec<(usize, f32)> {
    if router_logits.is_empty() || top_k == 0 {
        return Vec::new();
    }

    let max_logit = router_logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs = Vec::with_capacity(router_logits.len());
    let mut sum_exp = 0.0_f32;
    for &logit in router_logits {
        let e = if logit.is_finite() && max_logit.is_finite() {
            (logit - max_logit).exp()
        } else {
            0.0_f32
        };
        probs.push(e);
        sum_exp += e;
    }

    if sum_exp > 0.0_f32 && sum_exp.is_finite() {
        for p in &mut probs {
            *p /= sum_exp;
        }
    } else {
        #[expect(
            clippy::cast_precision_loss,
            reason = "expert count fits comfortably in f32"
        )]
        let uniform = 1.0_f32 / (router_logits.len() as f32);
        probs.fill(uniform);
    }

    let mut ranked: Vec<(usize, f32, f32)> = router_logits
        .iter()
        .copied()
        .zip(probs)
        .enumerate()
        .map(|(idx, (logit, prob))| {
            let finite_logit = if logit.is_finite() {
                logit
            } else {
                f32::NEG_INFINITY
            };
            (idx, finite_logit, prob)
        })
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let k = top_k.min(ranked.len());
    let mut selected: Vec<(usize, f32)> = ranked
        .into_iter()
        .take(k)
        .map(|(idx, _, prob)| (idx, prob))
        .collect();

    if normalize {
        let selected_sum: f32 = selected.iter().map(|&(_, w)| w).sum();
        if selected_sum > 0.0_f32 && selected_sum.is_finite() {
            for (_, w) in &mut selected {
                *w /= selected_sum;
            }
        } else {
            #[expect(clippy::cast_precision_loss, reason = "top_k fits comfortably in f32")]
            let uniform = 1.0_f32 / (k as f32);
            for (_, w) in &mut selected {
                *w = uniform;
            }
        }
    }

    selected
}

/// Validates ADR-023 revision separation (`qwen2` QKV bias vs `qwen3`/`qwen3moe` QK-norm)
/// when transformer attention weight tensors are present.
///
/// Supports both GGUF (`blk.0.attn_*`) and `HuggingFace` `SafeTensors` (`model.layers.0.self_attn.*`)
/// naming conventions.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if a `qwen2` model lacks QKV biases or provides
/// QK-norm weights, or if a `qwen3`/`qwen3moe` model lacks QK-norm weights or provides QKV biases.
pub fn validate_qwen_revision_from_probe(
    architecture: &str,
    has_tensor: impl Fn(&str) -> bool,
) -> Result<(), K3Error> {
    let uses_gguf = has_tensor("blk.0.attn_q.weight");
    let uses_hf = !uses_gguf && has_tensor("model.layers.0.self_attn.q_proj.weight");
    if !uses_gguf && !uses_hf {
        return Ok(());
    }

    let (bias_flags, norm_flags) = if uses_gguf {
        (
            [
                has_tensor("blk.0.attn_q.bias"),
                has_tensor("blk.0.attn_k.bias"),
                has_tensor("blk.0.attn_v.bias"),
            ],
            [
                has_tensor("blk.0.attn_q_norm.weight"),
                has_tensor("blk.0.attn_k_norm.weight"),
            ],
        )
    } else {
        (
            [
                has_tensor("model.layers.0.self_attn.q_proj.bias"),
                has_tensor("model.layers.0.self_attn.k_proj.bias"),
                has_tensor("model.layers.0.self_attn.v_proj.bias"),
            ],
            [
                has_tensor("model.layers.0.self_attn.q_norm.weight"),
                has_tensor("model.layers.0.self_attn.k_norm.weight"),
            ],
        )
    };

    let has_all_qkv_bias = bias_flags.iter().all(|&present| present);
    let has_any_qkv_bias = bias_flags.iter().any(|&present| present);
    let has_all_qk_norm = norm_flags.iter().all(|&present| present);
    let has_any_qk_norm = norm_flags.iter().any(|&present| present);

    match architecture {
        "qwen2" if !has_all_qkv_bias || has_any_qk_norm => Err(K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: "ADR-023 revision violation: qwen2 requires blk.0.attn_{q,k,v}.bias and forbids attn_{q,k}_norm.weight".to_owned(),
        }),
        "qwen3" | "qwen3moe" | "qwen3-embedding" if !has_all_qk_norm || has_any_qkv_bias => {
            Err(K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: format!(
                    "ADR-023 revision violation: {architecture} requires blk.0.attn_{{q,k}}_norm.weight and forbids attn_{{q,k,v}}.bias"
                ),
            })
        }
        _ => Ok(()),
    }
}

pub(crate) fn validate_model_weights_adr023(
    architecture: &str,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    validate_qwen_revision_from_probe(architecture, |name| weights.contains_tensor(name))
}

pub(crate) fn validate_model_quant_and_adr023(
    architecture_id: &str,
    supports_quant: impl Fn(QuantType) -> bool,
    metadata: &ModelMetadata,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    for desc in metadata.tensors() {
        if !supports_quant(desc.quant()) {
            return Err(K3Error::unsupported_tensor_quant(
                metadata.architecture(),
                desc.name(),
                desc.quant(),
            ));
        }
        if weights.tensor_count() > 0
            && let Ok(view) = weights.tensor(desc.name())
            && !supports_quant(view.quant())
        {
            return Err(K3Error::unsupported_tensor_quant(
                metadata.architecture(),
                view.name(),
                view.quant(),
            ));
        }
    }
    validate_model_weights_adr023(architecture_id, weights)
}

#[expect(
    clippy::too_many_arguments,
    reason = "shared autoregressive transformer layer loop parameterized by FFN block"
)]
pub(crate) fn execute_standard_token_with_ffn<F>(
    arch: &str,
    num_layers: usize,
    attn: DenseAttnLayerSpec,
    mut hidden: Vec<f32>,
    pos: usize,
    seq_len: usize,
    kv_cache: &mut [f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
    mut ffn_fn: F,
) -> Result<(), K3Error>
where
    F: FnMut(usize, &[f32]) -> Result<Vec<f32>, K3Error>,
{
    validate_model_weights_adr023(arch, weights)?;
    if num_layers == 0 || attn.hidden_dim == 0 || attn.n_heads_q == 0 || attn.n_heads_kv == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "invalid model dimensions: num_layers={num_layers}, hidden_dim={}, n_heads_q={}, n_heads_kv={}",
                attn.hidden_dim, attn.n_heads_q, attn.n_heads_kv
            ),
        });
    }
    if !attn.rms_eps.is_finite() || attn.rms_eps <= 0.0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("rms_eps must be finite and positive, got {}", attn.rms_eps),
        });
    }
    if hidden.len() != attn.hidden_dim {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "layer-0 hidden length {} does not match hidden_dim {}",
                hidden.len(),
                attn.hidden_dim
            ),
        });
    }

    let hidden_dim = attn.hidden_dim;
    let eps = attn.rms_eps;
    let layer_stride = kv_cache.len() / num_layers;

    for layer in 0..num_layers {
        let layer_kv = if layer_stride > 0 {
            &mut kv_cache[layer * layer_stride..(layer + 1) * layer_stride]
        } else {
            &mut []
        };
        run_dense_attention_sublayer(attn, layer, pos, seq_len, &mut hidden, layer_kv, weights)?;

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

        let ffn_out = ffn_fn(layer, &norm_ffn_in)?;
        if ffn_out.len() != hidden_dim {
            return Err(bounds_err(
                "ffn_out",
                0,
                size_of_val(ffn_out.as_slice()),
                hidden_dim.saturating_mul(size_of::<f32>()),
                format!(
                    "FFN output length {} != hidden_dim {hidden_dim}",
                    ffn_out.len()
                ),
            ));
        }
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
    project_lm_head(weights, &normed_final, logits)?;
    Ok(())
}

/// Applies standard `RMSNorm` (`output = input / sqrt(mean(input^2) + eps) * weight`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if slice lengths are empty or mismatched, or if `eps`
/// is invalid.
pub fn rmsnorm_standard(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    rmsnorm(output, input, weight, eps)
}

fn apply_per_head_standard_rmsnorm(
    heads: &mut [f32],
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    if head_dim == 0 || weight.len() != head_dim || heads.len() % head_dim != 0 {
        return Err(bounds_err(
            "qk_norm",
            0,
            size_of_val(heads),
            size_of_val(weight),
            format!(
                "invalid per-head rmsnorm dimensions: heads_len={}, head_dim={head_dim}, weight_len={}",
                heads.len(),
                weight.len()
            ),
        ));
    }
    let mut tmp = vec![0.0_f32; head_dim];
    for head in heads.chunks_exact_mut(head_dim) {
        rmsnorm(&mut tmp, head, weight, eps)?;
        head.copy_from_slice(&tmp);
    }
    Ok(())
}

/// Looks up and dequantizes a single token row from `token_embd.weight` (or `model.embed_tokens.weight`).
///
/// Fails closed per ADR-045 if the embedding table is missing, uses an unsupported quantization
/// format, does not match `hidden_dim`, `token_id` is out of bounds, or decoded elements are
/// non-finite.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] on missing,
/// out-of-bounds, non-finite, or unsupported embedding tensors.
pub fn embed_token(
    token_id: u32,
    hidden_dim: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    if hidden_dim == 0 {
        return Err(bounds_err(
            "token_embd.weight",
            0,
            0,
            0,
            "hidden_dim must be greater than zero",
        ));
    }
    let embd = resolve_tensor(weights, "token_embd.weight", &["model.embed_tokens.weight"])?;
    ensure_quant_supported("qwen", &embd)?;

    let total_elems = embd.num_elements();
    if total_elems == 0 || total_elems % hidden_dim != 0 {
        return Err(bounds_err(
            embd.name(),
            embd.offset(),
            embd.byte_len(),
            embd.as_bytes().len(),
            format!(
                "embedding elements ({total_elems}) is not a positive multiple of hidden_dim ({hidden_dim})"
            ),
        ));
    }
    let row_count = total_elems / hidden_dim;
    let token_idx = usize::try_from(token_id).map_err(|_| {
        bounds_err(
            embd.name(),
            0,
            embd.byte_len(),
            embd.as_bytes().len(),
            format!("token_id {token_id} exceeds usize range"),
        )
    })?;
    if token_idx >= row_count {
        return Err(bounds_err(
            embd.name(),
            token_idx,
            hidden_dim.saturating_mul(size_of::<f32>()),
            embd.as_bytes().len(),
            format!("token_id {token_id} out of bounds for embedding vocab rows {row_count}"),
        ));
    }

    let mut hidden = vec![0.0_f32; hidden_dim];
    dequantize_row(&embd, token_idx, &mut hidden)?;
    ensure_all_finite(embd.name(), &hidden)?;
    Ok(hidden)
}

fn slice_and_matvec_lm_head(
    view: &TensorView<'_>,
    logits: &mut [f32],
    normed_final: &[f32],
    weight_rows: usize,
    is_gguf_layout: bool,
) -> Result<(), K3Error> {
    if logits.len() > weight_rows {
        return Err(bounds_err(
            view.name(),
            view.offset(),
            view.byte_len(),
            view.as_bytes().len(),
            format!(
                "requested logits length {} exceeds LM head row count {weight_rows}",
                logits.len()
            ),
        ));
    }
    if logits.len() == weight_rows {
        return matvec_any(logits, view, normed_final);
    }
    if is_gguf_layout {
        let sub_view = view.slice_rows(0, logits.len())?;
        return matvec_any(logits, &sub_view, normed_final);
    }
    let hidden_dim = normed_final.len();
    let block_elems = view.quant().block_elements();
    if hidden_dim % block_elems != 0 {
        return Err(bounds_err(
            view.name(),
            view.offset(),
            view.byte_len(),
            view.as_bytes().len(),
            format!(
                "LM head hidden_dim {hidden_dim} not divisible by {} block size {block_elems}",
                view.quant()
            ),
        ));
    }
    let sliced_shape = [logits.len(), hidden_dim];
    let sub_view =
        TensorView::from_bytes(view.name(), view.quant(), &sliced_shape, 0, view.as_bytes())?;
    matvec_any(logits, &sub_view, normed_final)
}

/// Projects `normed_final` through `output.weight` / `lm_head.weight` or tied `token_embd.weight` /
/// `model.embed_tokens.weight` into `logits`.
///
/// Returns `Ok(())` immediately if `logits` is empty (e.g. intermediate prefill chunks where
/// logits materialization is skipped). Otherwise enforces ADR-045 fail-closed lookup and supports
/// zero-copy row slicing when the weight table has more rows than `logits.len()`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] if the LM head
/// tensor is missing, dimension-mismatched, or uses an unsupported quantization format.
pub fn project_lm_head(
    weights: &dyn ModelWeights,
    normed_final: &[f32],
    logits: &mut [f32],
) -> Result<(), K3Error> {
    if logits.is_empty() {
        return Ok(());
    }
    let hidden_dim = normed_final.len();
    if hidden_dim == 0 {
        return Err(bounds_err(
            "output.weight",
            0,
            0,
            0,
            "normed_final hidden slice cannot be empty",
        ));
    }

    let view = resolve_tensor(
        weights,
        "output.weight",
        &[
            "lm_head.weight",
            "token_embd.weight",
            "model.embed_tokens.weight",
        ],
    )?;
    ensure_quant_supported("qwen", &view)?;

    let shape = view.shape();
    if shape.len() != 2 {
        return Err(bounds_err(
            view.name(),
            view.offset(),
            view.byte_len(),
            view.as_bytes().len(),
            format!(
                "LM head tensor '{}' must be 2D, got shape {shape:?}",
                view.name()
            ),
        ));
    }

    if shape[0] == hidden_dim && shape[1] > 0 {
        return slice_and_matvec_lm_head(&view, logits, normed_final, shape[1], true);
    }
    if shape[1] == hidden_dim && shape[0] > 0 {
        return slice_and_matvec_lm_head(&view, logits, normed_final, shape[0], false);
    }

    Err(bounds_err(
        view.name(),
        view.offset(),
        view.byte_len(),
        view.as_bytes().len(),
        format!(
            "LM head tensor '{}' shape {shape:?} incompatible with hidden_dim={hidden_dim}",
            view.name()
        ),
    ))
}

/// Adds `bias_name` to `target` if the tensor is present in `weights`, failing closed if the
/// tensor exists with invalid dimensions, unsupported quantization, or non-finite values.
///
/// # Errors
///
/// Returns [`K3Error`] if `bias_name` exists in `weights` but fails bounds, quantization, or
/// finiteness validation.
pub fn add_bias_if_present(
    target: &mut [f32],
    weights: &dyn ModelWeights,
    bias_name: &str,
) -> Result<(), K3Error> {
    if !weights.contains_tensor(bias_name) {
        return Ok(());
    }
    let bias = read_required_1d_f32(weights, bias_name, &[], target.len(), "qwen")?;
    for (dst, &b) in target.iter_mut().zip(bias.iter()) {
        *dst += b;
    }
    Ok(())
}

pub(crate) fn add_required_bias(
    target: &mut [f32],
    weights: &dyn ModelWeights,
    primary: &str,
    aliases: &[&str],
    architecture: &str,
) -> Result<(), K3Error> {
    let bias = read_required_1d_f32(weights, primary, aliases, target.len(), architecture)?;
    for (dst, &b) in target.iter_mut().zip(bias.iter()) {
        *dst += b;
    }
    Ok(())
}

fn run_gqa_causal_with_cache_scaled(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    layer_kv: &mut [f32],
    pos: usize,
    dims: GqaHeadDims,
    attn_scale_override: Option<f32>,
) -> Result<Vec<f32>, K3Error> {
    let q_dim = q.len();
    let kv_dim = k.len();
    if kv_dim == 0 || dims.head_dim == 0 || dims.n_heads_q == 0 || dims.n_heads_kv == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "qwen".to_owned(),
            reason: "GQA attention requires non-zero head geometry and kv_dim".to_owned(),
        });
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let scale = attn_scale_override.unwrap_or_else(|| 1.0_f32 / (dims.head_dim as f32).sqrt());
    let mut attn_out = vec![0.0_f32; q_dim];

    let slot_stride = 2 * kv_dim;
    let capacity_slots = layer_kv.len() / slot_stride;
    if capacity_slots == 0 {
        return Err(bounds_err(
            "kv_cache",
            0,
            slot_stride.saturating_mul(size_of::<f32>()),
            size_of_val(layer_kv),
            format!(
                "layer KV cache length {} floats is smaller than 1 token slot ({slot_stride} floats)",
                layer_kv.len()
            ),
        ));
    }

    let slot = pos % capacity_slots;
    let active_len = (pos + 1).min(capacity_slots);
    let ring_elems = capacity_slots * kv_dim;
    let (k_ring, v_ring) = layer_kv[..2 * ring_elems].split_at_mut(ring_elems);
    k_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(k);
    v_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(v);
    gqa_causal_attention(
        &mut attn_out,
        q,
        &k_ring[..active_len * kv_dim],
        &v_ring[..active_len * kv_dim],
        dims.n_heads_q,
        dims.n_heads_kv,
        dims.head_dim,
        active_len,
        scale,
    )?;
    Ok(attn_out)
}

/// Runs a dense `SwiGLU` feed-forward block (`gate`, `up`, `down`) for `layer`.
///
/// # Errors
///
/// Returns [`K3Error`] if `hidden_dim == 0`, `ffn_dim == 0`, `norm_ffn_in.len() != hidden_dim`,
/// any required weight tensor is missing or mismatched, or kernel execution fails.
pub fn run_dense_swiglu_ffn(
    layer: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    norm_ffn_in: &[f32],
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    if hidden_dim == 0 || ffn_dim == 0 || norm_ffn_in.len() != hidden_dim {
        return Err(bounds_err(
            "ffn",
            0,
            size_of_val(norm_ffn_in),
            hidden_dim.saturating_mul(size_of::<f32>()),
            format!(
                "invalid dense FFN dimensions: hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, input_len={}",
                norm_ffn_in.len()
            ),
        ));
    }
    let gate = project_required(
        weights,
        &format!("blk.{layer}.ffn_gate.weight"),
        &[&format!("model.layers.{layer}.mlp.gate_proj.weight")],
        norm_ffn_in,
        ffn_dim,
        "qwen",
    )?;
    let up = project_required(
        weights,
        &format!("blk.{layer}.ffn_up.weight"),
        &[&format!("model.layers.{layer}.mlp.up_proj.weight")],
        norm_ffn_in,
        ffn_dim,
        "qwen",
    )?;
    let mut act = vec![0.0_f32; ffn_dim];
    swiglu(&mut act, &gate, &up)?;
    project_required(
        weights,
        &format!("blk.{layer}.ffn_down.weight"),
        &[&format!("model.layers.{layer}.mlp.down_proj.weight")],
        &act,
        hidden_dim,
        "qwen",
    )
}

/// Runs a dense causal `GQA` attention sublayer with unit attention and residual scaling.
///
/// # Errors
///
/// Returns [`K3Error`] if any required norm, projection, bias, or KV cache invariant fails.
pub(crate) fn run_dense_attention_sublayer(
    spec: DenseAttnLayerSpec,
    layer: usize,
    pos: usize,
    seq_len: usize,
    hidden: &mut [f32],
    layer_kv: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    run_dense_attention_sublayer_scaled(
        spec, layer, pos, seq_len, hidden, layer_kv, weights, None, 1.0,
    )
}

struct QkvTensors {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn prepare_attention_qkv(
    spec: DenseAttnLayerSpec,
    layer: usize,
    pos: usize,
    seq_len: usize,
    norm_hidden: &[f32],
    weights: &dyn ModelWeights,
) -> Result<QkvTensors, K3Error> {
    let head_d_even = even_head_dim(spec.head_dim);
    let q_dim = spec.n_heads_q * head_d_even;
    let kv_dim = spec.n_heads_kv * head_d_even;

    let mut q = project_required(
        weights,
        &format!("blk.{layer}.attn_q.weight"),
        &[&format!("model.layers.{layer}.self_attn.q_proj.weight")],
        norm_hidden,
        q_dim,
        "qwen",
    )?;
    let mut k = project_required(
        weights,
        &format!("blk.{layer}.attn_k.weight"),
        &[&format!("model.layers.{layer}.self_attn.k_proj.weight")],
        norm_hidden,
        kv_dim,
        "qwen",
    )?;
    let mut v = project_required(
        weights,
        &format!("blk.{layer}.attn_v.weight"),
        &[&format!("model.layers.{layer}.self_attn.v_proj.weight")],
        norm_hidden,
        kv_dim,
        "qwen",
    )?;

    if spec.use_qkv_bias {
        add_required_bias(
            &mut q,
            weights,
            &format!("blk.{layer}.attn_q.bias"),
            &[&format!("model.layers.{layer}.self_attn.q_proj.bias")],
            "qwen2",
        )?;
        add_required_bias(
            &mut k,
            weights,
            &format!("blk.{layer}.attn_k.bias"),
            &[&format!("model.layers.{layer}.self_attn.k_proj.bias")],
            "qwen2",
        )?;
        add_required_bias(
            &mut v,
            weights,
            &format!("blk.{layer}.attn_v.bias"),
            &[&format!("model.layers.{layer}.self_attn.v_proj.bias")],
            "qwen2",
        )?;
    }

    if spec.use_qk_norm {
        let q_norm_w = read_required_1d_f32(
            weights,
            &format!("blk.{layer}.attn_q_norm.weight"),
            &[&format!("model.layers.{layer}.self_attn.q_norm.weight")],
            head_d_even,
            "qwen3",
        )?;
        let k_norm_w = read_required_1d_f32(
            weights,
            &format!("blk.{layer}.attn_k_norm.weight"),
            &[&format!("model.layers.{layer}.self_attn.k_norm.weight")],
            head_d_even,
            "qwen3",
        )?;
        apply_per_head_standard_rmsnorm(&mut q, head_d_even, &q_norm_w, spec.rms_eps)?;
        apply_per_head_standard_rmsnorm(&mut k, head_d_even, &k_norm_w, spec.rms_eps)?;
    }

    apply_rope(
        &mut q,
        head_d_even,
        pos,
        seq_len.max(pos + 1),
        spec.rope_theta,
    )?;
    apply_rope(
        &mut k,
        head_d_even,
        pos,
        seq_len.max(pos + 1),
        spec.rope_theta,
    )?;
    Ok(QkvTensors { q, k, v })
}

/// Runs a dense causal `GQA` attention sublayer with optional `attn_scale_override` and
/// `residual_scale` multipliers.
///
/// # Errors
///
/// Returns [`K3Error`] if any required `RMSNorm`, Q/K/V/O projection, QKV bias, per-head QK-norm,
/// `RoPE`, attention kernel, or KV cache buffer check fails.
#[expect(
    clippy::too_many_arguments,
    reason = "shared GQA attention helper supports explicit attention_scale and residual_scale multipliers"
)]
pub fn run_dense_attention_sublayer_scaled(
    spec: DenseAttnLayerSpec,
    layer: usize,
    pos: usize,
    seq_len: usize,
    hidden: &mut [f32],
    layer_kv: &mut [f32],
    weights: &dyn ModelWeights,
    attn_scale_override: Option<f32>,
    residual_scale: f32,
) -> Result<(), K3Error> {
    let hidden_dim = hidden.len();
    if hidden_dim == 0 || spec.n_heads_q == 0 || spec.n_heads_kv == 0 || spec.head_dim < 2 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "qwen".to_owned(),
            reason: format!(
                "invalid DenseAttnLayerSpec geometry: hidden_dim={hidden_dim}, n_heads_q={}, n_heads_kv={}, head_dim={}",
                spec.n_heads_q, spec.n_heads_kv, spec.head_dim
            ),
        });
    }
    let attn_norm_w = read_required_1d_f32(
        weights,
        &format!("blk.{layer}.attn_norm.weight"),
        &[&format!("model.layers.{layer}.input_layernorm.weight")],
        hidden_dim,
        "qwen",
    )?;
    let mut norm_hidden = vec![0.0_f32; hidden_dim];
    rmsnorm_standard(&mut norm_hidden, hidden, &attn_norm_w, spec.rms_eps)?;

    let head_d_even = even_head_dim(spec.head_dim);
    let qkv = prepare_attention_qkv(spec, layer, pos, seq_len, &norm_hidden, weights)?;

    let attn_out = run_gqa_causal_with_cache_scaled(
        &qkv.q,
        &qkv.k,
        &qkv.v,
        layer_kv,
        pos,
        GqaHeadDims {
            n_heads_q: spec.n_heads_q,
            n_heads_kv: spec.n_heads_kv,
            head_dim: head_d_even,
        },
        attn_scale_override,
    )?;
    let proj_attn = project_required(
        weights,
        &format!("blk.{layer}.attn_output.weight"),
        &[&format!("model.layers.{layer}.self_attn.o_proj.weight")],
        &attn_out,
        hidden_dim,
        "qwen",
    )?;
    for (h, &r) in hidden.iter_mut().zip(proj_attn.iter()) {
        *h += r * residual_scale;
    }
    Ok(())
}

/// Validates and constructs Layer 0 hidden state vectors from optional `ExecutionContext::soft_prompt_embeds()`
/// followed by text tokens in `jump_forward_tokens()` or `prompt_tokens()`.
///
/// Returns `(layer0_hiddens, start_pos, seq_end)`.
pub(crate) fn collect_layer0_sequence_hidden(
    ctx: &ExecutionContext<'_>,
    architecture: &str,
    hidden_dim: usize,
    weights: &dyn ModelWeights,
) -> Result<(Vec<Vec<f32>>, usize, usize), K3Error> {
    let prompt = ctx
        .jump_forward_tokens()
        .unwrap_or_else(|| ctx.prompt_tokens());
    let start_pos = ctx.position();

    let (soft_embeds, num_soft) = if let Some(embeds) = ctx.soft_prompt_embeds() {
        let (n_soft, sp_dim) =
            ctx.soft_prompt_dims()
                .ok_or_else(|| K3Error::CapabilityMismatch {
                    architecture: architecture.to_owned(),
                    reason: "soft_prompt_embeds set without soft_prompt_dims".to_owned(),
                })?;
        if sp_dim != hidden_dim {
            return Err(K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: format!(
                    "soft prompt embed_dim {sp_dim} does not match model hidden_dim {hidden_dim}"
                ),
            });
        }
        let expected_soft_len = n_soft.checked_mul(hidden_dim).ok_or_else(|| {
            bounds_err(
                "soft_prompt_embeds",
                0,
                usize::MAX,
                size_of_val(embeds),
                "soft prompt length overflow",
            )
        })?;
        if embeds.len() != expected_soft_len {
            return Err(bounds_err(
                "soft_prompt_embeds",
                0,
                expected_soft_len.saturating_mul(size_of::<f32>()),
                size_of_val(embeds),
                format!(
                    "soft_prompt_embeds length {} != {n_soft} * {hidden_dim}",
                    embeds.len()
                ),
            ));
        }
        ensure_all_finite("soft_prompt_embeds", embeds)?;
        (Some(embeds), n_soft)
    } else {
        (None, 0)
    };

    let total_new = num_soft
        .checked_add(prompt.len())
        .ok_or(K3Error::ContextLengthExceeded {
            requested_tokens: usize::MAX,
            max_context_tokens: ctx.n_ctx(),
        })?;
    if total_new == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: "prefill prompt_tokens cannot be empty".to_owned(),
        });
    }
    let context_limit = ctx
        .kv_geometry()
        .map_or(ctx.n_ctx(), |g| g.seq_capacity.min(ctx.n_ctx()));
    let seq_end = start_pos
        .checked_add(total_new)
        .ok_or(K3Error::ContextLengthExceeded {
            requested_tokens: usize::MAX,
            max_context_tokens: context_limit,
        })?;
    if seq_end > context_limit {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: seq_end,
            max_context_tokens: context_limit,
        });
    }

    let mut hiddens = Vec::with_capacity(total_new);
    if let Some(embeds) = soft_embeds {
        for s in 0..num_soft {
            hiddens.push(embeds[s * hidden_dim..(s + 1) * hidden_dim].to_vec());
        }
    }
    for &tok in prompt {
        hiddens.push(embed_token(tok, hidden_dim, weights)?);
    }

    Ok((hiddens, start_pos, seq_end))
}

/// Shared Layer 0 soft-prompt + token prefill runner for autoregressive Qwen and `DeepSeek2`
/// recipes.
///
/// # Errors
///
/// Returns [`K3Error`] if soft-prompt dimensions mismatch `hidden_dim`, prompt is empty, context
/// bounds are exceeded, embedding lookup fails, or `step_fn` fails.
pub fn run_prefill_with_hidden<F>(
    ctx: &mut ExecutionContext<'_>,
    architecture: &str,
    hidden_dim: usize,
    weights: &dyn ModelWeights,
    mut step_fn: F,
) -> Result<(), K3Error>
where
    F: FnMut(Vec<f32>, usize, usize, &mut [f32], &mut [f32]) -> Result<(), K3Error>,
{
    let (hiddens, start_pos, seq_end) =
        collect_layer0_sequence_hidden(ctx, architecture, hidden_dim, weights)?;
    let total_new = hiddens.len();
    let compute_logits = ctx.compute_logits();
    let logits_len = if compute_logits {
        ctx.logits().len()
    } else {
        0
    };

    for (step_idx, hidden) in hiddens.into_iter().enumerate() {
        let is_last = step_idx + 1 == total_new;
        let mut step_logits = if is_last && compute_logits {
            vec![0.0_f32; logits_len]
        } else {
            Vec::new()
        };
        step_fn(
            hidden,
            start_pos + step_idx,
            seq_end,
            ctx.kv_cache_mut(),
            &mut step_logits,
        )?;
        if is_last && compute_logits {
            ctx.logits_mut().copy_from_slice(&step_logits);
        }
    }

    ctx.advance_position(total_new)?;
    Ok(())
}

/// Shared token-ID prefill runner across prompt or jump-forward tokens.
///
/// # Errors
///
/// Returns [`K3Error`] if prompt is empty, context bounds are exceeded, or `step_fn` fails.
pub fn run_prefill_with<F>(
    ctx: &mut ExecutionContext<'_>,
    architecture: &str,
    mut step_fn: F,
) -> Result<(), K3Error>
where
    F: FnMut(u32, usize, usize, &mut [f32], &mut [f32]) -> Result<(), K3Error>,
{
    let prompt = ctx
        .jump_forward_tokens()
        .unwrap_or_else(|| ctx.prompt_tokens())
        .to_vec();
    if prompt.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: "prefill prompt_tokens cannot be empty".to_owned(),
        });
    }
    let seq_len = prompt.len();
    let start_pos = ctx.position();
    let context_limit = ctx
        .kv_geometry()
        .map_or(ctx.n_ctx(), |g| g.seq_capacity.min(ctx.n_ctx()));
    let seq_end = start_pos
        .checked_add(seq_len)
        .ok_or(K3Error::ContextLengthExceeded {
            requested_tokens: usize::MAX,
            max_context_tokens: context_limit,
        })?;
    if seq_end > context_limit {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: seq_end,
            max_context_tokens: context_limit,
        });
    }

    let compute_logits = ctx.compute_logits();
    let logits_len = if compute_logits {
        ctx.logits().len()
    } else {
        0
    };
    for (offset, &tok) in prompt.iter().enumerate() {
        let is_last = offset + 1 == seq_len;
        let mut step_logits = if is_last && compute_logits {
            vec![0.0_f32; logits_len]
        } else {
            Vec::new()
        };
        step_fn(
            tok,
            start_pos + offset,
            seq_end,
            ctx.kv_cache_mut(),
            &mut step_logits,
        )?;
        if is_last && compute_logits {
            ctx.logits_mut().copy_from_slice(&step_logits);
        }
    }
    ctx.advance_position(seq_len)?;
    Ok(())
}

/// Shared single-token decode runner consuming [`ExecutionContext::decode_token`].
///
/// # Errors
///
/// Returns [`K3Error`] if no decode token is staged, context bounds are exceeded, or `step_fn`
/// fails.
pub fn run_decode_with<F>(
    ctx: &mut ExecutionContext<'_>,
    architecture: &str,
    mut step_fn: F,
) -> Result<(), K3Error>
where
    F: FnMut(u32, usize, &mut [f32], &mut [f32]) -> Result<(), K3Error>,
{
    let tok = ctx
        .decode_token()
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: "decode_step called without a staged decode token".to_owned(),
        })?;
    let pos = ctx.position();
    let mut step_logits = vec![0.0_f32; ctx.logits().len()];
    step_fn(tok, pos, ctx.kv_cache_mut(), &mut step_logits)?;
    ctx.logits_mut().copy_from_slice(&step_logits);
    ctx.advance_position(1)?;
    Ok(())
}
