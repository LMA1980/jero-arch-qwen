//! Compute kernel dispatch layer supporting `with-kernels` (`jero-kernels-rvv`) and
//! `--no-default-features` portable scalar builds.

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    BaselineKernel, VectorKernel, apply_rope_inplace, dequantize_token_row, dot_product_f32_scalar,
    gated_deltanet_step_f32, gqa_causal_attention_f32, mla_absorbed_attention_f32,
    pool_and_l2_normalize_f32, swiglu_f32, vec_dot_bf16, vec_dot_f16, vec_dot_i2_s, vec_dot_iq4_nl,
    vec_dot_q2_k, vec_dot_q3_k, vec_dot_q4_0, vec_dot_q4_1, vec_dot_q5_0, vec_dot_q5_1,
    vec_dot_q5_k, vec_dot_q6_k, vec_dot_q8_0,
};

/// Returns `true` when `quant` is supported by the active Qwen / `DeepSeek2` backend.
#[must_use]
pub(crate) const fn is_supported_quant(quant: QuantType) -> bool {
    #[cfg(feature = "with-kernels")]
    {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q4_1
                | QuantType::Q5_0
                | QuantType::Q5_1
                | QuantType::Q2_K
                | QuantType::Q3_K
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
                | QuantType::Q8_0
                | QuantType::IQ4_NL
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16)
    }
}

#[inline]
pub(crate) fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        dot_product_f32_scalar(a, b)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

#[cfg(not(feature = "with-kernels"))]
#[inline]
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    jero_core::f16_bits_to_f32(bits)
}

#[cfg(not(feature = "with-kernels"))]
#[inline]
pub(crate) fn bf16_bits_to_f32(bits: u16) -> f32 {
    let exp = (bits >> 7) & 0x00FF;
    let sign = (bits >> 15) & 0x0001;
    if exp == 0x00FF {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    f32::from_bits(u32::from(bits) << 16)
}

pub(crate) fn rmsnorm(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().rmsnorm(output, input, weight, eps)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if input.is_empty() || output.len() != input.len() || weight.len() != input.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rmsnorm".to_owned(),
                offset: 0,
                byte_len: size_of_val(output),
                buffer_len: size_of_val(input),
                reason: "rmsnorm slice lengths must be non-empty and equal".to_owned(),
            });
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rmsnorm".to_owned(),
                offset: 0,
                byte_len: size_of::<f32>(),
                buffer_len: size_of::<f32>(),
                reason: format!("invalid rmsnorm eps {eps}"),
            });
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "vector length fits cleanly in f32"
        )]
        let mean_sq = input.iter().map(|&v| v * v).sum::<f32>() / (input.len() as f32);
        let inv_rms = 1.0_f32 / (mean_sq + eps).sqrt();
        for ((dst, &x), &w) in output.iter_mut().zip(input.iter()).zip(weight.iter()) {
            *dst = x * inv_rms * w;
        }
        Ok(())
    }
}

pub(crate) fn swiglu(output: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        swiglu_f32(output, gate, up)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if output.is_empty() || output.len() != gate.len() || output.len() != up.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "swiglu".to_owned(),
                offset: 0,
                byte_len: size_of_val(output),
                buffer_len: size_of_val(gate),
                reason: "swiglu slice lengths must be non-empty and equal".to_owned(),
            });
        }
        for ((dst, &g), &u) in output.iter_mut().zip(gate.iter()).zip(up.iter()) {
            let sig = 1.0_f32 / (1.0_f32 + (-g).exp());
            *dst = (g * sig) * u;
        }
        Ok(())
    }
}

pub(crate) fn apply_rope(
    vec: &mut [f32],
    head_dim: usize,
    pos: usize,
    seq_len: usize,
    theta: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        apply_rope_inplace(vec, head_dim, pos, seq_len, theta, None)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let _ = seq_len;
        if head_dim < 2 || head_dim % 2 != 0 || vec.len() % head_dim != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "rope".to_owned(),
                reason: format!(
                    "invalid RoPE dimensions: vec_len={}, head_dim={head_dim}",
                    vec.len()
                ),
            });
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "rope".to_owned(),
                reason: format!("invalid RoPE theta {theta}"),
            });
        }
        let half = head_dim / 2;
        #[expect(clippy::cast_precision_loss, reason = "small position fits in f32")]
        let pos_f = pos as f32;
        #[expect(clippy::cast_precision_loss, reason = "head_dim fits in f32")]
        let hd_f = head_dim as f32;
        for head in vec.chunks_exact_mut(head_dim) {
            for pair in 0..half {
                #[expect(clippy::cast_precision_loss, reason = "pair index fits in f32")]
                let exp = -((2 * pair) as f32) / hd_f;
                let angle = pos_f * theta.powf(exp);
                let (sin_v, cos_v) = angle.sin_cos();
                let x0 = head[pair];
                let x1 = head[pair + half];
                head[pair] = x0 * cos_v - x1 * sin_v;
                head[pair + half] = x0 * sin_v + x1 * cos_v;
            }
        }
        Ok(())
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "matches jero_kernels_rvv::gqa_causal_attention_f32 contract"
)]
pub(crate) fn gqa_causal_attention(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gqa_causal_attention_f32(
            out, q, k_cache, v_cache, n_heads_q, n_heads_kv, head_dim, seq_len, scale,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0 || n_heads_q % n_heads_kv != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "gqa".to_owned(),
                reason: "invalid GQA head geometry".to_owned(),
            });
        }
        let kv_dim = n_heads_kv * head_dim;
        if out.len() != n_heads_q * head_dim
            || q.len() != n_heads_q * head_dim
            || seq_len == 0
            || k_cache.len() < seq_len * kv_dim
            || v_cache.len() < seq_len * kv_dim
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "gqa_kv_cache".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(k_cache),
                reason: "GQA buffer lengths mismatch dimensions".to_owned(),
            });
        }
        let group = n_heads_q / n_heads_kv;
        let mut scores = vec![0.0_f32; seq_len];
        for q_h in 0..n_heads_q {
            let kv_h = q_h / group;
            let q_vec = &q[q_h * head_dim..(q_h + 1) * head_dim];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_vec =
                    &k_cache[t * kv_dim + kv_h * head_dim..t * kv_dim + (kv_h + 1) * head_dim];
                *score = dot_product_f32(q_vec, k_vec) * scale;
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0_f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum_e += *s;
            }
            let inv = if sum_e > 0.0 && sum_e.is_finite() {
                1.0_f32 / sum_e
            } else {
                0.0
            };
            let out_h = &mut out[q_h * head_dim..(q_h + 1) * head_dim];
            out_h.fill(0.0);
            for (t, &s) in scores.iter().enumerate() {
                let p = s * inv;
                let v_vec =
                    &v_cache[t * kv_dim + kv_h * head_dim..t * kv_dim + (kv_h + 1) * head_dim];
                for (dst, &val) in out_h.iter_mut().zip(v_vec.iter()) {
                    *dst += p * val;
                }
            }
        }
        Ok(())
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "matches jero_kernels_rvv::mla_absorbed_attention_f32 contract"
)]
pub(crate) fn mla_absorbed_attention(
    out_latent: &mut [f32],
    q_nope: &[f32],
    q_rotary: &[f32],
    kv_latent_cache: &[f32],
    n_heads: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        mla_absorbed_attention_f32(
            out_latent,
            q_nope,
            q_rotary,
            kv_latent_cache,
            n_heads,
            kv_lora_rank,
            rope_dim,
            seq_len,
            scale,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let stride = kv_lora_rank + rope_dim;
        if n_heads == 0
            || kv_lora_rank == 0
            || seq_len == 0
            || out_latent.len() != n_heads * kv_lora_rank
            || q_nope.len() != n_heads * kv_lora_rank
            || q_rotary.len() != n_heads * rope_dim
            || kv_latent_cache.len() < seq_len * stride
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "mla_latent_cache".to_owned(),
                offset: 0,
                byte_len: size_of_val(out_latent),
                buffer_len: size_of_val(kv_latent_cache),
                reason: "MLA buffer dimensions mismatch".to_owned(),
            });
        }
        let mut scores = vec![0.0_f32; seq_len];
        for h in 0..n_heads {
            let q_abs_h = &q_nope[h * kv_lora_rank..(h + 1) * kv_lora_rank];
            let q_rot_h = &q_rotary[h * rope_dim..(h + 1) * rope_dim];
            for (t, slot) in scores.iter_mut().enumerate() {
                let base = t * stride;
                let c_kv = &kv_latent_cache[base..base + kv_lora_rank];
                let k_r = &kv_latent_cache[base + kv_lora_rank..base + stride];
                *slot = (dot_product_f32(q_abs_h, c_kv) + dot_product_f32(q_rot_h, k_r)) * scale;
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0_f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum_e += *s;
            }
            let inv = if sum_e > 0.0 && sum_e.is_finite() {
                1.0_f32 / sum_e
            } else {
                0.0
            };
            let out_h = &mut out_latent[h * kv_lora_rank..(h + 1) * kv_lora_rank];
            out_h.fill(0.0);
            for (t, &s) in scores.iter().enumerate() {
                let w = s * inv;
                let base = t * stride;
                let c_kv = &kv_latent_cache[base..base + kv_lora_rank];
                for (dst, &val) in out_h.iter_mut().zip(c_kv.iter()) {
                    *dst += w * val;
                }
            }
        }
        Ok(())
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "matches jero_kernels_rvv::gated_deltanet_step_f32 contract"
)]
pub(crate) fn gated_deltanet_step(
    state: &mut [f32],
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay_gate: &[f32],
    beta: &[f32],
    n_heads: usize,
    head_dim: usize,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gated_deltanet_step_f32(state, out, q, k, v, decay_gate, beta, n_heads, head_dim)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let vec_len = n_heads * head_dim;
        let state_len = vec_len * head_dim;
        if n_heads == 0
            || head_dim == 0
            || state.len() != state_len
            || out.len() != vec_len
            || q.len() != vec_len
            || k.len() != vec_len
            || v.len() != vec_len
            || decay_gate.len() != n_heads
            || beta.len() != n_heads
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "deltanet_state".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(state),
                reason: "Gated-DeltaNet buffer dimensions mismatch".to_owned(),
            });
        }
        let head_state = head_dim * head_dim;
        for h in 0..n_heads {
            let decay = decay_gate[h];
            let g_beta = beta[h];
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let k_h = &k[h * head_dim..(h + 1) * head_dim];
            let v_h = &v[h * head_dim..(h + 1) * head_dim];
            let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
            let s_h = &mut state[h * head_state..(h + 1) * head_state];
            for r in 0..head_dim {
                let row = &mut s_h[r * head_dim..(r + 1) * head_dim];
                for entry in &mut *row {
                    *entry *= decay;
                }
                let retrieved = dot_product_f32(row, k_h);
                let delta = g_beta * (v_h[r] - retrieved);
                for (entry, &kv) in row.iter_mut().zip(k_h.iter()) {
                    *entry += delta * kv;
                }
                out_h[r] = dot_product_f32(row, q_h);
            }
        }
        Ok(())
    }
}

pub(crate) fn pool_and_l2_normalize(
    out: &mut [f32],
    hidden_seq: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mode_code: u8,
    matryoshka_dim: Option<usize>,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        pool_and_l2_normalize_f32(
            out,
            hidden_seq,
            seq_len,
            hidden_dim,
            mode_code,
            matryoshka_dim,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if !matches!(mode_code, 1..=3)
            || seq_len == 0
            || hidden_dim == 0
            || hidden_seq.len() != seq_len.saturating_mul(hidden_dim)
            || hidden_seq.iter().any(|v| !v.is_finite())
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "embed_hidden_seq".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: size_of_val(hidden_seq),
                reason: "invalid embedding pooling arguments".to_owned(),
            });
        }
        let eff_dim = match matryoshka_dim {
            None => hidden_dim,
            Some(d) if d > 0 && d <= hidden_dim => d,
            Some(d) => {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "embed_matryoshka_dim".to_owned(),
                    offset: 0,
                    byte_len: d.saturating_mul(size_of::<f32>()),
                    buffer_len: hidden_dim.saturating_mul(size_of::<f32>()),
                    reason: format!("matryoshka_dim ({d}) must lie within 1..={hidden_dim}"),
                });
            }
        };
        if out.len() != eff_dim {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "embed_out".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: eff_dim.saturating_mul(size_of::<f32>()),
                reason: format!("embed_out len {} != effective_dim {eff_dim}", out.len()),
            });
        }
        match mode_code {
            1 => {
                out.copy_from_slice(&hidden_seq[..eff_dim]);
                for s in 1..seq_len {
                    let row = &hidden_seq[s * hidden_dim..s * hidden_dim + eff_dim];
                    for (dst, &val) in out.iter_mut().zip(row.iter()) {
                        *dst += val;
                    }
                }
                #[expect(clippy::cast_precision_loss, reason = "seq_len fits cleanly in f32")]
                let inv = 1.0_f32 / (seq_len as f32);
                for dst in out.iter_mut() {
                    *dst *= inv;
                }
            }
            2 => out.copy_from_slice(&hidden_seq[..eff_dim]),
            3 => {
                let start = (seq_len - 1) * hidden_dim;
                out.copy_from_slice(&hidden_seq[start..start + eff_dim]);
            }
            _ => unreachable!(),
        }
        let sum_sq = dot_product_f32(out, out);
        if sum_sq > 0.0 && sum_sq.is_finite() {
            let inv = 1.0_f32 / sum_sq.sqrt();
            for slot in out.iter_mut() {
                *slot *= inv;
            }
        } else {
            out.fill(0.0);
        }
        Ok(())
    }
}

pub(crate) fn dequantize_row(
    view: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        dequantize_token_row(view, row_idx, out)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let cols = out.len();
        if cols == 0 || view.num_elements() == 0 || view.num_elements() % cols != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: view.as_bytes().len(),
                reason: "invalid embedding row dimensions".to_owned(),
            });
        }
        let num_rows = view.num_elements() / cols;
        if row_idx >= num_rows {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: row_idx,
                byte_len: cols.saturating_mul(size_of::<f32>()),
                buffer_len: view.as_bytes().len(),
                reason: format!("row index {row_idx} out of bounds for {num_rows} rows"),
            });
        }
        let raw = view.as_bytes();
        match view.quant() {
            QuantType::F32 => {
                let base = row_idx * cols * 4;
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = base + i * 4;
                    *slot =
                        f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                }
                Ok(())
            }
            QuantType::F16 => {
                let base = row_idx * cols * 2;
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = base + i * 2;
                    *slot = f16_bits_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
                }
                Ok(())
            }
            QuantType::BF16 => {
                let base = row_idx * cols * 2;
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = base + i * 2;
                    *slot = bf16_bits_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
                }
                Ok(())
            }
            other => Err(K3Error::unsupported_tensor_quant(
                "qwen",
                view.name(),
                other,
            )),
        }
    }
}

pub(crate) fn dispatch_non_f32_matvec(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        match weights.quant() {
            QuantType::F16 => vec_dot_f16(out, weights, input),
            QuantType::BF16 => vec_dot_bf16(out, weights, input),
            QuantType::Q4_0 => vec_dot_q4_0(out, weights, input),
            QuantType::Q4_1 => vec_dot_q4_1(out, weights, input),
            QuantType::Q5_0 => vec_dot_q5_0(out, weights, input),
            QuantType::Q5_1 => vec_dot_q5_1(out, weights, input),
            QuantType::Q8_0 => vec_dot_q8_0(out, weights, input),
            QuantType::Q2_K => vec_dot_q2_k(out, weights, input),
            QuantType::Q3_K => vec_dot_q3_k(out, weights, input),
            QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(out, weights, input),
            QuantType::Q5_K => vec_dot_q5_k(out, weights, input),
            QuantType::Q6_K => vec_dot_q6_k(out, weights, input),
            QuantType::IQ4_NL => vec_dot_iq4_nl(out, weights, input),
            QuantType::I2_S => vec_dot_i2_s(out, weights, input),
            other => Err(K3Error::unsupported_tensor_quant(
                "qwen",
                weights.name(),
                other,
            )),
        }
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let raw = weights.as_bytes();
        let rows = out.len();
        let cols = input.len();
        let stride_bytes = cols * 2;
        let mut chunk_buf = [0.0_f32; 256];
        match weights.quant() {
            QuantType::F16 | QuantType::BF16 => {
                let is_bf16 = weights.quant() == QuantType::BF16;
                for (r, slot) in out.iter_mut().enumerate().take(rows) {
                    let row = &raw[r * stride_bytes..(r + 1) * stride_bytes];
                    let mut sum = 0.0_f32;
                    let mut col = 0;
                    while col < cols {
                        let n = (cols - col).min(chunk_buf.len());
                        for (i, dst) in chunk_buf.iter_mut().enumerate().take(n) {
                            let off = (col + i) * 2;
                            let bits = u16::from_le_bytes([row[off], row[off + 1]]);
                            *dst = if is_bf16 {
                                bf16_bits_to_f32(bits)
                            } else {
                                f16_bits_to_f32(bits)
                            };
                        }
                        sum += dot_product_f32(&chunk_buf[..n], &input[col..col + n]);
                        col += n;
                    }
                    *slot = sum;
                }
                Ok(())
            }
            other => Err(K3Error::unsupported_tensor_quant(
                "qwen",
                weights.name(),
                other,
            )),
        }
    }
}
