use std::collections::BTreeMap;

use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, EmbeddingPooling, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights,
    QuantType, SafeTensorsFile, TensorView,
};

use super::*;

#[derive(Debug)]
struct OwnedTensor {
    quant: QuantType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

struct OwnedWeights {
    tensors: BTreeMap<String, OwnedTensor>,
    total_bytes: usize,
    allow_tied_output: bool,
}

impl OwnedWeights {
    fn new(allow_tied_output: bool) -> Self {
        Self {
            tensors: BTreeMap::new(),
            total_bytes: 0,
            allow_tied_output,
        }
    }

    fn insert_f32(&mut self, name: &str, shape: &[usize], vals: &[f32]) {
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.total_bytes += bytes.len();
        self.tensors.insert(
            name.to_owned(),
            OwnedTensor {
                quant: QuantType::F32,
                shape: shape.to_vec(),
                bytes,
            },
        );
    }

    fn insert_filled(&mut self, name: &str, shape: &[usize], val: f32) {
        let elems: usize = shape.iter().product();
        let vals = vec![val; elems];
        self.insert_f32(name, shape, &vals);
    }

    fn insert_raw(&mut self, name: &str, quant: QuantType, shape: &[usize], bytes: Vec<u8>) {
        self.total_bytes += bytes.len();
        self.tensors.insert(
            name.to_owned(),
            OwnedTensor {
                quant,
                shape: shape.to_vec(),
                bytes,
            },
        );
    }

    fn remove(&mut self, name: &str) {
        if let Some(removed) = self.tensors.remove(name) {
            self.total_bytes = self.total_bytes.saturating_sub(removed.bytes.len());
        }
    }
}

impl ModelWeights for OwnedWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let resolved = if self.tensors.contains_key(name) {
            name
        } else if name == "output.weight"
            && self.allow_tied_output
            && self.tensors.contains_key("token_embd.weight")
        {
            "token_embd.weight"
        } else {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: self.total_bytes,
                reason: format!("tensor '{name}' missing in OwnedWeights"),
            });
        };
        let (resolved_name, entry) =
            self.tensors
                .get_key_value(resolved)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: self.total_bytes,
                    reason: format!("tensor '{name}' missing in OwnedWeights"),
                })?;
        let view_name = if name == "output.weight" && resolved == "token_embd.weight" {
            "output.weight"
        } else {
            resolved_name.as_str()
        };
        TensorView::from_bytes(view_name, entry.quant, &entry.shape, 0, &entry.bytes)
    }

    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.total_bytes
    }
}

fn f32s_to_le_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn build_qwen_dense_weights(
    hidden: usize,
    ffn: usize,
    vocab: usize,
    is_qwen3: bool,
) -> OwnedWeights {
    let mut w = OwnedWeights::new(true);
    let mut emb = vec![0.0_f32; vocab * hidden];
    for (i, slot) in emb.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "small synthetic test matrix")]
        {
            *slot = ((i % 19) as f32 - 9.0) * 0.04;
        }
    }
    w.insert_f32("token_embd.weight", &[hidden, vocab], &emb);
    w.insert_filled("output_norm.weight", &[hidden], 1.0);
    w.insert_filled("blk.0.attn_norm.weight", &[hidden], 1.0);
    w.insert_filled("blk.0.attn_q.weight", &[hidden, hidden], 0.05);
    w.insert_filled("blk.0.attn_k.weight", &[hidden, hidden], 0.04);
    w.insert_filled("blk.0.attn_v.weight", &[hidden, hidden], 0.03);
    w.insert_filled("blk.0.attn_output.weight", &[hidden, hidden], 0.05);
    if is_qwen3 {
        w.insert_filled("blk.0.attn_q_norm.weight", &[hidden / 2], 1.0);
        w.insert_filled("blk.0.attn_k_norm.weight", &[hidden / 2], 1.0);
    } else {
        w.insert_filled("blk.0.attn_q.bias", &[hidden], 0.02);
        w.insert_filled("blk.0.attn_k.bias", &[hidden], -0.01);
        w.insert_filled("blk.0.attn_v.bias", &[hidden], 0.03);
    }
    w.insert_filled("blk.0.ffn_norm.weight", &[hidden], 1.0);
    w.insert_filled("blk.0.ffn_gate.weight", &[hidden, ffn], 0.04);
    w.insert_filled("blk.0.ffn_up.weight", &[hidden, ffn], 0.03);
    w.insert_filled("blk.0.ffn_down.weight", &[ffn, hidden], 0.04);
    w
}

fn build_qwen3_moe_weights(
    hidden: usize,
    expert_ffn: usize,
    num_experts: usize,
    vocab: usize,
    winners: &[usize],
) -> OwnedWeights {
    let mut w = OwnedWeights::new(true);
    let mut emb = vec![0.0_f32; vocab * hidden];
    for (i, slot) in emb.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "small synthetic test matrix")]
        {
            *slot = ((i % 17) as f32 - 5.0) * 0.03 + 0.15;
        }
    }
    w.insert_f32("token_embd.weight", &[hidden, vocab], &emb);
    w.insert_filled("output_norm.weight", &[hidden], 1.0);
    w.insert_filled("blk.0.attn_norm.weight", &[hidden], 1.0);
    w.insert_filled("blk.0.attn_q.weight", &[hidden, hidden], 0.05);
    w.insert_filled("blk.0.attn_k.weight", &[hidden, hidden], 0.04);
    w.insert_filled("blk.0.attn_v.weight", &[hidden, hidden], 0.03);
    w.insert_filled("blk.0.attn_q_norm.weight", &[hidden / 2], 1.0);
    w.insert_filled("blk.0.attn_k_norm.weight", &[hidden / 2], 1.0);
    w.insert_filled("blk.0.attn_output.weight", &[hidden, hidden], 0.05);
    w.insert_filled("blk.0.ffn_norm.weight", &[hidden], 1.0);

    let mut router = vec![-1.0_f32; num_experts * hidden];
    for (idx, exp) in winners.iter().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "small expert rank fits in f32")]
        let score = 3.0 + (winners.len() - idx) as f32 * 0.25;
        for c in 0..hidden {
            router[exp * hidden + c] = score;
        }
    }
    w.insert_f32("blk.0.ffn_gate_inp.weight", &[hidden, num_experts], &router);
    let mut gate_stack = Vec::with_capacity(num_experts * hidden * expert_ffn);
    let mut up_stack = Vec::with_capacity(num_experts * hidden * expert_ffn);
    let mut down_stack = Vec::with_capacity(num_experts * expert_ffn * hidden);
    for exp in 0..num_experts {
        #[expect(clippy::cast_precision_loss, reason = "small expert index fits in f32")]
        let scale = ((exp % 7) as f32 + 1.0) * 0.02;
        gate_stack.extend(std::iter::repeat_n(scale, hidden * expert_ffn));
        up_stack.extend(std::iter::repeat_n(0.1, hidden * expert_ffn));
        down_stack.extend(std::iter::repeat_n(0.05, expert_ffn * hidden));
    }
    w.insert_f32(
        "blk.0.ffn_gate_exps.weight",
        &[hidden, expert_ffn, num_experts],
        &gate_stack,
    );
    w.insert_f32(
        "blk.0.ffn_up_exps.weight",
        &[hidden, expert_ffn, num_experts],
        &up_stack,
    );
    w.insert_f32(
        "blk.0.ffn_down_exps.weight",
        &[expert_ffn, hidden, num_experts],
        &down_stack,
    );
    w
}

fn build_deepseek2_test_recipe_and_weights() -> (DeepSeek2Recipe, OwnedWeights) {
    let recipe = DeepSeek2Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 12,
        n_heads: 2,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 10_000.0,
        mla: DeepSeek2MlaConfig {
            kv_lora_rank: 4,
            q_lora_rank: None,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 2,
            v_head_dim: 4,
        },
        num_experts: 8,
        num_experts_per_tok: 2,
        num_shared_experts: 1,
        expert_ffn_dim: 8,
    };
    let mut w = OwnedWeights::new(true);
    let mut emb = vec![0.0_f32; 16 * 8];
    for (i, slot) in emb.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "small synthetic test matrix")]
        {
            *slot = 0.1 + ((i % 9) as f32) * 0.02;
        }
    }
    w.insert_f32("token_embd.weight", &[8, 16], &emb);
    w.insert_filled("output_norm.weight", &[8], 1.0);
    w.insert_filled("blk.0.attn_norm.weight", &[8], 1.0);
    w.insert_filled("blk.0.attn_q.weight", &[8, 8], 0.05);
    w.insert_filled("blk.0.attn_q_rope.weight", &[8, 4], 0.04);
    w.insert_filled("blk.0.attn_kv_a_mqa.weight", &[8, 6], 0.06);
    w.insert_filled("blk.0.attn_kv_a_norm.weight", &[4], 1.0);
    w.insert_filled("blk.0.attn_w_uk.weight", &[4, 8], 0.05);
    w.insert_filled("blk.0.attn_w_uv.weight", &[4, 8], 0.05);
    w.insert_filled("blk.0.attn_output.weight", &[8, 8], 0.05);
    w.insert_filled("blk.0.ffn_norm.weight", &[8], 1.0);
    w.insert_filled("blk.0.ffn_gate_shexp.weight", &[8, 8], 0.08);
    w.insert_filled("blk.0.ffn_up_shexp.weight", &[8, 8], 0.05);
    w.insert_filled("blk.0.ffn_down_shexp.weight", &[8, 8], 0.04);
    w.insert_filled("blk.0.ffn_gate_inp.weight", &[8, 8], 0.1);
    w.insert_f32(
        "blk.0.exp_probs_b.bias",
        &[8],
        &[0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
    );
    w.insert_filled("blk.0.ffn_gate_exps.weight", &[8, 8, 8], 0.03);
    w.insert_filled("blk.0.ffn_up_exps.weight", &[8, 8, 8], 0.02);
    w.insert_filled("blk.0.ffn_down_exps.weight", &[8, 8, 8], 0.03);
    (recipe, w)
}

fn build_qwen35_weights(
    hidden: usize,
    ffn: usize,
    vocab: usize,
    num_layers: usize,
    is_moe: bool,
) -> OwnedWeights {
    let mut w = OwnedWeights::new(true);
    let mut emb = vec![0.0_f32; vocab * hidden];
    for (i, slot) in emb.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "small synthetic test matrix")]
        {
            *slot = 0.08 + ((i % 13) as f32) * 0.02;
        }
    }
    w.insert_f32("token_embd.weight", &[hidden, vocab], &emb);
    w.insert_filled("output_norm.weight", &[hidden], 1.0);

    for layer in 0..num_layers {
        let is_full = (layer + 1) % 4 == 0;
        w.insert_filled(&format!("blk.{layer}.attn_norm.weight"), &[hidden], 1.0);
        w.insert_filled(
            &format!("blk.{layer}.attn_q.weight"),
            &[hidden, hidden],
            0.05,
        );
        w.insert_filled(
            &format!("blk.{layer}.attn_k.weight"),
            &[hidden, hidden],
            0.04,
        );
        w.insert_filled(
            &format!("blk.{layer}.attn_v.weight"),
            &[hidden, hidden],
            0.03,
        );
        w.insert_filled(
            &format!("blk.{layer}.attn_output.weight"),
            &[hidden, hidden],
            0.05,
        );
        if is_full {
            w.insert_filled(
                &format!("blk.{layer}.attn_q_norm.weight"),
                &[hidden / 2],
                1.0,
            );
            w.insert_filled(
                &format!("blk.{layer}.attn_k_norm.weight"),
                &[hidden / 2],
                1.0,
            );
        } else {
            w.insert_filled(&format!("blk.{layer}.ssm_a.weight"), &[hidden, 2], 0.06);
            w.insert_filled(&format!("blk.{layer}.ssm_beta.weight"), &[hidden, 2], 0.04);
        }
        w.insert_filled(&format!("blk.{layer}.ffn_norm.weight"), &[hidden], 1.0);
        if is_moe {
            w.insert_filled(
                &format!("blk.{layer}.ffn_gate_shexp.weight"),
                &[hidden, 8],
                0.06,
            );
            w.insert_filled(
                &format!("blk.{layer}.ffn_up_shexp.weight"),
                &[hidden, 8],
                0.04,
            );
            w.insert_filled(
                &format!("blk.{layer}.ffn_down_shexp.weight"),
                &[8, hidden],
                0.05,
            );
            w.insert_filled(
                &format!("blk.{layer}.ffn_gate_inp.weight"),
                &[hidden, 8],
                0.09,
            );
            w.insert_filled(
                &format!("blk.{layer}.ffn_gate_exps.weight"),
                &[hidden, 8, 8],
                0.03,
            );
            w.insert_filled(
                &format!("blk.{layer}.ffn_up_exps.weight"),
                &[hidden, 8, 8],
                0.03,
            );
            w.insert_filled(
                &format!("blk.{layer}.ffn_down_exps.weight"),
                &[8, hidden, 8],
                0.03,
            );
        } else {
            w.insert_filled(
                &format!("blk.{layer}.ffn_gate.weight"),
                &[hidden, ffn],
                0.04,
            );
            w.insert_filled(&format!("blk.{layer}.ffn_up.weight"), &[hidden, ffn], 0.03);
            w.insert_filled(
                &format!("blk.{layer}.ffn_down.weight"),
                &[ffn, hidden],
                0.04,
            );
        }
    }
    w
}

fn build_gguf_and_safetensors_from_owned(
    arch: &str,
    extra_kvs: &[(&str, GgufValue<'_>)],
    owned: &OwnedWeights,
    use_hf_names_for_safetensors: bool,
) -> (Vec<u8>, Vec<u8>) {
    let specs: Vec<(&str, QuantType, &[usize])> = owned
        .tensors
        .iter()
        .map(|(name, entry)| (name.as_str(), entry.quant, entry.shape.as_slice()))
        .collect();
    let mut gguf_bytes = create_synthetic_gguf_v3_with_kvs(arch, 64, extra_kvs, &specs);

    // Overwrite synthetic placeholder payload in gguf_bytes with exact OwnedWeights bytes
    let offsets: Vec<(usize, &[u8])> = {
        let parsed = GgufFile::parse(&gguf_bytes).expect("parse scaffold GGUF");
        owned
            .tensors
            .iter()
            .map(|(name, entry)| {
                let view = parsed.tensor(name).expect("tensor exists in scaffold");
                (view.offset(), entry.bytes.as_slice())
            })
            .collect()
    };
    for (offset, bytes) in offsets {
        gguf_bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    let map_name = |name: &str| -> String {
        if !use_hf_names_for_safetensors {
            return name.to_owned();
        }
        match name {
            "token_embd.weight" => "model.embed_tokens.weight".to_owned(),
            "output_norm.weight" => "model.norm.weight".to_owned(),
            "output.weight" => "lm_head.weight".to_owned(),
            other => other
                .replace("blk.", "model.layers.")
                .replace(".attn_norm.", ".input_layernorm.")
                .replace(".ffn_norm.", ".post_attention_layernorm.")
                .replace(".attn_q.", ".self_attn.q_proj.")
                .replace(".attn_k.", ".self_attn.k_proj.")
                .replace(".attn_v.", ".self_attn.v_proj.")
                .replace(".attn_output.", ".self_attn.o_proj.")
                .replace(".attn_q_norm.", ".self_attn.q_norm.")
                .replace(".attn_k_norm.", ".self_attn.k_norm.")
                .replace(".ffn_gate.", ".mlp.gate_proj.")
                .replace(".ffn_up.", ".mlp.up_proj.")
                .replace(".ffn_down.", ".mlp.down_proj."),
        }
    };

    let st_entries: Vec<(String, QuantType, Vec<usize>, &[u8])> = owned
        .tensors
        .iter()
        .map(|(name, entry)| {
            let mapped = map_name(name);
            let shape = if use_hf_names_for_safetensors && entry.shape.len() == 2 {
                vec![entry.shape[1], entry.shape[0]]
            } else {
                entry.shape.clone()
            };
            (mapped, entry.quant, shape, entry.bytes.as_slice())
        })
        .collect();
    let st_refs: Vec<(&str, QuantType, &[usize], &[u8])> = st_entries
        .iter()
        .map(|(n, q, s, b)| (n.as_str(), *q, s.as_slice(), *b))
        .collect();
    let st_bytes = create_synthetic_safetensors(&[("general.architecture", arch)], &st_refs);

    (gguf_bytes, st_bytes)
}

#[test]
fn bootstrap_metadata_matches_policy() {
    assert_eq!(CRATE_NAME, "jero-arch-qwen");
    assert_eq!(
        CRATE_LICENSE,
        "LGPL-3.0-only WITH LGPL-3.0-linking-exception"
    );
    assert_ne!(crate_description(), "");
    assert_eq!(jero_core::DEFAULT_ALIGNMENT, 32);
}

#[test]
fn test_qwen2_vs_qwen3_and_tied_embeddings() {
    let q2: QwenRecipe = Qwen2Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let q3 = Qwen3Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };

    let q2_w = build_qwen_dense_weights(8, 16, 16, false);
    let q3_w = build_qwen_dense_weights(8, 16, 16, true);
    let out_alias = q3_w
        .tensor("output.weight")
        .expect("tied output.weight alias");
    assert_eq!(out_alias.shape(), &[8, 16]);

    let mut kv2 = vec![0.0_f32; 32 * 2 * 8];
    let mut logits2 = vec![0.0_f32; 16];
    let mut ctx2 = ExecutionContext::new(&[1, 3], 16, &mut logits2, &mut kv2).unwrap();
    q2.prefill_step(&mut ctx2, &q2_w).expect("qwen2 prefill");
    ctx2.prepare_decode(5).unwrap();
    q2.decode_step(&mut ctx2, &q2_w).expect("qwen2 decode");
    assert_eq!(ctx2.position(), 3);
    assert!(logits2.iter().all(|v| v.is_finite()) && logits2.iter().any(|v| v.abs() > 1e-6));

    let mut kv3 = vec![0.0_f32; 32 * 2 * 8];
    let mut logits3 = vec![0.0_f32; 16];
    let mut ctx3 = ExecutionContext::new(&[1, 3], 16, &mut logits3, &mut kv3).unwrap();
    q3.prefill_step(&mut ctx3, &q3_w).expect("qwen3 prefill");
    assert!(logits3.iter().all(|v| v.is_finite()) && logits3.iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn test_qwen2_vs_qwen3_adr023_separation() {
    let qwen2_kvs = [
        ("qwen2.block_count", GgufValue::Uint32(1)),
        ("qwen2.embedding_length", GgufValue::Uint32(16)),
        ("qwen2.feed_forward_length", GgufValue::Uint32(32)),
        ("qwen2.attention.head_count", GgufValue::Uint32(2)),
        ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
    ];
    let qwen2_specs = [
        ("blk.0.attn_q.weight", QuantType::F32, &[16, 16][..]),
        ("blk.0.attn_q.bias", QuantType::F32, &[16][..]),
        ("blk.0.attn_k.bias", QuantType::F32, &[16][..]),
        ("blk.0.attn_v.bias", QuantType::F32, &[16][..]),
    ];
    let q2_bytes = create_synthetic_gguf_v3_with_kvs("qwen2", 64, &qwen2_kvs, &qwen2_specs);
    let q2_gguf = GgufFile::parse(&q2_bytes).expect("parse qwen2");
    Qwen2Recipe::new()
        .validate_model(q2_gguf.metadata(), &q2_gguf)
        .expect("valid qwen2 passes validate_model");

    let wrong_q2_specs = [
        ("blk.0.attn_q.weight", QuantType::F32, &[16, 16][..]),
        ("blk.0.attn_q_norm.weight", QuantType::F32, &[8][..]),
        ("blk.0.attn_k_norm.weight", QuantType::F32, &[8][..]),
    ];
    let bad_q2_bytes = create_synthetic_gguf_v3_with_kvs("qwen2", 64, &qwen2_kvs, &wrong_q2_specs);
    let bad_q2_gguf = GgufFile::parse(&bad_q2_bytes).expect("parse mismatched qwen2");
    assert!(matches!(
        Qwen2Recipe::new().validate_model(bad_q2_gguf.metadata(), &bad_q2_gguf),
        Err(K3Error::CapabilityMismatch { .. })
    ));

    let qwen3_kvs = [
        ("qwen3.block_count", GgufValue::Uint32(1)),
        ("qwen3.embedding_length", GgufValue::Uint32(16)),
        ("qwen3.feed_forward_length", GgufValue::Uint32(32)),
        ("qwen3.attention.head_count", GgufValue::Uint32(2)),
        ("qwen3.attention.head_count_kv", GgufValue::Uint32(2)),
    ];
    let q3_bytes = create_synthetic_gguf_v3_with_kvs("qwen3", 64, &qwen3_kvs, &wrong_q2_specs);
    let q3_gguf = GgufFile::parse(&q3_bytes).expect("parse valid qwen3");
    Qwen3Recipe::new()
        .validate_model(q3_gguf.metadata(), &q3_gguf)
        .expect("valid qwen3 passes validate_model");

    let bad_q3_bytes = create_synthetic_gguf_v3_with_kvs("qwen3", 64, &qwen3_kvs, &qwen2_specs);
    let bad_q3_gguf = GgufFile::parse(&bad_q3_bytes).expect("parse mismatched qwen3");
    assert!(matches!(
        Qwen3Recipe::new().validate_model(bad_q3_gguf.metadata(), &bad_q3_gguf),
        Err(K3Error::CapabilityMismatch { .. })
    ));
}

#[test]
fn test_qwen2_qwen3_capability_mismatch_on_wrong_tensors() {
    let q2_recipe = Qwen2Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let q3_recipe = Qwen3Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };

    let q2_weights = build_qwen_dense_weights(8, 16, 16, false);
    let q3_weights = build_qwen_dense_weights(8, 16, 16, true);
    let mut kv = vec![0.0_f32; 64];
    let mut logits = vec![0.0_f32; 16];

    let mut ctx_a = ExecutionContext::new(&[1], 8, &mut logits, &mut kv).unwrap();
    let err_a = q2_recipe
        .prefill_step(&mut ctx_a, &q3_weights)
        .expect_err("Qwen2Recipe must reject qwen3 QK-norm tensors");
    assert!(matches!(err_a, K3Error::CapabilityMismatch { .. }));

    let mut ctx_b = ExecutionContext::new(&[1], 8, &mut logits, &mut kv).unwrap();
    let err_b = q3_recipe
        .prefill_step(&mut ctx_b, &q2_weights)
        .expect_err("Qwen3Recipe must reject qwen2 QKV bias tensors");
    assert!(matches!(err_b, K3Error::CapabilityMismatch { .. }));
}

#[test]
fn test_qwen3moe_128_top8_dispatch_and_forward_pass() {
    let (num_experts, expert_ffn, winners): (usize, usize, &[usize]) = if cfg!(miri) {
        (16, 4, &[1, 3, 5, 7, 9, 11, 13, 15])
    } else {
        (128, 12, &[5, 19, 31, 47, 63, 79, 95, 111])
    };
    let w = build_qwen3_moe_weights(8, expert_ffn, num_experts, 16, winners);
    let norm_in = vec![0.4_f32; 8];
    let mut out = vec![0.0_f32; 8];
    let selected =
        dispatch_qwen3_moe_ffn(&mut out, &norm_in, 0, num_experts, 8, expert_ffn, false, &w)
            .expect("top-8 MoE dispatch succeeds");
    assert_eq!(selected.len(), 8);
    let weight_sum: f32 = selected.iter().map(|&(_, weight)| weight).sum();
    assert!((weight_sum - 1.0).abs() < 1e-4);
    assert_eq!(selected[0].0, winners[0]);
    assert_eq!(selected[7].0, winners[7]);
    assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|v| v.abs() > 1e-6));

    let mut recipe = Qwen3MoeRecipe::new_30b_a3b();
    assert_eq!(recipe.architecture_id(), "qwen3moe");
    assert_eq!(recipe.num_experts, 128);
    assert_eq!(recipe.num_experts_per_tok, 8);
    recipe.base.num_layers = 1;
    recipe.base.hidden_dim = 8;
    recipe.base.ffn_dim = expert_ffn;
    recipe.base.n_heads_q = 2;
    recipe.base.n_heads_kv = 2;
    recipe.base.head_dim = 4;
    recipe.base.vocab_size = 16;
    recipe.num_experts = num_experts;
    recipe.expert_ffn_dim = expert_ffn;

    let mut kv = vec![0.0_f32; 32 * 2 * 8];
    let mut logits = vec![0.0_f32; 16];
    let mut ctx = ExecutionContext::new(&[1, 2], 16, &mut logits, &mut kv).unwrap();
    recipe.prefill_step(&mut ctx, &w).expect("qwen3moe prefill");
    ctx.prepare_decode(4).unwrap();
    recipe.decode_step(&mut ctx, &w).expect("qwen3moe decode");
    assert_eq!(ctx.position(), 3);
    assert!(logits.iter().all(|v| v.is_finite()) && logits.iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn test_deepseek2_mla_matrix_absorption_and_e2e() {
    let n_heads = 2_usize;
    let d_c = 6_usize;
    let d_nope = 4_usize;
    let d_r = 2_usize;
    let d_v = 5_usize;
    let seq_len = 4_usize;
    let latent_dim = d_c + d_r;

    #[expect(clippy::cast_precision_loss, reason = "small test tensors fit in f32")]
    let pattern = |len: usize, modulo: usize, offset: f32, factor: f32| -> Vec<f32> {
        (0..len)
            .map(|i| ((i % modulo) as f32 - offset) * factor)
            .collect()
    };
    let q_nope = pattern(n_heads * d_nope, 7, 3.0, 0.13);
    let q_rotary = pattern(n_heads * d_r, 5, 2.0, 0.17);
    let kv_latent = pattern(seq_len * latent_dim, 11, 5.0, 0.09);
    let k_up_proj = pattern(n_heads * d_nope * d_c, 13, 6.0, 0.07);
    let v_up_proj = pattern(n_heads * d_v * d_c, 9, 4.0, 0.08);

    let mut absorbed_out = vec![0.0_f32; n_heads * d_v];
    run_mla_absorbed_attention(
        &mut absorbed_out,
        &q_nope,
        &q_rotary,
        &kv_latent,
        &k_up_proj,
        &v_up_proj,
        n_heads,
        d_c,
        d_nope,
        d_r,
        d_v,
        seq_len,
    )
    .expect("absorbed MLA succeeds");

    #[expect(
        clippy::cast_precision_loss,
        reason = "small test head_dim fits in f32"
    )]
    let scale = 1.0_f32 / ((d_nope + d_r) as f32).sqrt();
    let mut naive_out = vec![0.0_f32; n_heads * d_v];
    for h in 0..n_heads {
        let q_nope_head = &q_nope[h * d_nope..(h + 1) * d_nope];
        let q_rotary_head = &q_rotary[h * d_r..(h + 1) * d_r];
        let k_up_head = &k_up_proj[h * d_nope * d_c..(h + 1) * d_nope * d_c];
        let v_up_head = &v_up_proj[h * d_v * d_c..(h + 1) * d_v * d_c];

        let mut scores = Vec::with_capacity(seq_len);
        let mut vals = Vec::with_capacity(seq_len * d_v);
        for s in 0..seq_len {
            let c_s = &kv_latent[s * latent_dim..s * latent_dim + d_c];
            let kr_s = &kv_latent[s * latent_dim + d_c..(s + 1) * latent_dim];
            let mut score_nope = 0.0_f32;
            for d in 0..d_nope {
                let row = &k_up_head[d * d_c..(d + 1) * d_c];
                let k_d: f32 = row.iter().zip(c_s.iter()).map(|(&a, &b)| a * b).sum();
                score_nope += q_nope_head[d] * k_d;
            }
            let score_rotary: f32 = q_rotary_head
                .iter()
                .zip(kr_s.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            scores.push((score_nope + score_rotary) * scale);
            for v_i in 0..d_v {
                let row = &v_up_head[v_i * d_c..(v_i + 1) * d_c];
                vals.push(
                    row.iter()
                        .zip(c_s.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>(),
                );
            }
        }
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        let dst = &mut naive_out[h * d_v..(h + 1) * d_v];
        for (s, &e) in exps.iter().enumerate() {
            let p = e / sum_e;
            for (v_i, d_slot) in dst.iter_mut().enumerate() {
                *d_slot += p * vals[s * d_v + v_i];
            }
        }
    }

    let max_diff = absorbed_out
        .iter()
        .zip(naive_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "absorbed MLA must match naive KV decompression (max_diff={max_diff})"
    );
}

#[test]
fn test_deepseek2_recipe_e2e_and_q_lora() {
    let (recipe, w) = build_deepseek2_test_recipe_and_weights();
    let _: DeepSeek2MlaRecipe = recipe.clone();
    assert_eq!(recipe.architecture_id(), "deepseek2");
    let mut kv = vec![0.0_f32; 16 * 6];
    let mut logits = vec![0.0_f32; 16];
    let mut ctx = ExecutionContext::new(&[1, 2, 3], 16, &mut logits, &mut kv).unwrap();
    recipe
        .prefill_step(&mut ctx, &w)
        .expect("deepseek2 prefill");
    assert_eq!(ctx.position(), 3);
    ctx.prepare_decode(4).unwrap();
    recipe.decode_step(&mut ctx, &w).expect("deepseek2 decode");
    assert_eq!(ctx.position(), 4);
    assert!(logits.iter().all(|v| v.is_finite()) && logits.iter().any(|v| v.abs() > 1e-6));
    assert!(kv[..4 * 6].iter().any(|v| v.abs() > 1e-6));

    // Also verify compressed query (q_lora_rank: Some(4)) path
    let (mut q_lora_recipe, mut q_lora_w) = build_deepseek2_test_recipe_and_weights();
    q_lora_recipe.mla.q_lora_rank = Some(4);
    q_lora_w.remove("blk.0.attn_q.weight");
    q_lora_w.remove("blk.0.attn_q_rope.weight");
    q_lora_w.insert_filled("blk.0.attn_q_a.weight", &[8, 4], 0.05);
    q_lora_w.insert_filled("blk.0.attn_q_a_norm.weight", &[4], 1.0);
    q_lora_w.insert_filled("blk.0.attn_q_b.weight", &[4, 12], 0.04);
    let mut kv_q = vec![0.0_f32; 16 * 6];
    let mut logits_q = vec![0.0_f32; 16];
    let mut ctx_q = ExecutionContext::new(&[1, 2], 16, &mut logits_q, &mut kv_q).unwrap();
    q_lora_recipe
        .prefill_step(&mut ctx_q, &q_lora_w)
        .expect("deepseek2 q_lora_rank prefill");
    assert!(logits_q.iter().all(|v| v.is_finite()) && logits_q.iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn test_qwen35_hybrid_gated_deltanet_and_moe_recipes() {
    let recipe = Qwen35Recipe {
        num_layers: 4,
        hidden_dim: 8,
        ffn_dim: 12,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
        full_attention_interval: 4,
    };
    assert!(!recipe.is_full_attention_layer(0));
    assert!(!recipe.is_full_attention_layer(1));
    assert!(!recipe.is_full_attention_layer(2));
    assert!(recipe.is_full_attention_layer(3));

    let w_dense = build_qwen35_weights(8, 12, 16, 4, false);
    let mut kv = vec![0.0_f32; 4 * 16 * 16];
    let mut logits = vec![0.0_f32; 16];
    let mut ctx = ExecutionContext::new(&[1, 2], 16, &mut logits, &mut kv).unwrap();
    recipe
        .prefill_step(&mut ctx, &w_dense)
        .expect("qwen35 prefill");
    ctx.prepare_decode(3).unwrap();
    recipe
        .decode_step(&mut ctx, &w_dense)
        .expect("qwen35 decode");
    assert_eq!(ctx.position(), 3);
    assert!(logits.iter().all(|v| v.is_finite()) && logits.iter().any(|v| v.abs() > 1e-6));
    assert!(kv[..32].iter().any(|v| v.abs() > 1e-6));

    let moe_recipe = Qwen35MoeRecipe {
        base: recipe,
        num_experts: 8,
        num_experts_per_tok: 2,
        num_shared_experts: 1,
        expert_ffn_dim: 8,
    };
    let w_moe = build_qwen35_weights(8, 12, 16, 4, true);
    let mut kv_m = vec![0.0_f32; 4 * 16 * 16];
    let mut logits_m = vec![0.0_f32; 16];
    let mut ctx_m = ExecutionContext::new(&[2, 4], 16, &mut logits_m, &mut kv_m).unwrap();
    moe_recipe
        .prefill_step(&mut ctx_m, &w_moe)
        .expect("qwen35moe prefill");
    ctx_m.prepare_decode(6).unwrap();
    moe_recipe
        .decode_step(&mut ctx_m, &w_moe)
        .expect("qwen35moe decode");
    assert_eq!(ctx_m.position(), 3);
    assert!(logits_m.iter().all(|v| v.is_finite()) && logits_m.iter().any(|v| v.abs() > 1e-6));
}

#[test]
fn test_adr045_fail_closed_missing_and_corrupted_weights() {
    let q2 = Qwen2Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let q3 = Qwen3Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let empty = OwnedWeights::new(false);
    let mut kv = vec![0.0_f32; 128];
    let mut logits = vec![0.0_f32; 16];

    // 1. Empty weights MUST fail closed on token_embd.weight across all recipes
    let mut ctx = ExecutionContext::new(&[1], 16, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        q2.prefill_step(&mut ctx, &empty),
        Err(K3Error::TensorBoundsError { .. })
    ));
    let mut ctx3 = ExecutionContext::new(&[1], 16, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        q3.prefill_step(&mut ctx3, &empty),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 2. Missing individual required tensors in Qwen2 / Qwen3 fail closed
    for missing in [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_q.bias",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
    ] {
        let mut w = build_qwen_dense_weights(8, 16, 16, false);
        w.remove(missing);
        let mut c = ExecutionContext::new(&[1], 16, &mut logits, &mut kv).unwrap();
        let res = q2.prefill_step(&mut c, &w);
        assert!(
            res.is_err(),
            "Qwen2Recipe must fail closed when '{missing}' is missing"
        );
    }

    // 3. Missing both output.weight and token_embd.weight (or truncated output.weight rows) fails closed
    assert!(matches!(
        project_lm_head(&empty, &[0.1_f32; 8], &mut logits),
        Err(K3Error::TensorBoundsError { .. })
    ));
    let mut w_short_head = build_qwen_dense_weights(8, 16, 16, true);
    w_short_head.insert_filled("output.weight", &[8, 4], 0.1);
    let mut c_short_head = ExecutionContext::new(&[1], 16, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        q3.prefill_step(&mut c_short_head, &w_short_head),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 4. Out-of-bounds token_id fails closed
    let w_ok = build_qwen_dense_weights(8, 16, 16, true);
    let mut c_oob = ExecutionContext::new(&[99], 16, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        q3.prefill_step(&mut c_oob, &w_ok),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 5. NaN weight fails closed
    let mut w_nan = build_qwen_dense_weights(8, 16, 16, true);
    w_nan.insert_f32("output_norm.weight", &[8], &[f32::NAN; 8]);
    let mut c_nan = ExecutionContext::new(&[1], 16, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        q3.prefill_step(&mut c_nan, &w_nan),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_adr045_fail_closed_moe_mla_deltanet_and_quant() {
    let q3 = Qwen3Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let mut kv = vec![0.0_f32; 128];
    let mut logits = vec![0.0_f32; 16];

    // 1. Missing MoE router or expert weights fail closed
    let mut w_no_router = build_qwen3_moe_weights(8, 4, 8, 16, &[0, 1]);
    w_no_router.remove("blk.0.ffn_gate_inp.weight");
    let mut moe_out = [0.0_f32; 8];
    assert!(matches!(
        dispatch_qwen3_moe_ffn(&mut moe_out, &[0.5; 8], 0, 8, 2, 4, false, &w_no_router),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let mut w_no_exp = build_qwen3_moe_weights(8, 4, 8, 16, &[0, 1]);
    w_no_exp.remove("blk.0.ffn_gate_exps.weight");
    assert!(matches!(
        dispatch_qwen3_moe_ffn(&mut moe_out, &[0.5; 8], 0, 8, 2, 4, false, &w_no_exp),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 2. Missing DeepSeek2 MLA attn_w_uk.weight fails closed (no identity fallback)
    let (ds_recipe, mut ds_w) = build_deepseek2_test_recipe_and_weights();
    ds_w.remove("blk.0.attn_w_uk.weight");
    let mut ds_kv = vec![0.0_f32; 16 * 6];
    let mut c_ds = ExecutionContext::new(&[1], 16, &mut logits, &mut ds_kv).unwrap();
    assert!(matches!(
        ds_recipe.prefill_step(&mut c_ds, &ds_w),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 3. Missing Qwen3.5 DeltaNet ssm_a.weight fails closed
    let q35 = Qwen35Recipe {
        num_layers: 4,
        hidden_dim: 8,
        ffn_dim: 12,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
        full_attention_interval: 4,
    };
    let mut w_q35 = build_qwen35_weights(8, 12, 16, 4, false);
    w_q35.remove("blk.0.ssm_a.weight");
    let mut kv35 = vec![0.0_f32; 4 * 16 * 16];
    let mut c_35 = ExecutionContext::new(&[1], 16, &mut logits, &mut kv35).unwrap();
    assert!(matches!(
        q35.prefill_step(&mut c_35, &w_q35),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 4. Unsupported quant (TQ1_0) fails closed
    let mut w_bad_q = build_qwen_dense_weights(8, 16, 16, true);
    w_bad_q.insert_raw(
        "blk.0.attn_q.weight",
        QuantType::TQ1_0,
        &[256, 1],
        vec![0; 54],
    );
    let mut c_bad_q = ExecutionContext::new(&[1], 16, &mut logits, &mut kv).unwrap();
    assert!(matches!(
        q3.prefill_step(&mut c_bad_q, &w_bad_q),
        Err(K3Error::CapabilityMismatch { .. })
    ));

    // 5. select_top_k_experts deterministic ascending tie-breaking
    let tied = [0.1_f32, 3.0, 0.5, 1.0, 3.0, 2.0];
    let top2 = select_top_k_experts(&tied, 2, true);
    assert_eq!(top2.len(), 2);
    assert_eq!(top2[0].0, 1);
    assert_eq!(top2[1].0, 4);
    assert!((top2[0].1 - 0.5).abs() < 1e-6);
    assert!((top2[1].1 - 0.5).abs() < 1e-6);
}

#[test]
fn test_vlm_soft_prompt_conditioning_across_qwen_variants_and_errors() {
    let soft = vec![
        0.25_f32, -0.15, 0.35, 0.05, 0.12, -0.08, 0.22, 0.18, // token 0
        -0.10, 0.40, -0.20, 0.15, 0.30, 0.05, -0.25, 0.11, // token 1
    ];

    // 1. QwenRecipe / Qwen2Recipe (Qwen2-VL / Qwen2.5-VL)
    let q2 = QwenRecipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let q2_w = build_qwen_dense_weights(8, 16, 16, false);
    let mut kv_plain = vec![0.0_f32; 32 * 2 * 8];
    let mut logits_plain = vec![0.0_f32; 16];
    let mut ctx_plain =
        ExecutionContext::new(&[2, 3], 16, &mut logits_plain, &mut kv_plain).unwrap();
    q2.prefill_step(&mut ctx_plain, &q2_w).unwrap();

    let mut kv_soft = vec![0.0_f32; 32 * 2 * 8];
    let mut logits_soft = vec![0.0_f32; 16];
    let mut ctx_soft = ExecutionContext::new(&[2, 3], 16, &mut logits_soft, &mut kv_soft)
        .unwrap()
        .with_soft_prompt_embeds(&soft, 2, 8)
        .unwrap();
    q2.prefill_step(&mut ctx_soft, &q2_w)
        .expect("Qwen2 soft-prompt prefill");
    assert_eq!(ctx_soft.position(), 4);
    assert_ne!(logits_plain.as_slice(), ctx_soft.logits());
    ctx_soft.prepare_decode(5).unwrap();
    q2.decode_step(&mut ctx_soft, &q2_w).unwrap();
    assert_eq!(ctx_soft.position(), 5);

    // 2. Qwen3Recipe
    let q3 = Qwen3Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let q3_w = build_qwen_dense_weights(8, 16, 16, true);
    let mut ctx3 = ExecutionContext::new(&[2], 16, &mut logits_soft, &mut kv_soft)
        .unwrap()
        .with_soft_prompt_embeds(&soft, 2, 8)
        .unwrap();
    q3.prefill_step(&mut ctx3, &q3_w)
        .expect("Qwen3 soft-prompt prefill");
    assert_eq!(ctx3.position(), 3);

    // 3. Qwen3MoeRecipe
    let q3m = Qwen3MoeRecipe {
        base: q3.clone(),
        num_experts: 8,
        num_experts_per_tok: 2,
        expert_ffn_dim: 4,
    };
    let moe_weights = build_qwen3_moe_weights(8, 4, 8, 16, &[0, 3]);
    let mut moe_ctx = ExecutionContext::new(&[1], 16, &mut logits_soft, &mut kv_soft)
        .unwrap()
        .with_soft_prompt_embeds(&soft, 2, 8)
        .unwrap();
    q3m.prefill_step(&mut moe_ctx, &moe_weights)
        .expect("Qwen3Moe soft-prompt prefill");
    assert_eq!(moe_ctx.position(), 3);

    // 4. Qwen35Recipe
    let q35 = Qwen35Recipe {
        num_layers: 4,
        hidden_dim: 8,
        ffn_dim: 12,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
        full_attention_interval: 4,
    };
    let hybrid_weights = build_qwen35_weights(8, 12, 16, 4, false);
    let mut kv35 = vec![0.0_f32; 4 * 16 * 16];
    let mut hybrid_ctx = ExecutionContext::new(&[1], 16, &mut logits_soft, &mut kv35)
        .unwrap()
        .with_soft_prompt_embeds(&soft, 2, 8)
        .unwrap();
    q35.prefill_step(&mut hybrid_ctx, &hybrid_weights)
        .expect("Qwen35 soft-prompt prefill");
    assert_eq!(hybrid_ctx.position(), 3);

    // 5. Negative: mismatched soft_prompt_dims fails closed with CapabilityMismatch
    let wrong_soft = vec![0.1_f32; 2 * 4];
    let mut ctx_bad = ExecutionContext::new(&[1], 16, &mut logits_soft, &mut kv_soft)
        .unwrap()
        .with_soft_prompt_embeds(&wrong_soft, 2, 4)
        .unwrap();
    assert!(matches!(
        q2.prefill_step(&mut ctx_bad, &q2_w),
        Err(K3Error::CapabilityMismatch { .. })
    ));
}

#[test]
fn test_qwen_embedding_recipe_l2_norm_matryoshka_and_errors() {
    let mut emb_recipe = QwenEmbeddingRecipe::new();
    emb_recipe.base.num_layers = 1;
    emb_recipe.base.hidden_dim = 8;
    emb_recipe.base.ffn_dim = 16;
    emb_recipe.base.n_heads_q = 2;
    emb_recipe.base.n_heads_kv = 2;
    emb_recipe.base.head_dim = 4;
    emb_recipe.base.vocab_size = 16;
    assert_eq!(emb_recipe.architecture_id(), "qwen3-embedding");
    assert!(!emb_recipe.requires_kv_cache());

    // Weights have NO output.weight / LM head (tie disabled!) -> embedding recipe must still succeed!
    let mut w = build_qwen_dense_weights(8, 16, 16, true);
    w.allow_tied_output = false;

    // 1. Default LastToken pooling with zero-KV and empty logits (&mut [])
    let mut emb_out = vec![0.0_f32; 8];
    let mut empty_logits: [f32; 0] = [];
    let mut empty_kv: [f32; 0] = [];
    let mut ctx = ExecutionContext::new(&[1, 3, 5], 16, &mut empty_logits, &mut empty_kv)
        .unwrap()
        .with_embedding_output(&mut emb_out, EmbeddingPooling::LastToken, None)
        .unwrap();
    emb_recipe
        .prefill_step(&mut ctx, &w)
        .expect("qwen3-embedding prefill succeeds without LM head or KV cache");
    assert_eq!(ctx.position(), 3);
    let norm_sq: f32 = emb_out.iter().map(|v| v * v).sum();
    assert!(
        (norm_sq.sqrt() - 1.0).abs() < 1e-5,
        "L2 norm must equal 1.0, got {}",
        norm_sq.sqrt()
    );

    // 2. Matryoshka truncation (dim=4) into exact-size buffer and full-size buffer
    let mut mat_exact = vec![0.0_f32; 4];
    let mut ctx_mat_exact = ExecutionContext::new(&[1, 3, 5], 16, &mut empty_logits, &mut empty_kv)
        .unwrap()
        .with_embedding_output(&mut mat_exact, EmbeddingPooling::LastToken, Some(4))
        .unwrap();
    emb_recipe
        .prefill_step(&mut ctx_mat_exact, &w)
        .expect("Matryoshka exact buffer succeeds");
    let mat_norm: f32 = mat_exact.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((mat_norm - 1.0).abs() < 1e-5);

    let mut mat_padded = vec![99.0_f32; 8];
    let mut ctx_mat_padded =
        ExecutionContext::new(&[1, 3, 5], 16, &mut empty_logits, &mut empty_kv)
            .unwrap()
            .with_embedding_output(&mut mat_padded, EmbeddingPooling::LastToken, Some(4))
            .unwrap();
    emb_recipe
        .prefill_step(&mut ctx_mat_padded, &w)
        .expect("Matryoshka padded buffer succeeds");
    assert_float_parity(&mat_padded[..4], &mat_exact[..]);
    assert_eq!(mat_padded[4..], [0.0_f32; 4]);

    // 3. Mean pooling differs from LastToken pooling and still has unit L2 norm
    let mut mean_out = vec![0.0_f32; 8];
    let mut ctx_mean = ExecutionContext::new(&[1, 3, 5], 16, &mut empty_logits, &mut empty_kv)
        .unwrap()
        .with_embedding_output(&mut mean_out, EmbeddingPooling::Mean, None)
        .unwrap();
    emb_recipe.prefill_step(&mut ctx_mean, &w).unwrap();
    assert!((mean_out.iter().map(|v| v * v).sum::<f32>().sqrt() - 1.0).abs() < 1e-5);
    assert_ne!(mean_out, emb_out);

    // 4. Negative bounds & capability checks
    // 4a. Zero-dim Matryoshka rejected by ExecutionContext
    let mut bad_buf = vec![0.0_f32; 8];
    assert!(matches!(
        ExecutionContext::new(&[1], 16, &mut empty_logits, &mut empty_kv)
            .unwrap()
            .with_embedding_output(&mut bad_buf, EmbeddingPooling::LastToken, Some(0)),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 4b. Matryoshka > hidden_dim rejected by QwenEmbeddingRecipe
    let mut big_buf = vec![0.0_f32; 12];
    let mut ctx_big_mat = ExecutionContext::new(&[1], 16, &mut empty_logits, &mut empty_kv)
        .unwrap()
        .with_embedding_output(&mut big_buf, EmbeddingPooling::LastToken, Some(10))
        .unwrap();
    assert!(matches!(
        emb_recipe.prefill_step(&mut ctx_big_mat, &w),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 4c. Missing embedding_output rejected
    let mut ctx_no_emb = ExecutionContext::new(&[1], 16, &mut empty_logits, &mut empty_kv).unwrap();
    assert!(matches!(
        emb_recipe.prefill_step(&mut ctx_no_emb, &w),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 4d. decode_step unsupported
    let mut ctx_dec = ExecutionContext::new(&[1], 16, &mut empty_logits, &mut empty_kv).unwrap();
    ctx_dec.prepare_decode(2).unwrap();
    assert!(matches!(
        emb_recipe.decode_step(&mut ctx_dec, &w),
        Err(K3Error::CapabilityMismatch { .. })
    ));
}

fn assert_float_parity(lhs: &[f32], rhs: &[f32]) {
    assert_eq!(lhs.len(), rhs.len());
    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        #[cfg(not(miri))]
        assert_eq!(a.to_bits(), b.to_bits());
        #[cfg(miri)]
        assert!(
            (a - b).abs() < 1e-5,
            "Miri float parity mismatch: {a} vs {b}"
        );
    }
}

#[test]
fn test_gguf_and_safetensors_parity_and_n3_determinism() {
    // 1. Qwen2 + Layer 0 soft_prompt_embeds parity: GGUF vs SafeTensors (both GGUF & HF tensor names)
    let q2 = Qwen2Recipe {
        num_layers: 1,
        hidden_dim: 8,
        ffn_dim: 16,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        vocab_size: 16,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
    };
    let q2_owned = build_qwen_dense_weights(8, 16, 16, false);
    let q2_kvs = [
        ("qwen2.block_count", GgufValue::Uint32(1)),
        ("qwen2.embedding_length", GgufValue::Uint32(8)),
        ("qwen2.feed_forward_length", GgufValue::Uint32(16)),
        ("qwen2.attention.head_count", GgufValue::Uint32(2)),
        ("qwen2.attention.head_count_kv", GgufValue::Uint32(2)),
    ];
    let (q2_gguf_bytes, q2_st_gguf_names) =
        build_gguf_and_safetensors_from_owned("qwen2", &q2_kvs, &q2_owned, false);
    let (_, q2_st_hf_names) =
        build_gguf_and_safetensors_from_owned("qwen2", &q2_kvs, &q2_owned, true);

    let q2_gguf = GgufFile::parse(&q2_gguf_bytes).unwrap();
    let q2_st_a = SafeTensorsFile::from_bytes(&q2_st_gguf_names).unwrap();
    let q2_st_b = SafeTensorsFile::from_bytes(&q2_st_hf_names).unwrap();

    let soft = [0.2_f32, -0.1, 0.3, 0.05, 0.1, 0.4, -0.2, 0.15];
    let run_q2 = |weights: &dyn ModelWeights| -> Vec<f32> {
        let mut kv = vec![0.0_f32; 32 * 2 * 8];
        let mut logits = vec![0.0_f32; 16];
        let mut ctx = ExecutionContext::new(&[1, 3], 16, &mut logits, &mut kv)
            .unwrap()
            .with_soft_prompt_embeds(&soft, 1, 8)
            .unwrap();
        q2.prefill_step(&mut ctx, weights).unwrap();
        logits
    };
    let gguf_logits = run_q2(&q2_gguf);
    let st_gguf_logits = run_q2(&q2_st_a);
    let st_hf_logits = run_q2(&q2_st_b);
    assert_float_parity(&gguf_logits, &st_gguf_logits);
    assert_float_parity(&gguf_logits, &st_hf_logits);

    // 2. QwenEmbeddingRecipe parity across GGUF, SafeTensors, and N=3 runs
    let mut emb = QwenEmbeddingRecipe::new();
    emb.base.num_layers = 1;
    emb.base.hidden_dim = 8;
    emb.base.ffn_dim = 16;
    emb.base.n_heads_q = 2;
    emb.base.n_heads_kv = 2;
    emb.base.head_dim = 4;
    emb.base.vocab_size = 16;

    let q3_owned = build_qwen_dense_weights(8, 16, 16, true);
    let q3_kvs = [
        ("qwen3.block_count", GgufValue::Uint32(1)),
        ("qwen3.embedding_length", GgufValue::Uint32(8)),
        ("qwen3.feed_forward_length", GgufValue::Uint32(16)),
        ("qwen3.attention.head_count", GgufValue::Uint32(2)),
        ("qwen3.attention.head_count_kv", GgufValue::Uint32(2)),
    ];
    let (q3_gguf_bytes, q3_st_bytes) =
        build_gguf_and_safetensors_from_owned("qwen3", &q3_kvs, &q3_owned, true);
    let q3_gguf = GgufFile::parse(&q3_gguf_bytes).unwrap();
    let q3_st = SafeTensorsFile::from_bytes(&q3_st_bytes).unwrap();

    let run_emb = |weights: &dyn ModelWeights| -> Vec<f32> {
        let mut out = vec![0.0_f32; 4];
        let mut empty_logits: [f32; 0] = [];
        let mut empty_kv: [f32; 0] = [];
        let mut ctx = ExecutionContext::new(&[2, 4, 7], 16, &mut empty_logits, &mut empty_kv)
            .unwrap()
            .with_soft_prompt_embeds(&soft, 1, 8)
            .unwrap()
            .with_embedding_output(&mut out, EmbeddingPooling::LastToken, Some(4))
            .unwrap();
        emb.prefill_step(&mut ctx, weights).unwrap();
        out
    };

    let baseline_gguf = run_emb(&q3_gguf);
    for _ in 0..3 {
        let iter_gguf = run_emb(&q3_gguf);
        let iter_st = run_emb(&q3_st);
        assert_float_parity(&baseline_gguf, &iter_gguf);
        assert_float_parity(&baseline_gguf, &iter_st);
    }
}

#[test]
fn test_f16_matvec_and_parallel_rayon_row_path() {
    // Tests F16 matvec (both with-kernels and --no-default-features) and Rayon out.len() >= 64
    let rows = if cfg!(miri) { 64 } else { 128 };
    let cols = 8;
    // F16 1.0 = 0x3C00
    let raw = [0x00_u8, 0x3C_u8].repeat(rows * cols);
    let st = create_synthetic_safetensors(
        &[("general.architecture", "qwen3")],
        &[("proj.weight", QuantType::F16, &[rows, cols], &raw)],
    );
    let sf = SafeTensorsFile::from_bytes(&st).unwrap();
    let view = sf.tensor("proj.weight").unwrap();
    let input = [1.0_f32; 8];
    let mut out = vec![0.0_f32; rows];
    matvec_any(&mut out, &view, &input).expect("F16 matvec succeeds");
    assert!(out.iter().all(|&v| (v - 8.0).abs() < 1e-4));

    // Also test F32 parallel row path with GGUF [cols, rows] layout
    let f32_bytes = f32s_to_le_bytes(&vec![0.5_f32; rows * cols]);
    let f32_shape = [cols, rows];
    let f32_view =
        TensorView::from_bytes("f32_proj", QuantType::F32, &f32_shape, 0, &f32_bytes).unwrap();
    let mut f32_out = vec![0.0_f32; rows];
    matvec_any(&mut f32_out, &f32_view, &input).expect("F32 parallel matvec succeeds");
    assert!(f32_out.iter().all(|&v| (v - 4.0).abs() < 1e-5));
}

#[cfg(feature = "with-kernels")]
#[test]
fn test_q8_0_quantized_embedding_and_matvec() {
    // 1. Q8_0 matvec_any: 2 rows of 32 elements with scale 0.5 (0x3800) and quants = 2 -> 1.0 each
    let mut q8_bytes = Vec::with_capacity(2 * 34);
    for _ in 0..2 {
        q8_bytes.extend_from_slice(&0x3800_u16.to_le_bytes());
        q8_bytes.resize(q8_bytes.len() + 32, 2_u8);
    }
    let view = TensorView::from_bytes("q8_mat", QuantType::Q8_0, &[32, 2], 0, &q8_bytes).unwrap();
    let input = [1.0_f32; 32];
    let mut out = [0.0_f32; 2];
    matvec_any(&mut out, &view, &input).expect("Q8_0 matvec succeeds");
    assert!((out[0] - 32.0).abs() < 1e-3);
    assert!((out[1] - 32.0).abs() < 1e-3);

    // 2. embed_token with Q8_0 table
    let mut w = OwnedWeights::new(false);
    w.insert_raw("token_embd.weight", QuantType::Q8_0, &[32, 2], q8_bytes);
    let row = embed_token(1, 32, &w).expect("Q8_0 token row lookup succeeds");
    assert!(row.iter().all(|&v| (v - 1.0).abs() < 1e-5));
}
