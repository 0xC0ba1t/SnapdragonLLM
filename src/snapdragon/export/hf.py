"""Export to the Hugging Face ``Qwen3ForCausalLM`` format.

Our block is architecturally a Qwen3 block: pre-norm RMSNorm, grouped-query attention with per-head
RMSNorm on q and k (QK-norm), rotary embeddings with the half-split convention, SwiGLU MLP, no biases,
optionally tied embeddings. So a snapdragon checkpoint loads in ``transformers`` (and everything built on
it: vLLM, llama.cpp converters, ...) with only a tensor renaming and a split of the fused projections.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from snapdragon.config.model import ModelConfig
from snapdragon.model.lm import CausalLM


def qwen3_config(cfg: ModelConfig, bos_id: int | None = None, eos_id: int | None = None) -> dict:
    if not cfg.qk_norm:
        raise ValueError("Qwen3 always applies QK-norm; export a model trained with qk_norm=True")
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.mlp_hidden,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": cfg.tie_embeddings,
        "use_sliding_window": False,
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "torch_dtype": "bfloat16",
    }


def to_qwen3_state_dict(model: CausalLM) -> dict[str, torch.Tensor]:
    """Rename and split our tensors into the layout ``Qwen3ForCausalLM`` expects."""
    cfg = model.cfg
    sd = model.state_dict()
    q_dim = cfg.n_heads * cfg.head_dim
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    out = {"model.embed_tokens.weight": sd["transformer.embed.weight"],
           "model.norm.weight": sd["transformer.final_norm.weight"]}
    for i in range(cfg.n_layers):
        p = f"transformer.blocks.{i}."
        h = f"model.layers.{i}."
        q, k, v = sd[p + "attn.qkv_proj.weight"].split([q_dim, kv_dim, kv_dim], dim=0)
        gate, up = sd[p + "mlp.gate_up_proj.weight"].chunk(2, dim=0)
        out.update({
            h + "input_layernorm.weight": sd[p + "attn_norm.weight"],
            h + "self_attn.q_proj.weight": q,
            h + "self_attn.k_proj.weight": k,
            h + "self_attn.v_proj.weight": v,
            h + "self_attn.o_proj.weight": sd[p + "attn.o_proj.weight"],
            h + "self_attn.q_norm.weight": sd[p + "attn.q_norm.weight"],
            h + "self_attn.k_norm.weight": sd[p + "attn.k_norm.weight"],
            h + "post_attention_layernorm.weight": sd[p + "mlp_norm.weight"],
            h + "mlp.gate_proj.weight": gate,
            h + "mlp.up_proj.weight": up,
            h + "mlp.down_proj.weight": sd[p + "mlp.down_proj.weight"],
        })
    if not cfg.tie_embeddings:
        out["lm_head.weight"] = sd["transformer.lm_head.weight"]
    return out


def export_hf(
    model: CausalLM, out_dir: str | Path, tokenizer_path: str | Path | None = None,
    bos_id: int | None = None, eos_id: int | None = None, dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Write ``config.json`` + ``model.safetensors`` that ``AutoModelForCausalLM.from_pretrained`` accepts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().to(dtype).cpu().contiguous().clone() for k, v in to_qwen3_state_dict(model).items()}
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(qwen3_config(model.cfg, bos_id, eos_id), indent=2), encoding="utf-8")
    if tokenizer_path is not None:
        import shutil

        shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    return out
