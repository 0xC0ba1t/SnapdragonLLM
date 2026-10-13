import dataclasses
import json

import pytest
import torch

from snapdragon.config.model import ModelConfig
from snapdragon.export.hf import export_hf, qwen3_config, to_qwen3_state_dict
from snapdragon.model.factory import build_model

CFG = ModelConfig(vocab_size=96, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, mlp_hidden=64, max_seq_len=32)


def test_config_fields():
    c = qwen3_config(CFG, bos_id=1, eos_id=2)
    assert c["model_type"] == "qwen3"
    assert c["num_key_value_heads"] == 2
    assert c["head_dim"] == 8
    assert c["tie_word_embeddings"] is True
    assert (c["bos_token_id"], c["eos_token_id"]) == (1, 2)


def test_models_without_qk_norm_cannot_be_exported_as_qwen3():
    with pytest.raises(ValueError, match="QK-norm"):
        qwen3_config(dataclasses.replace(CFG, qk_norm=False))


def test_state_dict_names_shapes_and_total_size():
    m = build_model(CFG)
    sd = to_qwen3_state_dict(m)
    assert sd["model.layers.1.self_attn.q_proj.weight"].shape == (32, 32)
    assert sd["model.layers.1.self_attn.k_proj.weight"].shape == (16, 32)
    assert sd["model.layers.0.mlp.gate_proj.weight"].shape == (64, 32)
    assert sd["model.layers.0.mlp.up_proj.weight"].shape == (64, 32)
    assert "lm_head.weight" not in sd  # tied
    assert sum(t.numel() for t in sd.values()) == sum(p.numel() for p in m.parameters())


def test_qkv_and_gate_up_are_split_in_the_right_order():
    m = build_model(CFG)
    sd = to_qwen3_state_dict(m)
    fused = m.state_dict()["transformer.blocks.0.attn.qkv_proj.weight"]
    assert torch.equal(torch.cat([sd["model.layers.0.self_attn.q_proj.weight"],
                                  sd["model.layers.0.self_attn.k_proj.weight"],
                                  sd["model.layers.0.self_attn.v_proj.weight"]]), fused)


@pytest.mark.parametrize("tie", [True, False])
def test_logits_match_transformers_qwen3(tmp_path, tie):
    transformers = pytest.importorskip("transformers")
    cfg = dataclasses.replace(CFG, tie_embeddings=tie)
    torch.manual_seed(0)
    ours = build_model(cfg).eval()
    export_hf(ours, tmp_path, dtype=torch.float32)
    hf = transformers.AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.float32).eval()
    x = torch.randint(0, 96, (2, 12), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        mine = ours(x)
        theirs = hf(x).logits
    assert torch.allclose(mine, theirs, atol=1e-4), (mine - theirs).abs().max()


def test_export_writes_loadable_files(tmp_path):
    out = export_hf(build_model(CFG), tmp_path)
    assert json.loads((out / "config.json").read_text())["architectures"] == ["Qwen3ForCausalLM"]
    assert (out / "model.safetensors").stat().st_size > 0


def test_greedy_generation_matches_transformers_generate(tmp_path):
    transformers = pytest.importorskip("transformers")
    from snapdragon.inference.generate import generate
    from snapdragon.inference.sampling import SamplingConfig

    torch.manual_seed(0)
    ours = build_model(CFG).eval()
    export_hf(ours, tmp_path, dtype=torch.float32)
    hf = transformers.AutoModelForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.float32).eval()
    prompt = [5, 17, 3, 44, 9]
    mine = generate(ours, prompt, 10, SamplingConfig(temperature=0)).token_ids
    with torch.no_grad():
        theirs = hf.generate(torch.tensor([prompt]), max_new_tokens=10, do_sample=False, pad_token_id=0)[0, 5:].tolist()
    assert mine == theirs  # cached incremental decoding agrees with HF's reference implementation
